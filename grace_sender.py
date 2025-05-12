# grace_sender.py
import socket, struct, zlib, pickle, argparse
from grace_gpu_new_version import init_ae_model, encode_frame, decode_frame
from PIL import Image
import torch
import numpy as np
import cv2
import os, time
import random
import torchac
import sys

I_FULL   = 0  
P_BLOCK  = 1  
I_PATCH  = 2 

def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))

def split_into_blocks(tensor, block_height=4, block_width=4):
    """Splits (C, H, W) tensor into spatial (i, j) blocks of shape (C, B, B)"""
    C, H, W = tensor.shape
    blocks = []
    for i in range(0, H, block_height):
        for j in range(0, W, block_width):
            block = tensor[:, i:i+block_height, j:j+block_width].clone().contiguous()
            blocks.append((i, j, block))
    return blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to video file")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    # 1) Init model
    models = init_ae_model()
    model = models["1024"]
    model.set_gop(8)

    # 2) Open video & socket
    cap = cv2.VideoCapture(args.input)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.ip, args.port)

    frame_idx = 0
    INPUT_SIZE = (256, 256)

    os.makedirs("grace_sender_frames/", exist_ok=True)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert to RGB & PIL, resize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize(INPUT_SIZE)

        start_time = time.monotonic_ns() / 1e6

        if frame_idx == 0:
            ref_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        # Encode using GRACE
        size, eframe, entropy_encoded_eframe = encode_frame(model, frame_idx == 0, ref_tensor, pil_img) # frame_idx == 0 means: if frame_idx == 0 AKA -> I-frame --> True; else --> P-frame = false 
        if frame_idx == 0:
            print("I-frame size:", size)
        else: # P-frame
            print("P-frame size and entropy_encoded_eframe size:", size, len(entropy_encoded_eframe))
            print("P-eframe.code size:", eframe.code.size(), eframe.shapex, eframe.shapey, eframe.z.size())
        total_bytes_sent = 0

        # Prepare for sending
        if eframe.frame_type == "I":
            # Step 1)
            print("frame_idx", frame_idx, "is an I-frame")
                # eframe.code is a bytes object
            raw_bytes = eframe.code # already a BPG bytestream, and so doing zlib.compress() is not necessary (it will make it worse!)
            # pack it as a single “I-frame code” packet (type=2)
            header = struct.pack("!BBBBI",
                            frame_idx,   # which frame
                            1,           # exactly one packet
                            0,           # packet index 0
                            I_FULL,           # custom type=0 --> I-frame code
                            len(raw_bytes))
            sock.sendto(header + raw_bytes, dest)

            total_bytes_sent += len(raw_bytes)
            print(f"Sent I-frame {frame_idx} in {total_bytes_sent} bytes")

        else: # P-frame
            print("[sender] P-frame latent shape", eframe.code.shape)
            latent = eframe.code.view(32, 32, 32)  # (C, H, W)
            blocks = split_into_blocks(latent, block_height=32, block_width=32) # (i, j, block)
            n_blocks = len(blocks)

            print("n_blocks", n_blocks)
            for blk_idx, (i, j, block) in enumerate(blocks):
                block_bytes = block.cpu().numpy().astype(np.float32).tobytes()
                compressed = zlib.compress(block_bytes)
                header = struct.pack("!BBBBBB", 
                                    frame_idx,
                                    n_blocks,
                                    blk_idx,
                                    P_BLOCK,        # type = 1 for P-frame code block
                                    i, j) # i = starting row of the sub-block, j = starting column of the sub-block
                # print("len of compressed, header, and total", len(compressed), len(header), len(header) + len(compressed), "blk_idx", blk_idx, "i, j", i, j)
                sock.sendto(header + compressed, dest)
                total_bytes_sent += len(compressed)

            # Step 3) send I-part as its own packet 
            if eframe.ipart is not None:
                raw_bytes = eframe.ipart.code # already a BPG bytestream, and so doing zlib.compress() is not necessary (it will make it worse!)
                header = struct.pack("!BBBBI",
                                frame_idx,   # which frame
                                1,           # exactly one packet
                                0,           # packet index 0
                                I_PATCH,           # type=2 --> I-part/patch
                                len(raw_bytes))
                sock.sendto(header + raw_bytes, dest)
                total_bytes_sent += len(raw_bytes)
                print(f"Sent I-part/patch {frame_idx} in {len(raw_bytes)} bytes")

            print("[sender] type eframe", type(eframe))
            print("[sender] type eframe.code", type(eframe.code))
            print("[sender] eframe.code shape", eframe.code.shape)
            print("[sender] eframe.code shapex", eframe.shapex)
            print("[sender] eframe.code shapey", eframe.shapey)
            print("[sender] eframe.code z", eframe.z.shape)

        end_time = time.monotonic_ns() / 1e6
        print(f"Sent frame {frame_idx} with TOTAL OF {total_bytes_sent} bytes in {end_time - start_time:.6f} ms")

        # NOTE: OPTIONAL SANITY CHECK: save all frames, simulating receiver
        if frame_idx == 0:
            save_img(ref_tensor, "grace_sender_frames/", frame_idx)
        else: 
            loss_ratio = random.random()       # float in [0.0, 1.0)
            print(f"[sender] eframe code and ref tensor shapes and type: {eframe.code.shape}, {ref_tensor.shape}, {type(eframe)}, {eframe.frame_type} ")
            eframe.code.view(32 * 32 * 32)
            recon = decode_frame(model, eframe, ref_tensor, loss=0)
            ref_tensor = recon.detach() # update reference for the next P‐frame
            save_img(recon, "grace_sender_frames/", frame_idx)

        frame_idx += 1

    sock.close()

if __name__ == "__main__":
    main()

    # NOTE: quick smoke test about pickle's size 
    # t = torch.arange(32768, dtype=torch.)
    # view = t[0:1024]                 # slice
    # print(sys.getsizeof(pickle.dumps(view)))  # ~65 kB, not 2 kB!
