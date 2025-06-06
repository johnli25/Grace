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

# def split_into_blocks(tensor, block_height=4, block_width=4):
#     """Splits (C, H, W) tensor into spatial (i, j) blocks of shape (C, B, B)"""
#     C, H, W = tensor.shape
#     blocks = []
#     for i in range(0, H, block_height):
#         for j in range(0, W, block_width):
#             block = tensor[:, i:i+block_height, j:j+block_width].clone().contiguous()
#             blocks.append((i, j, block))
#     return blocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to video file")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    # 1) Init model
    models = init_ae_model()
    model = models["1024"]
    model.set_gop(52)

    # 2) Open video & socket
    cap = cv2.VideoCapture(args.input)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    DESIRED_BUF_SIZE = 2**21
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, DESIRED_BUF_SIZE)

    sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    print(f"[socket] SNDBUF size: {sndbuf} bytes")

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
        total_bytes_sent = 0

        # Prepare for sending
        if eframe.frame_type == "I":
            # Step 1)
            print("frame_idx", frame_idx, "is an I-frame")
                # eframe.code is a bytes object
            raw_bytes = eframe.code # already a BPG bytestream, and so doing zlib.compress() is not necessary (it will make it worse!)
            # pack it as a single “I-frame code” packet (type=2)
            header = struct.pack("!BBBI",
                            frame_idx,   # which frame
                            0,           # packet index 0
                            I_FULL,      # Type=0 for I-frame code block
                            len(raw_bytes)) # Total bytes in I-frame 
            sock.sendto(header + raw_bytes, dest)

            total_bytes_sent += len(raw_bytes)
            print(f"Sent I-frame {frame_idx} in {total_bytes_sent} bytes")

        else: # P-frame
            BLOCK_SIZE = eframe.code.shape[0] // 32  # 32 blocks per P-frame
            for i in range(32):
                # print("[sender] elements: ", i * BLK_SIZE, (i + 1) * BLK_SIZE)
                block = eframe.code[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]  # Extract block
                block_bytes = block.cpu().numpy().astype(np.float32).tobytes()
                compressed = zlib.compress(block_bytes)
                header = struct.pack("!BBBB", 
                                    frame_idx,
                                    i,  # Block/packet index
                                    P_BLOCK,  # Type = 1 for P-frame code block
                                    32)  # Total number of blocks planned to be sent
                sock.sendto(header + compressed, dest)
                total_bytes_sent += len(compressed)
            
            # Step 3) send I-part as its own packet 
            if eframe.ipart is not None:
                raw_bytes = eframe.ipart.code # already a BPG bytestream, and so doing zlib.compress() is not necessary (it will make it worse!)
                header = struct.pack("!BBBI",
                                frame_idx,   # which frame
                                0,           # packet index 0
                                I_PATCH,      # Type=0 for I-frame code block
                                len(raw_bytes)) # Total bytes in I-part 
                sock.sendto(header + raw_bytes, dest)
                total_bytes_sent += len(raw_bytes)
                print(f"Sent I-part of P-frame {frame_idx} in {len(raw_bytes)} bytes")

            # print("[sender] type eframe", type(eframe))
            # print("[sender] type eframe.code", type(eframe.code))
            # print("[sender] eframe.code shape", eframe.code.shape)
            # print("[sender] eframe.code shapex", eframe.shapex)
            # print("[sender] eframe.code shapey", eframe.shapey)
            # print("[sender] eframe.code z", eframe.z.shape)

        end_time = time.monotonic_ns() / 1e6
        print(f"Sent frame {frame_idx} (0 = I-frame, else P-frame) with TOTAL OF {total_bytes_sent} bytes in {end_time - start_time:.6f} ms")

        # NOTE: OPTIONAL SANITY CHECK: save all frames on sender, simulating receiver
        eframe.ipart = None # NOTE: ignore I-parts for now for simplicity LOL
        recon = decode_frame(model, eframe, ref_tensor, loss=0)  
        save_img(recon, "grace_sender_frames/", frame_idx)
        ref_tensor = recon.detach()  # update reference for the next P-frame

        # if frame_idx == 0:
        #     save_img(ref_tensor, "grace_sender_frames/", frame_idx)
        # else: 
        #     loss_ratio = random.random()       # float in [0.0, 1.0)
        #     # print(f"[sender] eframe code and ref tensor shapes and type: {eframe.code.shape}, {ref_tensor.shape}, {type(eframe)}, {eframe.frame_type} ")
        #     recon = decode_frame(model, eframe, ref_tensor, loss=0)
        #     ref_tensor = recon.detach() # update reference for the next P‐frame
        #     save_img(recon, "grace_sender_frames/", frame_idx)

        frame_idx += 1

    sock.close()

if __name__ == "__main__":
    main()

    # NOTE: quick smoke test about pickle's size 
    # t = torch.arange(32768, dtype=torch.)
    # view = t[0:1024]                 # slice
    # print(sys.getsizeof(pickle.dumps(view)))  # ~65 kB, not 2 kB!
