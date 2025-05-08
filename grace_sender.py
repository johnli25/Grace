# grace_sender.py
import socket, struct, zlib, pickle, argparse
from grace_gpu_new_version import init_ae_model, encode_frame, decode_frame
from PIL import Image
import torch
import numpy as np
import cv2
import os, time
import random


def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))

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
    BLOCK_SIZE = 32 * 32333333333 # 32x32 = 1024

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
        total_bytes_sent = 0

        # Prepare for sending
        if eframe.frame_type == "I":
            # Step 1)
            print("frame_idx", frame_idx, "is an I-frame")
            # eframe.code is a bytes object
            raw_bytes = eframe.code                 # already a BPG bytestream
            compressed_payload = zlib.compress(raw_bytes)
            # pack it as a single “I-frame code” packet (type=2)
            header = struct.pack("!IIIII",
                            frame_idx,   # which frame
                            1,           # exactly one packet
                            0,           # packet index 0
                            2,           # custom type=2 --> I-frame code
                            len(compressed_payload))
            sock.sendto(header + compressed_payload, dest)
            total_bytes_sent += len(compressed_payload)
            print(f"Sent I-frame {frame_idx} in {total_bytes_sent} bytes")

        else: # P-frame
            # Step 1.5) 
            code_flat = eframe.code.flatten()
            n_blocks = (code_flat.numel() - 1) // BLOCK_SIZE + 1
            code_blocks = code_flat.split(BLOCK_SIZE)

            # Step 2) Encode subsequent P-frame as blocks
            print("code blocks:", len(code_blocks))
            for blk_idx, block in enumerate(code_blocks):
                payload = pickle.dumps(block) 
                compressed_payload = zlib.compress(payload)

                header = struct.pack("!IIIII", 
                                    frame_idx, 
                                    n_blocks,
                                    blk_idx, 
                                    0, # type 0 = P-frame code-block
                                    len(compressed_payload)) 
                sock.sendto(header + compressed_payload, dest)
                total_bytes_sent += len(compressed_payload)

            # Step 3) send I-part as its own packet 
            if eframe.ipart is not None:
                ipayload = pickle.dumps(eframe.ipart) 
                icompressed_payload = zlib.compress(ipayload)
                header = struct.pack("!IIIII", 
                                    frame_idx, 
                                    1,
                                    0, # block index 0 for I-part
                                    1, # type 1 = I-part
                                    len(icompressed_payload))
                sock.sendto(header + icompressed_payload, dest)
                # print(f"Sent I-part for frame {frame_idx} in {len(icompressed_payload)} bytes")
                total_bytes_sent += len(icompressed_payload) 
            # print(f"Sent {len(code_blocks)} P-frame blocks for frame {frame_idx} in {total_bytes_sent} bytes")

        end_time = time.monotonic_ns() / 1e6
        print(f"Sent frame {frame_idx} with {total_bytes_sent} bytes in {end_time - start_time:.6f} ms")

        # NOTE: OPTIONAL SANITY CHECK: save all frames, simulating receiver
        if frame_idx == 0:
            save_img(ref_tensor, "grace_sender_frames/", frame_idx)
        else:
            loss_ratio = random.random()       # float in [0.0, 1.0)
            # decode under that simulated loss
            recon = decode_frame(model, eframe, ref_tensor, loss=0)
            ref_tensor = recon.detach() # update reference for the next P‐frame
            save_img(recon, "grace_sender_frames/", frame_idx)

        frame_idx += 1

    sock.close()

if __name__ == "__main__":
    main()
