# grace_sender.py
import socket, struct, zlib, pickle, argparse
from grace_gpu_new_version import init_ae_model, encode_frame, decode_frame
from PIL import Image
import torch
import numpy as np
import cv2
import os, time

def split_into_chunks(data: bytes, n_chunks: int) -> list:
    """Splits data into n_chunks of nearly equal size."""
    chunk_size = len(data) // n_chunks
    return [data[i * chunk_size: (i + 1) * chunk_size] if i < n_chunks - 1 else data[i * chunk_size:]
            for i in range(n_chunks)]

def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to video file")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--npackets", type=int, default=32, help="Number of packets to split each frame into")
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
        size, eframe = encode_frame(model, frame_idx == 0, ref_tensor, pil_img)

        # Serialize + compress
        raw = pickle.dumps(eframe)
        # raw = serialize_eframe(eframe)
        comp = zlib.compress(raw)

        # Split into npackets chunks
        chunks = split_into_chunks(comp, args.npackets)
        print("chunks:", len(chunks), "total size:", len(comp))

        for packet_id, chunk in enumerate(chunks):
            header = struct.pack("!IIII", frame_idx, packet_id, args.npackets, len(chunk))
            # print("size of entire packet: ", len(header) + len(chunk))
            sock.sendto(header + chunk, dest)

        end_time = time.monotonic_ns() / 1e6
        print(f"Sent frame {frame_idx} ({size} bytes, {len(comp)} compressed) in {end_time - start_time:.2f} ms")

        # NOTE: SANITY CHECK: save all frames, simulating receiver
        if frame_idx == 0:
            save_img(ref_tensor, "grace_sender_frames/", frame_idx)
        else:
            # Decode the received data
            # eframe = deserialize_eframe(raw)
            # decoded_img = model.decode(eframe)
            decoded_img_0_loss = decode_frame(model, eframe, ref_tensor, loss=0.0)
            decoded_img = decode_frame(model, eframe, ref_tensor, loss=50.0)
            ref_tensor = decoded_img_0_loss
            save_img(decoded_img, "grace_sender_frames/", frame_idx)
        frame_idx += 1

    sock.close()

if __name__ == "__main__":
    main()
