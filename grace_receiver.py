import argparse, os, socket, struct, zlib, time, pickle, shutil
import torch
import numpy as np
from grace_gpu_new_version import init_ae_model, decode_frame
from PIL import Image
from collections import defaultdict

def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ip",   default="0.0.0.0")
    parser.add_argument("--deadline_ms", type=float, default=13000)
    parser.add_argument("--output", type=str, default="receiver_frames_grace")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = init_ae_model()["4096"]
    model.set_gop(8)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 2**21)
    sock.bind((args.ip, args.port))
    sock.settimeout(1.0)
    print(f"[receiver] Listening on {args.ip}:{args.port}")

    cur_frame = 0
    ref_frame = None
    deadline_ms = args.deadline_ms

    packet_buffer = defaultdict(dict)     # frame_idx -> {packet_id: bytes}
    frame_timestamps = {}                # frame_idx -> timestamp

    while True:
        try:
            pkt, _ = sock.recvfrom(65536)

            if len(pkt) < 16:
                print("[receiver] Incomplete header, skipping.")
                continue

            frame_idx, packet_id, packet_count, data_len = struct.unpack("!IIII", pkt[:16])
            data = pkt[16:]

            if len(data) != data_len:
                print(f"[receiver] Malformed packet: got {len(data)} bytes, expected {data_len}")
                continue

            if frame_idx not in frame_timestamps:
                frame_timestamps[frame_idx] = time.monotonic_ns() / 1e6  # in ms

            packet_buffer[frame_idx][packet_id] = data

            if len(packet_buffer[frame_idx]) == packet_count:
                now = time.monotonic_ns() / 1e6
                delay = now - frame_timestamps[frame_idx]

                if delay > deadline_ms:
                    print(f"[receiver:drop] Frame {frame_idx} exceeded deadline ({delay:.2f} ms), dropping.")
                    del packet_buffer[frame_idx]
                    del frame_timestamps[frame_idx]
                    continue

                print(f"[receiver] Assembling frame {frame_idx} after {delay:.2f} ms")

                all_data = b"".join([packet_buffer[frame_idx][i] for i in range(packet_count)])
                raw = zlib.decompress(all_data)
                eframe = pickle.loads(raw)

                start_decode = time.monotonic() * 1000
                recon = decode_frame(model, eframe, ref_frame, loss=0.0) # TODO: change/update this loss value! 
                end_decode = time.monotonic() * 1000

                print(f"[receiver] Decoded frame {frame_idx} in {end_decode - start_decode:.2f} ms")

                save_img(recon, args.output, frame_idx)
                ref_frame = recon.detach()

                del packet_buffer[frame_idx]
                del frame_timestamps[frame_idx]
                cur_frame = frame_idx + 1

        except socket.timeout:
            # Optional: flush incomplete old frames here
            continue
        except Exception as e:
            print(f"[receiver:error] {e}")
            continue

if __name__ == "__main__":
    main()
