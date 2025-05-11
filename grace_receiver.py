# grace_receiver.py
import argparse, socket, struct, zlib, os, json, time
import cv2, torch, numpy as np
from models import PNC32, PNC32Encoder, ConvLSTM_AE          # same as sender
from matplotlib import pyplot as plt                         # only if you want quick plots
from torchvision.utils import save_image

# ---------- util -----------------------------------------------------------

def load_model(model_name, model_path, device, lstm_kwargs=None):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        if model_name == "pnc32":
            model = PNC32()
        elif model_name == "conv_lstm_PNC32_ae":
            if lstm_kwargs is None:
                raise ValueError("--lstm_kwargs required for ConvLSTM_AE")
            model = ConvLSTM_AE(**lstm_kwargs)
        else:
            raise ValueError(f"Unknown model {model_name}")
        # strip “module.” if trained with DataParallel
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict({k: v for k, v in ckpt.items()
                               if k in model.state_dict()})
    return model.to(device).eval()

def save_rgb(t, outdir, idx):
    os.makedirs(outdir, exist_ok=True)
    save_image(t, os.path.join(outdir, f"frame_{idx:04d}.png"))

HEADER_FMT = "!III"           # frame_idx, feat_idx, payload_len  (3×uint32)
HEADER_SZ  = struct.calcsize(HEADER_FMT)

def parse_header(buf):
    """return frame, index, payload_len"""
    # If you later change to 1‑byte fields, replace HEADER_FMT with "!BBB"
    return struct.unpack_from(HEADER_FMT, buf, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pnc32", "conv_lstm_PNC32_ae"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lstm_kwargs", type=str, default=None)
    ap.add_argument("--ip",   default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--quant", action="store_true",
                    help="Expect 8‑bit quantised features")
    ap.add_argument("--save_dir", default=None,
                    help="Save decoded rgb frames here (optional)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_kwargs = json.loads(args.lstm_kwargs) if args.lstm_kwargs else None
    net = load_model(args.model, args.model_path, device, lstm_kwargs)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    print(f"[receiver] listening on {(args.ip, args.port)}")

    BUFSIZE = 60_000                       # big enough for one UDP datagram
    FEATS   = 32                           # number of feature maps

    current_frame = None                   # frame id we are collecting
    feat_buf      = {}                     # {idx: np.ndarray}
    start_t       = None
    frame_counter = 0

    while True:
        pkt, _addr = sock.recvfrom(BUFSIZE)
        if len(pkt) < HEADER_SZ:
            continue                       # skip garbage

        fid, fidx, paylen = parse_header(pkt)
        compressed       = pkt[HEADER_SZ:HEADER_SZ+paylen]
        payload          = zlib.decompress(compressed)

        # ---------------- new frame begins? ----------------
        if current_frame is None or fid != current_frame:
            # decode the previous frame if we had collected all 32 features
            if feat_buf and len(feat_buf) == FEATS:
                decode_and_show(feat_buf, net, args.quant,
                                device, frame_counter, args.save_dir)
                frame_counter += 1
            # reset for the new frame
            current_frame = fid
            feat_buf      = {}
            start_t       = time.time()

        # store this feature
        feat_dtype = np.uint8 if args.quant else np.float32
        feat       = np.frombuffer(payload, dtype=feat_dtype).reshape(32, 32)
        feat_buf[fidx] = feat

        # If we have all feature‑packets, reconstruct immediately
        if len(feat_buf) == FEATS:
            decode_and_show(feat_buf, net, args.quant,
                            device, frame_counter, args.save_dir)
            dt = (time.time() - start_t)*1e3
            print(f"[receiver] decoded frame {fid} in {dt:.1f} ms")
            frame_counter += 1
            current_frame = None
            feat_buf      = {}

# ---------- reconstruction --------------------------------------------------
def decode_and_show(feat_buf, model, quantised, device, frame_idx, save_dir):
    # order features by index
    feats = np.stack([feat_buf[i] for i in range(32)], axis=0)  # (32,32,32)
    if quantised:
        feats = feats.astype(np.float32) / 255.0
    z = torch.from_numpy(feats).unsqueeze(0).to(device)         # (1,32,32,32)

    with torch.no_grad():
        if hasattr(model, "decoder"):
            rgb = model.decoder(z)      # (1,3,H,W)
        else:
            rgb = model.decode(z)

    rgb = rgb.clamp(0, 1)
    if save_dir:
        save_rgb(rgb, save_dir, frame_idx)

if __name__ == "__main__":
    main()