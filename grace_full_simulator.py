#!/usr/bin/env python3
import os, csv, cv2, torch, time, numpy as np, random
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
from grace_gpu_new_version import init_ae_model, encode_frame, decode_frame

# ------------------------------------------------------------------ config ---
TEST_VIDEOS = {
    "Diving-Side001":           "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Diving-Side001.mp4",
    "Golf-Swing-Front005":      "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Golf-Swing-Front005.mp4",
    "Kicking-Front003":         "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Kicking-Front003.mp4",
    "Lifting002":               "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Lifting002.mp4",
    "Riding-Horse006":          "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Riding-Horse006.mp4",
    "Run-Side001":              "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Run-Side001.mp4",
    "SkateBoarding-Front003":   "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/SkateBoarding-Front003.mp4",
    "Swing-Bench016":           "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Swing-Bench016.mp4",
    "Swing-SideAngle006":       "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Swing-SideAngle006.mp4",
    "Walk-Front021":            "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Walk-Front021.mp4",
}

MODEL_SIZES = [128, 256, 512, 1024] 
# MODEL_SIZES = [16384]
LOSS_RATES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# BURST_NS    = [1, 3, 5, 10, 20, 1000]                     # 1000 ≅ “all P-frames lossy”
BURST_NS = [3] # forgot about this
INPUT_SIZE  = (256, 256)

OUT_DIR  = "grace_full_simulator"
CSV_FILE = "grace_full_simulator_testcopy.csv"

# ---------------------------------------------------------------- utils ------
def save_png(t, path):
    """t: (3,H,W) tensor in [0,1]  →  PNG"""
    arr = (t.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")

def pil2tensor(pil):
    return torch.from_numpy(np.asarray(pil, np.float32)/255.).permute(2,0,1)

def drop_bytes_random(bytestream: bytes, loss_rate: float) -> bytes:
    if loss_rate <= 0.0 or len(bytestream) == 0:
        return bytestream

    keep = int(len(bytestream) * (1.0 - loss_rate))
    if keep <= 0:
        return b''

    # sample indices without replacement, keep order
    idx = sorted(random.sample(range(len(bytestream)), keep))
    return bytes(bytestream[i] for i in idx)

def drop_bytes_head(bs, loss_rate):
    keep = int(len(bs) * (1.0 - loss_rate))
    return bs[:keep]

# --------------------------------------------------------------- main --------
if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load all GRACE checkpoints once
    grace_models = init_ae_model()                      # returns dict str(size) → model
    for sz in MODEL_SIZES:
        grace_models[str(sz)].set_gop(8)                # keep encoder settings identical

    # prepare CSV
    fieldnames = ["size","mse","psnr","ssim","loss",
                  "frame_num","nframes","model_id","video"]
    csv_rows = []

    # 2) iterate videos
    for vid_name, vid_path in TEST_VIDEOS.items():
        # --- read all video frames into RAM (keeps loop simple) -----------
        cap, frames_bgr = cv2.VideoCapture(vid_path), []
        while True:
            ok, fr = cap.read()
            if not ok: break
            frames_bgr.append(fr)
        cap.release()
        total_frames = len(frames_bgr)

        # -------------------------------------------------------------------
        for size in MODEL_SIZES:
            model = grace_models[str(size)]
            for loss_rate in LOSS_RATES:
                for nburst in BURST_NS:
                    out_dir = os.path.join(
                        OUT_DIR, vid_name,
                        f"model_{size}",
                        f"loss_{loss_rate:.1f}_n={nburst}"
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    ref = None  # reference RGB tensor
                    # -------- per-frame pass --------------------------------
                    for idx, fr_bgr in enumerate(frames_bgr):
                        pil = Image.fromarray(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)).resize(INPUT_SIZE)
                        gt  = pil2tensor(pil).to(device)                # ground-truth tensor

                        if idx == 0:                                    # I-frame only once
                            bytes_size, eframe, whatisthis = encode_frame(model, True,  None, pil) 
                            print("eframe type:" ,type(eframe.code))

                            # eframe.code = drop_bytes_head(eframe.code, loss_rate)
                            # bytes_size = len(eframe.code) 

                            out = decode_frame(model, eframe, None, loss=0.0)
                            ref = out.detach()                          # set reference once
                        else:                                           # P-frame
                            bytes_size, eframe, whatisthis = encode_frame(model, False, ref, pil)

                            # drop I-part only when idx % nburst != 0
                            if (idx % nburst) != 0:
                                eframe.ipart = None

                            out = decode_frame(model, eframe, ref, loss=loss_rate)
                            ref = out.detach()                          # chain reference

                        # save PNG
                        save_png(out, os.path.join(out_dir, f"{vid_name}_frame_{idx:04d}.png"))

                        # metrics
                        mse  = F.mse_loss(out, gt).item()
                        psnr = -10.0 * np.log10(mse) if mse > 0 else float("inf")
                        ssim_val = ssim(out.unsqueeze(0), gt.unsqueeze(0),
                                        data_range=1.0, size_average=False).item()

                        csv_rows.append({
                            "size":      size,
                            "mse":       mse,
                            "psnr":      psnr,
                            "ssim":      ssim_val,
                            "loss":      loss_rate,
                            "frame_num": idx,
                            "nframes":   nburst,
                            "model_id":  f"model_{size}",
                            "video":     vid_name,
                        })

                        print(f"{vid_name} | model{size} | loss={loss_rate:.1f} | n={nburst}"
                              f" | frame={idx:04d}/{total_frames}  →  "
                              f"MSE={mse:.4e}  PSNR={psnr:.2f}  SSIM={ssim_val:.3f}")

    # 3) write CSV
    with open(CSV_FILE, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        # writer.writeheader() # NOTE: comment this out if need to append to existing CSV
        writer.writerows(csv_rows)

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
    print(f"\nAll done  →  metrics written to {CSV_FILE}")
