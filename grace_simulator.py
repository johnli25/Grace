import os
import cv2
import torch
import numpy as np
from PIL import Image
from grace_gpu_new_version import init_ae_model, encode_frame, decode_frame
import argparse

def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", required=True, help="path to video file")
    # parser.add_argument("--outdir", required=True, help="directory to save reconstructed frames")
    input_dir = "../LRAE-VC/TUCF_sports_action_224x224_mp4_vids/Diving-Side001.mp4"
    output_dir = "grace_simulator_frames"
    args = parser.parse_args()

    models = init_ae_model()
    model = models["1024"]
    model.set_gop(8)

    cap = cv2.VideoCapture(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    INPUT_SIZE = (256, 256)
    ref_tensor = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).resize(INPUT_SIZE)

        if frame_idx == 0:
            ref_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0

        _, eframe, _ = encode_frame(model, frame_idx == 0, ref_tensor, pil_img)
        eframe.ipart = None  # NOTE: ignore I-part for now
        if frame_idx == 0:
            save_img(ref_tensor, output_dir, frame_idx)
        else:
            recon = decode_frame(model, eframe, ref_tensor, loss=0.0)
            save_img(recon, output_dir, frame_idx)
            ref_tensor = recon.detach()

        frame_idx += 1

    cap.release()

if __name__ == "__main__":
    main()
