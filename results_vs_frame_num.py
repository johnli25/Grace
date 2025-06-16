import pandas as pd, os
import matplotlib.pyplot as plt
import numpy as np

grace_df = pd.read_csv("grace_full_simulator.csv")
# pnc_df   = pd.read_csv("../LRAE-VC/PNC32_Diving_results_per_frame_NOQUANT_on_UCF_high.csv")
# castr_df = pd.read_csv("../LRAE-VC/CASTR_Diving_per_frame_all_motion_convlstm_drop32_NOQUANT_on_UCF_high.csv")
pnc_df = pd.read_csv("../LRAE-VC/PNC_Diving_per_frame_results.csv")
castr_df = pd.read_csv("../LRAE-VC/CASTR_Diving_per_frame_results.csv")
CASTR_CONSECUTIVE = 9 # Due to edge case 
CONSECUTIVE = 10 # For GRACE and PNC 
tail_to_percent = {
    0:  0.0, 
    3: 10.0, 
    6: 20.0, 
    10: 30.0,
    13: 40.0,
    16: 50.0,
    19: 60.0,
    22: 70.0,
    26: 80.0,
    28: 90.0,
    29: 91.0,
    30: 94.0,
    31: 97.0
}


# NOTE: for now, these have already been filtered for the diving video! Also, convert to loss % 
pnc_df['loss'] = pnc_df['tail_len_drop'].map(tail_to_percent)
castr_df['loss'] = castr_df['tail_len_drop'].map(tail_to_percent)
grace_df['loss'] = grace_df['loss'] * 100

# 1) Filter GRACE for our single video *and* only 10‐frame bursts
grace_single = grace_df[
    (grace_df['video']=="Diving-Side001") &
    (grace_df['nframes']==CONSECUTIVE)
].copy()

# 2) PNC: still use the full table (it has no nframes column)
pnc_single = pnc_df.copy()
print(f"PNC DF single video shape: {pnc_single.shape}")
print(f"PNC DF first 5 rows:\n{pnc_single.head()}")

# 3) GRACE should be automatically grouped by loss, model_id, frame_num
grace_grp_single = grace_single
print(f"GRACE DF single video shape: {grace_grp_single.shape}")
print(f"GRACE DF first 5 rows:\n{grace_grp_single.head()}")

# 4) CASTR
castr_df = castr_df[castr_df['consecutive'] == CASTR_CONSECUTIVE]
print(f"CASTR DF single video shape: {castr_df.shape}")
print(f"CASTR DF first 5 rows:\n{castr_df.head()}")

packet_losses = sorted(grace_grp_single['loss'].unique())
print(f"All losses in GRACE single video: {packet_losses}")

for loss_pct in packet_losses:
    grace_slice = grace_grp_single[grace_grp_single['loss']==loss_pct]
    if grace_slice.empty:
        print("How is Grace's data empty??")
        continue

    x_ticks = [frame for frame in grace_slice['frame_num'] if frame % 10 == 0]  # Only include every 10th frame


    # MSE vs frame_num
    plt.figure(figsize=(8,6))
    # 1) PNC
    pnc_loss = pnc_single[pnc_single['loss']==loss_pct]
    print("pnc_loss for loss_pct", loss_pct, "head:", pnc_loss.head())
    if not pnc_loss.empty:
        plt.plot(
            pnc_loss['frame_num'], 
            pnc_loss['mse'], 
            label="PNC", 
            color='orange', 
            linestyle='--'
        )

    # 2) CASTR
    castr_loss = castr_df[castr_df['loss']==loss_pct]
    print("castr_loss for loss_pct", loss_pct, "head:", castr_loss.head())
    if not castr_loss.empty:
        plt.plot(
            castr_loss['frame_num'], 
            castr_loss['mse'], 
            label="CASTR", 
            color='green', 
            linestyle='--'
        )

    # 3) GRACE
    for mid in grace_slice['model_id'].unique():
        d = grace_slice[grace_slice['model_id']==mid]
        plt.plot(d['frame_num'], d['mse'], label=f"GRACE {mid}")

    plt.title(f"MSE vs Frame Number @ {loss_pct:.0f}% Loss (10-frame bursts)")
    plt.xlabel("Frame Number")
    plt.ylabel("MSE")       
    plt.xticks(x_ticks)  # Set x-axis ticks to every 10th frame
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_framenum/mse_vs_framenum_n10_loss_{int(loss_pct)}.pdf')
    plt.close()


    # PSNR vs frame_num
    plt.figure(figsize=(8,6))

    # 1) PNC
    if not pnc_loss.empty:
        pnc_psnr = 10 * np.log10(1.0 / pnc_loss['mse'])
        plt.plot(
            pnc_loss['frame_num'], 
            pnc_psnr, 
            label="PNC", 
            color='orange', 
            linestyle='--'
        )

    # 2) CASTR
    if not castr_loss.empty:
        castr_psnr = 10 * np.log10(1.0 / castr_loss['mse'])
        plt.plot(
            castr_loss['frame_num'], 
            castr_psnr, 
            label="CASTR", 
            color='green', 
            linestyle='--'
        )

    # 3) GRACE
    for mid in grace_slice['model_id'].unique():
        d = grace_slice[grace_slice['model_id']==mid]
        psnr_vals = 10 * np.log10(1.0 / d['mse'])
        plt.plot(d['frame_num'], psnr_vals, label=f"GRACE {mid}")

    plt.title(f"PSNR vs Frame Number @ {loss_pct:.0f}% Loss (10-frame bursts)")
    plt.xlabel("Frame Number")
    plt.ylabel("PSNR (dB)")
    plt.xticks(x_ticks)  # Set x-axis ticks to every 10th frame
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_framenum/psnr_vs_framenum_n10_loss_{int(loss_pct)}.pdf')
    plt.close()

    # SSIM vs frame_num
    plt.figure(figsize=(8,6))
    # 1) PNC
    if not pnc_loss.empty:
        plt.plot(
            pnc_loss['frame_num'], 
            pnc_loss['ssim'], 
            label="PNC", 
            color='orange', 
            linestyle='--'
        )

    # 2) CASTR
    if not castr_loss.empty:
        plt.plot(
            castr_loss['frame_num'], 
            castr_loss['ssim'], 
            label="CASTR", 
            color='green', 
            linestyle='--'
        )

    # 3) GRACE
    for mid in grace_slice['model_id'].unique():
        d = grace_slice[grace_slice['model_id']==mid]
        plt.plot(d['frame_num'], d['ssim'], label=f"GRACE {mid}")

    plt.title(f"SSIM vs Frame Number @ {loss_pct:.0f}% Loss (10-frame bursts)")
    plt.xlabel("Frame Number")
    plt.ylabel("SSIM")
    plt.ylim(0, 1)
    plt.xticks(x_ticks)  # Set x-axis ticks to every 10th frame
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_framenum/ssim_vs_framenum_n10_loss_{int(loss_pct)}.pdf')
    plt.close()