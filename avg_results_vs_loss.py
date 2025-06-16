import pandas as pd, os
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
GRACE_CSV = "grace_full_simulator.csv"
# PNC_CSV   = "../LRAE-VC/PNC_results_w_taildrops_quant_on_REAL_TUCF.csv"
PNC_CSV = "../LRAE-VC/PNC_results_w_taildrops_quant_on_UCF_high.csv"
# PNC_CSV = "../LRAE-VC/PNC_results_w_taildrops_NOQUANT_on_UCF_high.csv"

# CASTR_CSV = "../LRAE-VC/CASTR_results_single_convlstm_drop32_quant_on_UCF_high.csv"
CASTR_CSV = "../LRAE-VC/CASTR_results_all_motion_multi_convlstm_drop32_quant_on_UCF_high.csv"
# CASTR_CSV = "../LRAE-VC/CASTR_results_all_motion_multi_convlstm_drop32_NOQUANT_on_UCF_high.csv"


### NOTE: below is for specified testing!
# PNC_CSV = "../LRAE-VC/PNC_results.csv"
# CASTR_CSV = "../LRAE-VC/CASTR_results.csv"

os.makedirs("./new_plots/quality_vs_loss", exist_ok=True)
os.makedirs("./new_plots/performance_vs_frame_num", exist_ok=True)

# Map nframes to labels
def burst_label(n):
    return f"{n}-frame burst" if n != 1000 else "all-frames burst"

# Tail-to-% mapping for PNC
tail_to_percent = {
    # 0:  0.0, 
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

# --- LOAD DATA ---
grace_df = pd.read_csv(GRACE_CSV)
pnc_df   = pd.read_csv(PNC_CSV)
castr_df = pd.read_csv(CASTR_CSV)

# Exclude I-frame entries (nframes=0)
videos = ["Kicking-Front003", "Golf-Swing-Front005"] # to include
grace_df = grace_df[(grace_df['nframes'] != 0)] # & (grace_df['video'].isin(videos))]
# Normalize loss to percentage if in [0,1]
if grace_df['loss'].max() <= 1.0:
    grace_df['loss'] *= 100

# PNC: map tail_len_drop --> loss
pnc_df['loss'] = pnc_df['tail_len_drop'].map(tail_to_percent)
castr_df['loss'] = castr_df['tail_len_drop'].map(tail_to_percent)
# grace_df = grace_df[grace_df['loss'] >= 50]  # Filter GRACE to only include losses >= 50%

# --- AGGREGATE ---
# GRACE: group by nframes, model_id, loss → mean mse & ssim
grace_grp = (
    grace_df
    .groupby(['nframes','model_id','loss'])
    .agg({'mse':'mean','ssim':'mean'})
    .reset_index()
)
# PNC: treat as 1-frame burst
grp_pnc = pnc_df.groupby('loss').agg({'MSE':'mean','SSIM':'mean'}).reset_index()

castr_df = castr_df[castr_df['consecutive'].isin([1, 5])].copy()
castr_df.columns = castr_df.columns.str.strip()
castr_df.rename(columns={
    'consecutive': 'nframes'}, inplace=True) # in CASTR, change 'conseuctive' to 'nframes' for simplicity and consistency (NOTE: doesn't change original CASTR code! :)

# Compute PNC PSNR from MSE only once
grp_pnc['psnr'] = 10 * np.log10(1.0 / grp_pnc['MSE'])
grp_castr = (
    castr_df
    .groupby(['nframes', 'loss'])
    .agg({'mse':'mean','ssim':'mean'})
    .reset_index()
)

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})

# NOTE: --- PLOTTING MSE/SSIM/PSNR vs. Packet loss, generating one graph for each nframes ---
unique_n = sorted(grace_grp['nframes'].unique())
for n in unique_n:
    subset = grace_grp[grace_grp['nframes']==n]
    plt.figure(figsize=(8,6))
    # plot each model_id
    for mid in subset['model_id'].unique():
        d = subset[subset['model_id']==mid]
        psnr = 10 * np.log10(1.0 / d['mse'])
        plt.plot(d['loss'], psnr, label=f"GRACE model {mid}")
    # add PNC only in 1-frame burst
    plt.plot(grp_pnc['loss'], grp_pnc['psnr'], label='PNC')
    castr_subset = grp_castr[grp_castr['nframes'] == n]
    if not castr_subset.empty:
        psnr = 10 * np.log10(1.0 / castr_subset['mse'])
        plt.plot(castr_subset['loss'], psnr, label='CASTR')

    plt.title(f"PSNR vs. Packet Loss ({burst_label(n)})")
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('PSNR (dB)')
    plt.ylim(16, 35)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_loss/psnr_Ipatch_sent_every_{n}_frames.pdf')
    plt.close()

    # SSIM plot
    plt.figure(figsize=(8,6))
    for mid in subset['model_id'].unique():
        d = subset[subset['model_id']==mid]
        plt.plot(d['loss'], d['ssim'], label=f"GRACE model {mid}")
    plt.plot(grp_pnc['loss'], grp_pnc['SSIM'], label='PNC')
    if not castr_subset.empty:
        plt.plot(castr_subset['loss'], castr_subset['ssim'], label='CASTR')
    plt.title(f"SSIM vs. Packet Loss ({burst_label(n)})")
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('SSIM')
    plt.ylim(0,1)
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_loss/sim_Ipatch_sent_every_{n}_frames.pdf')
    plt.close()

    # MSE plot
    plt.figure(figsize=(8,6))
    for mid in subset['model_id'].unique():
        d = subset[subset['model_id']==mid]
        plt.plot(d['loss'], d['mse'] * 256, label=f"GRACE model {mid}")
    # Add PNC (mean squared error from grp_pnc)
    plt.plot(grp_pnc['loss'], grp_pnc['MSE'] * 256, label='PNC')
    # Add CASTR if available
    if not castr_subset.empty:
        plt.plot(castr_subset['loss'], castr_subset['mse'] * 256, label='CASTR')
    plt.title(f"MSE vs. Packet Loss ({burst_label(n)})")
    plt.xlabel('Packet Loss (%)')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./new_plots/quality_vs_loss/mse_Ipatch_sent_every_{n}_frames.pdf')
    plt.close()
