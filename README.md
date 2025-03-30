# GRACE: Loss-Resilient Real-Time Video through Neural Codecs

## Important License Information


Before using any software or code from this repository, please refer to the LICENSE file for detailed terms and conditions. It is crucial to understand that while the majority of the code in this repository is governed by the Academic Software License © 2023 UChicago

Additionally, please note that the content within the following specific subfolders and files ("baselines/error_concealment", "grace/subnet", and "grace/net.py") did not originate from the authors and may be subject to different terms and conditions, not covered under the provided license. 

## Install the dependencies

**Note: you need to have `git` and `conda` before installation**
```bash
# clone the repo
git clone https://github.com/UChi-JCL/Grace # clone the repo
cd Grace

# install the dependencies
sudo apt install ffmpeg ninja-build # install the ffmpeg and ninja
conda env create -f env.yml # creating the conda environment
```

**Note: you may need to verify the PyTorch installation and reinstall it yourself if there are any PyTorch-related errors**




## Download the model and the test videos

The testing videos can be downloaded at: https://drive.google.com/file/d/1iQhTfb7Kew_z97kDVoj2vOmQqaNjBK9J/view?usp=sharing

The models for Grace can be downloaded at: https://drive.google.com/file/d/1IWD-VUc0RPXXhBzoH5j9YD6bl8kzYYJ1/view?usp=sharing

```bash
# download the models
cp /your/download/path/grace_models.tar
cd models/
tar xzvf grace_models.tar 

# download and extract the videos
cd ../videos/
cp /your/download/path/GraceVideos.zip .
unzip GraceVideos.zip
```


## Running GRACE

```bash
# activate the conda env
conda activate grace-test

# run Grace
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python3 grace-gpu.py

# run pretrained AE models
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python3 pretrained-gpu.py

# run h265/h264 baseline
# NOTE: after running this, you may have to restart your terminal
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python3 h26x.py

# run error concealment
# WARNING: this could take a long time (couple of hours)
LD_LIBRARY_PATH=libs/:${LD_LIBRARY_PATH} python3 error-concealment.py
```

After running the above scripts, you can use the following script to plot the results
```bash
cd results/
python3 plot.py
```
The result will be saved to the pngs in the same folder

Extra Notes (b/c Grace's documentation is poorly documented):
---

### Summary of Video Loss Processing Code

- **Video Structure:**
  - **I-Frame:** The very first frame is an I-frame (decoded independently without loss).
  - **P-Frames:** Every subsequent frame is treated as a P-frame, meaning each relies on a reference frame (the previous decoded frame).

- **Loss Application:**
  - For each test, the code loops over a set of loss values (e.g., 0, 0.1, 0.2, …, 0.8).
  - For each loss value, a **loss array** is created:
    ```python
    loss_arr = [loss] * nframe
    ```
    This produces a list of length `nframe` where every element is the same loss value.

- **Processing by Group (`nframe`):**
  - **When `nframe = 1`:**
    - Each P-frame is processed individually.
    - The reference frame for decoding is the immediately previous decoded frame from `decoded_frames`.
  - **When `nframe = 3` or `nframe = 5`:**
    - Frames are processed in consecutive groups.
    - **First frame in the group:** Uses the decoded frame immediately preceding the group (from `decoded_frames`) as its reference.
    - **Subsequent frames in the group:** Each damaged (loss-applied) frame becomes the reference for the next frame in the group.
    - Although grouped, the loss level is identical for all frames in the group because the loss array is uniform.

- **Key Takeaway:**
  - The grouping (`nframe`) controls how many frames are chained together (and hence how errors might propagate), but it does not vary the loss level applied — every frame in a group gets the same loss value.

---

This summary captures the core points about how the code applies loss and uses reference frames in different group settings.

