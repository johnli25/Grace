import argparse, socket, struct, zlib, os, time
import numpy as np
import torch

# Grace helper -------------------------------------------------------------
from grace_gpu_new_version import init_ae_model, decode_frame   # same module as sender
from grace_gpu_new_version import EncodedFrame, IPartFrame      # classes defined there
from grace.grace_gpu_interface import GraceBasicCode
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = init_ae_model()        # load once
model  = models["128"]          # must match the sender

# packet helpers -----------------------------------------------------------
HDR_P  = struct.Struct("!BBBBBB")     # P‑block   (6 bytes)
HDR_I  = struct.Struct("!BBBBI")      # I‑frame / I‑patch (8 bytes)

BLOCK_H, BLOCK_W = 8, 4              # NOTE: keep in sync with sender! 
C, H, W = 32, 32, 32

I_FULL   = 0  
P_BLOCK  = 1  
I_PATCH  = 2  

# NOTE: hardcoded shapex, shapey, z for grace.grace_gpu_interface.GraceBasicCode for P-frame for now!
shapex_ = ([1, 128, 8, 8])
shapey_ = ([1, 96, 16, 16])
z_ = torch.zeros((1, 64, 4, 4), dtype=torch.float32, device=device) # NOTE: dummy z_ for now

def parse_p_frame(pkt):
    frame_idx, num_blocks, blk_idx, _type, i, j = HDR_P.unpack_from(pkt)
    payload = pkt[HDR_P.size:]
    return frame_idx, num_blocks, blk_idx, i, j, payload

def parse_i_frame(pkt):
    """Parse I-frame packet and return the frame data."""
    frame_idx, _, _, _type, paylen = HDR_I.unpack_from(pkt)
    payload = pkt[HDR_I.size:HDR_I.size + paylen]   
    return frame_idx, _type, payload


def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))


class FrameBuf:
    def __init__(self, frame_idx, deadline_ms=1000):
        self.frame_idx = frame_idx
        self.num_blocks_expected = 32
        self.latent = torch.zeros(C, H, W, dtype=torch.float32, device=device) # you can change to torch.float16 if yo uwant
        self.blocks_received = np.zeros((self.num_blocks_expected,), dtype=bool) # tracks which blocks have been received thus far
        self.ipart = None # (bytes, i, j) if present
        self.iframe = None # bytes, only if frame_idx == 0

        self.start_time_ms = time.time() * 1000
        self.deadline_ms = deadline_ms
        self.decoded = False

    def add_pblock(self, n_blocks, blk_idx, i, j, payload):
        if self.num_blocks_expected is None:
            self.num_blocks_expected = n_blocks
        block = np.frombuffer(zlib.decompress(payload), dtype=np.float32).reshape(C, BLOCK_H, BLOCK_W)
        self.latent[:, i:i+BLOCK_H, j:j+BLOCK_W] = torch.from_numpy(block).to(device)
        self.blocks_received[blk_idx] = True
    
    def add_iframe(self, payload):
        self.iframe = payload
    
    def add_ipart(self, payload):
        self.ipart = payload
    
    def ready_to_decode(self):
        # always decode I-frames immediately--not conditioned on deadlines
        if self.frame_idx == 0:
            return self.iframe is not None
        # for P-frames: decode if EITHER
        #  1) we got _all_ expected blocks, OR
        #  2) we’ve hit the deadline
        all_here = (self.num_blocks_expected is not None
                    and self.blocks_received.sum() == self.num_blocks_expected)
        timed_out = (time.time() * 1000 - self.start_time_ms) > self.deadline_ms
        return all_here or timed_out


def decode_framebuf(buf : FrameBuf, ref_tensor: torch.Tensor):
    """Decode the frame buffer using the reference tensor."""
    if buf.frame_idx == 0:
        # I-frame
        eframe = EncodedFrame(code=buf.iframe, 
                              shapex=256, shapey=256, # NOTE: shapex and shapey are apparently not used in EncodedFrame in grace_gpu_new_version.py
                              frame_type="I",
                              frame_id=buf.frame_idx)
        rgb = decode_frame(model, eframe, None, loss=0) 
        return rgb
    
    # build EncodedFrame for P-frame
    

    ##### Save buf.latent to a .txt file with its 3D structure
    latent_np = buf.latent.cpu().numpy()  # Convert to NumPy array
    with open("receiver_latent_3d.txt", "w") as f:
        for channel in range(latent_np.shape[0]):  # Iterate over channels (C)
            f.write(f"Channel {channel}:\n")
            np.savetxt(f, latent_np[channel], fmt="%.16f")  # Save each 2D slice (H x W)
            f.write("\n")  # Add a blank line between channels
    print("3D latent tensor saved to receiver_latent_3d.txt")
    ##### 

    buf.latent = buf.latent.view(C * H * W) # reshape buf.latent from C, H, W to C * H * W 

    # latent_np = buf.latent.cpu().numpy()  # Convert to NumPy array
    # with open("almost_flattened_receiver_latent_3d.txt", "w") as f:
    #     for channel in range(latent_np.shape[0]):  # Iterate over channels (C)
    #         f.write(f"Channel {channel}:\n")
    #         np.savetxt(f, latent_np[channel], fmt="%.6f")  # Save each 2D slice (H x W)
    #         f.write("\n")  # Add a blank line between channels
    # print("3D latent tensor saved to receiver_latent_3d.txt")

    eframe = GraceBasicCode(code=buf.latent,
                            shapex=shapex_,
                            shapey=shapey_,
                            z=z_,)    
    if buf.ipart is not None:
        # add I-part
        ipart = IPartFrame(code=buf.ipart,
                    shapex=128, shapey=128, # NOTE: shapex and shapey are apparently not used in EncodedFrame in grace_gpu_new_version.py
                    offset_width=0, offset_height=0)
        eframe.ipart = ipart
        
    eframe.frame_type = "P" # dynamically set frame type NOTE: bit of a hack, but it works
    print(f"[receiver] eframe code and ref tensor shapes and type: {eframe.code.shape}, {ref_tensor.shape}, {type(eframe)}, {eframe.frame_type} ")
    recon = decode_frame(model, eframe, ref_tensor, loss=0)
    return recon


def main():
    parser = argparse.ArgumentParser(description="GRACE receiver")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deadline_ms", type=float, default=1000, help="Deadline for receiving packets")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    print(f"Listening on {args.ip}:{args.port}")
    buffer_map = {}
    ref_rgb = None
    frame_cnt = 0

    os.makedirs("grace_receiver_frames/", exist_ok=True)

    while True:
        pkt, addr = sock.recvfrom(65536)
        if len(pkt) < HDR_I.size: # could be either header
            print("Packet too small lol!")
            continue
        # else
        type_byte = pkt[3] # 4th unsigned char in both headers
        if type_byte in (I_FULL, I_PATCH):
            frame_idx, type_, payload = parse_i_frame(pkt)
            fb = buffer_map.setdefault(frame_idx, FrameBuf(frame_idx)) # .setdefault() creates a new FrameBuf if not present OR returns the existing one
            print(f"Received I-frame {frame_idx} of type {type_}")
            if type_ == I_FULL:
                fb.add_iframe(payload)
            if type_ == I_PATCH: 
                fb.add_ipart(payload)
        else:
            frame_idx, n_blocks, blk_idx, i, j, payload = parse_p_frame(pkt)
            print(f"Received P-block {blk_idx} for frame {frame_idx} at ({i}, {j})")
            fb = buffer_map.setdefault(frame_idx, FrameBuf(frame_idx))
            fb.add_pblock(n_blocks, blk_idx, i, j, payload)

        # ready to decode yet?
        fb = buffer_map[frame_idx]
        if not fb.decoded and fb.ready_to_decode():
            t0 = time.time() * 1000
            if ref_rgb is not None: print(f"[receiver] fb.latent and ref_rgb:", fb.latent.shape, ref_rgb.shape)
            recon = decode_framebuf(fb, ref_rgb) # NOTE: if fb.frame_idx == 0, ref_rgb is None anyway, and will decode an I-frame
            print(f"[receiver] Decoded frame {frame_idx} in {time.time() * 1000 - t0:.2f}ms and received {fb.blocks_received.sum()}/{fb.num_blocks_expected} blocks")

            save_img(recon, "grace_receiver_frames/", frame_idx)
            frame_cnt += 1

            # update reference frame
            ref_rgb = recon
            fb.decoded = True

            # drop old buffer to save/recover memory
            del buffer_map[frame_idx]
        
if __name__ == "__main__":
    main()
