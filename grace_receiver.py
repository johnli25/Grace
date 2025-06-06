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
model  = models["1024"]          # NOTE: must match the sender

# packet helpers -----------------------------------------------------------
HDR_P  = struct.Struct("!BBBB") 
HDR_I  = struct.Struct("!BBBI") 

I_FULL   = 0  
P_BLOCK  = 1  
I_PATCH  = 2  

# NOTE: hardcoded shapex, shapey, z for grace.grace_gpu_interface.GraceBasicCode for P-frame for now!
shapex_ = ([1, 128, 8, 8])
shapey_ = ([1, 96, 16, 16])
pframe_latent_shape = ([32768])
z_ = torch.zeros((1, 64, 4, 4), dtype=torch.float32, device=device) # NOTE: dummy z_ for now

def parse_p_frame(pkt):
    frame_idx, blk_idx, _type, num_blocks = HDR_P.unpack_from(pkt)
    payload = pkt[HDR_P.size:]
    return frame_idx, num_blocks, blk_idx, payload

def parse_i_frame(pkt):
    """Parse I-frame packet and return the frame data."""
    frame_idx, _, type_, paylen = HDR_I.unpack_from(pkt)
    payload = pkt[HDR_I.size:HDR_I.size + paylen]
    print(f"[receiver] Received I-frame {frame_idx} of type {type_}")
    return frame_idx, type_, payload


def save_img(rgb_tensor, outdir, idx):
    img_array = rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(os.path.join(outdir, f"frame_{idx:04d}.png"))


class FrameBuf:
    def __init__(self, frame_idx, pframe_latent_shape, deadline_ms=1000):
        self.frame_idx = frame_idx
        self.num_pframe_pkts = 32
        # NOTE: self.latent_pframe will NOT MATTER if frame_idx == 0 (I-frame), but we still need to initialize it
        self.latent_pframe = torch.zeros(pframe_latent_shape, dtype=torch.float32, device=device) # you can change to torch.float16 if yo uwant
        self.blocks_received = [False] * self.num_pframe_pkts
        self.ipart = None 
        self.iframe = None # bytes, only if frame_idx == 0

        self.start_time_ms = time.time() * 1000
        self.deadline_ms = deadline_ms
        self.decoded = False

    def add_pblock(self, n_pframe_pkts, blk_pkt_i, payload):
        self.num_pframe_pkts = n_pframe_pkts
        BLOCK_SIZE = self.latent_pframe.shape[0] // n_pframe_pkts # e.g. BLOCK_SIZE = 32768 / 32 = 1024
        block = np.frombuffer(zlib.decompress(payload), dtype=np.float32)

        self.latent_pframe[blk_pkt_i * BLOCK_SIZE : (blk_pkt_i + 1) * BLOCK_SIZE] = torch.tensor(block, dtype=torch.float32, device=device)
        self.blocks_received[blk_pkt_i] = True
    
    def add_iframe(self, payload):
        self.iframe = payload
    
    def add_ipart(self, payload):
        self.ipart = payload
    
    def ready_to_decode(self):
        # always decode I-frames immediately--not conditioned on deadlines
        if self.frame_idx == 0:
            return self.iframe is not None
        # for P-frames: decode if EITHER
        #  1) we got ALL expected blocks OR
        #  2) we’ve hit the deadline
        all_here = (sum(self.blocks_received) == self.num_pframe_pkts)  # NOTE: keep all_here if you want to decode when you receive all P-frame blocks 
        ipart_received = True if self.ipart else False # NOTE: keep ipart_received if you want to decode when you receive the I-part
        timed_out = (time.time() * 1000 - self.start_time_ms) > self.deadline_ms
        return all_here or timed_out


def decode_framebuf(buf : FrameBuf, ref_tensor: torch.Tensor):
    """Decode the frame buffer using the reference tensor."""
    if buf.frame_idx == 0:
        # I-frame
        eframe = EncodedFrame(code=buf.iframe, 
                              shapex=256, shapey=256, # NOTE: shapex and shapey are apparently not used in EncodedFrame in grace_gpu_new_version.py (not 100% sure)
                              frame_type="I",
                              frame_id=buf.frame_idx)
        rgb = decode_frame(model, eframe, None, loss=0.0) 
        return rgb

    # P-frame:
    eframe = GraceBasicCode(code=buf.latent_pframe,
                            shapex=shapex_,
                            shapey=shapey_,
                            z=z_,)    
    if buf.ipart is not None:
        print("[receiver] HIT/Adding I-part to P-frame")
        # add I-part
        ipart = IPartFrame(code=buf.ipart,
                    shapex=128, shapey=128, # NOTE: shapex and shapey are apparently not used in EncodedFrame in grace_gpu_new_version.py
                    offset_width=0, offset_height=0)
        eframe.ipart = ipart
        
    eframe.frame_type = "P" # dynamically set frame type NOTE: bit of a hack, but it works
    print(f"[receiver] eframe code and ref tensor shapes and type: {eframe.code.shape}, {ref_tensor.shape}, {type(eframe)}, {eframe.frame_type} ")
    if buf.frame_idx % 5 == 0:
        recon = decode_frame(model, eframe, ref_tensor, loss=0.0)
    else:
        recon = decode_frame(model, eframe, ref_tensor, loss=0.0)
    return recon


def main():
    parser = argparse.ArgumentParser(description="GRACE receiver")
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--deadline_ms", type=float, default=1000, help="Deadline for receiving packets")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))

    # NOTE: manually update socket buffer sizes!
    DESIRED_BUF_SIZE = 2**21  # 1 MiB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, DESIRED_BUF_SIZE)

    rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"[socket] RCVBUF size: {rcvbuf} bytes")

    print(f"Listening on {args.ip}:{args.port}")
    buffer_map = {}
    ref_rgb = None
    frame_cnt = 0

    os.makedirs("grace_receiver_frames/", exist_ok=True)

    while True:
        pkt, addr = sock.recvfrom(65536)
        type_byte = pkt[2] # 4th unsigned char in both headers
        if type_byte in (I_FULL, I_PATCH):
            frame_idx, type_, payload = parse_i_frame(pkt)
            print("payload length:", len(payload))
            fb = buffer_map.setdefault(frame_idx, FrameBuf(frame_idx=frame_idx, pframe_latent_shape=pframe_latent_shape, deadline_ms=args.deadline_ms)) # .setdefault() creates a new FrameBuf if not present OR returns the existing one
            print(f"Received I-frame OR I-patch for {frame_idx} of type {type_}")
            if type_ == I_FULL:
                fb.add_iframe(payload)
            if type_ == I_PATCH: 
                fb.add_ipart(payload)
        else:
            frame_idx, n_blocks, blk_idx, payload = parse_p_frame(pkt)
            # print(f"Received P-block {blk_idx}/{n_blocks} for frame {frame_idx}")
            fb = buffer_map.setdefault(frame_idx, FrameBuf(frame_idx=frame_idx, pframe_latent_shape=pframe_latent_shape, deadline_ms=args.deadline_ms)) # .setdefault() creates a new FrameBuf if not present OR returns the existing one
            fb.add_pblock(n_blocks, blk_idx, payload)

        # ready to decode yet?
        fb = buffer_map[frame_idx]
        if not fb.decoded and fb.ready_to_decode():
            t0 = time.time() * 1000
            if ref_rgb is not None: print(f"[receiver] fb.latent and ref_rgb:", fb.latent_pframe.shape, ref_rgb.shape)
            recon = decode_framebuf(fb, ref_rgb) # NOTE: if fb.frame_idx == 0, ref_rgb is None anyway, and will decode an I-frame
            print(f"[receiver] Decoded frame {frame_idx} in {time.time() * 1000 - t0:.2f}ms and received {sum(fb.blocks_received)}/{fb.num_pframe_pkts} blocks")
            save_img(recon, "grace_receiver_frames/", frame_idx)
            frame_cnt += 1

            # update reference frame
            ref_rgb = recon
            fb.decoded = True

            # drop old buffer to save/recover memory
            del buffer_map[frame_idx]
        
if __name__ == "__main__":
    main()
