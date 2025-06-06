import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import ctypes
import copy
import pandas as pd
import pandas as pd
#from cabac_coder.cabac_coder import CABACCoder, CABACCoderTorchWrapper
import os, sys
from tqdm import tqdm
from grace.grace_gpu_interface import GraceInterface
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image, ImageFile, ImageFilter
# from skimage.metrics import structural_similarity as ssim
import random
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchvision import transforms


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


df_psnr = None

def create_output_dirs(base_dir):
    video_dir = os.path.join(base_dir, "grace_videos")
    frame_dir = os.path.join(base_dir, "grace_frames")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    return video_dir, frame_dir


def resize_to_224(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image)

def PSNR(Y1_raw, Y1_com): # also return MSE for speedup
    # Y1_com = Y1_com.to(Y1_raw.device)
    Y1_raw = resize_to_224(Y1_raw)
    Y1_com = resize_to_224(Y1_com).to(Y1_raw.device)

    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).to(Y1_raw.device)
    train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
    quality = 10.0*torch.log(1/train_mse)/log10
    return float(quality), train_mse.item()

def SSIM(Y1_raw, Y1_com):
    #y1 = Y1_raw.permute([1,2,0]).cpu().detach().numpy()
    #y2 = Y1_com.permute([1,2,0]).cpu().detach().numpy()
    #return ssim(y1, y2, multichannel=True)
    Y1_raw = resize_to_224(Y1_raw).float().cuda().unsqueeze(0)
    Y1_com = resize_to_224(Y1_com).float().cuda().unsqueeze(0)
    return float(ssim(Y1_raw, Y1_com, data_range=1, size_average=False).cpu().detach())


METRIC_FUNC = PSNR

def read_video_into_frames(video_path, frame_size = None, nframes=100): # NOTE: nframes is not used!
    """
    Input:
        video_path: the path to the video
        frame_size: resize the frame to a (width, height), if None, it will not do resize
        nframes: number of frames
    Output:
        frames: a list of PIL images
    """
    def create_temp_path():
        path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        while os.path.isdir(path):
            path = f"/tmp/yihua_frames-{np.random.randint(0, 1000)}/"
        os.makedirs(path, exist_ok=True)
        return path

    def remove_temp_path(tmp_path):
        os.system("rm -rf {}".format(tmp_path))

    frame_path = create_temp_path()
    if frame_size is None:
        cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"
        #cmd = f"ffmpeg -i {video_path} {frame_path}/%03d.png"
    else:
        width, height = frame_size
        cmd = f"ffmpeg -i {video_path} -s {width}x{height} {frame_path}/%03d.png 2>/dev/null 1>/dev/null"

    print(cmd)
    os.system(cmd)
    
    image_names = os.listdir(frame_path)
    frames = []
    for img_name in sorted(image_names): # [:nframes]: NOTE: nframes is not used!
        frame = Image.open(os.path.join(frame_path, img_name))

        ''' pad to nearest 64 for Grace model '''
        padsz = 128
        w, h = frame.size
        pad_w = int(np.ceil(w / padsz) * padsz)
        pad_h = int(np.ceil(h / padsz) * padsz)
 
        frames.append(frame.resize((pad_w, pad_h)))

    print(f"frame path is: {frame_path}")
    print(f"Got {len(image_names)} image names and {len(frames)} frames")
    print("frameSize", len(frames))
    print("Resizing image to", frames[0].size)
    remove_temp_path(frame_path)
    return frames


lib = ctypes.CDLL("libs/bpgenc.so")
lib2 = ctypes.CDLL("libs/bpgdec.so")
bpg_encode_bytes = lib.bpg_encode_bytes
bpg_decode_bytes = lib2.bpg_decode_bytes
get_buf = lib.get_buf
get_buflen = lib.get_buf_length
free_mem = lib.free_memory
get_buf.restype = ctypes.POINTER(ctypes.c_char)
bpg_decode_bytes.restype = ctypes.POINTER(ctypes.c_char)

def bpg_encode(img):
    frame = (torch.clamp(img, min = 0, max = 1) * 255).round().byte()
    _, h, w = frame.shape
    frame2 = frame.permute((1, 2, 0)).flatten()
    bs = frame2.numpy().tobytes()
    # print("bs bytestream/string length is: ", len(bs))
    ubs = (ctypes.c_ubyte * len(bs)).from_buffer(bytearray(bs))
    bpg_encode_bytes(ubs, h, w)
    buflen =  get_buflen()
    buf = get_buf()
    bpg_stream = ctypes.string_at(buf, buflen)
    free_mem(buf)
    # print("bpg_stream length is: ", len(bpg_stream))
    return bpg_stream, h, w, len(bpg_stream)

def bpg_decode(bpg_stream, h, w):
    ub_result = (ctypes.c_ubyte * len(bpg_stream)).from_buffer(bytearray(bpg_stream))
    rgb_decoded = bpg_decode_bytes(ub_result, len(bpg_stream), h, w)
    b = ctypes.string_at(rgb_decoded, h * w * 3)
    bytes = np.frombuffer(b, dtype=np.byte).reshape((h, w, 3))
    image = torch.tensor(bytes).permute((2, 0, 1)).byte().float().cuda()
    image = image / 255
    free_mem(rgb_decoded)
    return image

class IPartFrame:
    def __init__(self, code, shapex, shapey, offset_width, offset_height):
        self.code = code
        self.shapex = shapex # these appear to be unused??
        self.shapey = shapey # these appear to be unused??
        self.offset_width = offset_width
        self.offset_height = offset_height

class EncodedFrame:
    """
    self.code is torch.tensor
    """
    def __init__(self, code, shapex, shapey, frame_type, frame_id):
        self.code = code
        self.shapex = shapex # these appear to be unused??
        self.shapey = shapey # these appear to be unused??
        self.frame_type = frame_type
        self.frame_id = frame_id
        self.loss_applied = False
        self.ipart = None
        self.isize = None
        self.tot_size = None

    def apply_loss(self, loss_ratio, blocksize = 100):
        """
        default block size is 100
        """
        print("apply_loss self.code shape:", self.code.shape)
        leng = torch.numel(self.code)
        nblocks = (leng - 1) // blocksize + 1
        print("leng, nblocks, blocksize:", leng, nblocks, blocksize)

        rnd = torch.rand(nblocks).to(self.code.device)
        rnd = (rnd > loss_ratio).long()
        #print("DEBUG: loss ratio =", loss_ratio, ", first 16 elem:", rnd[:16])
        rnd = rnd.repeat_interleave(blocksize)
        rnd = rnd[:leng].reshape(self.code.shape)
        self.code = self.code * rnd
        
        if self.ipart is not None and np.random.random() < loss_ratio:
            self.ipart = None

    def apply_loss_determ(self, loss_prob):
        REPEATS=64
        nelem = torch.numel(self.code)
        group_len = int((nelem - 1) // REPEATS + 1)
        rnd = torch.rand(group_len).cuda()
        rnd = (rnd > loss_prob).long()
        rnd = rnd.repeat(REPEATS)[:nelem]
        rnd = rnd.reshape(self.code.shape)
        self.code = self.code * rnd

    def apply_mask(self, mask):
        self.code = self.code * mask

    def np_code(self):
        """
        return the code in flattened numpy array
        """
        return self.code.cpu().detach().numpy().flatten()

def find_mn_from_ab(a, b):
    """
    return m, n such that a = mp, b = nq and p > 1, q > 1 and mn = {10, 12, 8, 15, 6}
    """
    mnlist = [(2, 5), (5, 2), (10, 1), (1, 10), 
              (2, 6), (6, 2), (3, 4), (4, 3), (1, 12), (12, 1), 
              (3, 5), (5, 3), (2, 3), (3, 2), (1, 6), (6, 1)]
    for m, n in mnlist:
        if a % m == 0 and a // m > 1 and b % n == 0 and b // n > 1:
            return m, n
    raise RuntimeError(f"No suitable m, n found for a, b = {a}, {b}")

def set_hw_step(h, w):
    """
    returns h_step and w_step
    """
    a, b = h // 64, w // 64
    m, n = find_mn_from_ab(a, b)
    return h // m, w // n

class AEModel:
    def __init__(self, qmap_coder, grace_coder: GraceInterface, only_P=True):
        self.qmap_coder = None 
        self.grace_coder = grace_coder

        self.reference_frame = None
        self.frame_counter = 0
        self.gop = 8

        self.debug_output_dir = None

        self.p_index = 0
        # self.w_step = 256
        # self.h_step = 384
        self.w_step = 128
        self.h_step = 128



    def set_gop(self, gop):
        self.gop = gop

    def encode_ipart(self, frame, no_index_referesh=False):
        """
        Input:
            frame: the PIL image
        Output:
            ipart, isize: encoded frame and it's size, icode is torch.tensor on GPU
        Note:
            this function will NOT update the reference
        """
        c, h, w = frame.shape
        if w % self.w_step != 0 or h % self.h_step != 0:
            raise RuntimeError("w_step and h_step need to divide W and H")
        w_tot = w / self.w_step
        h_tot = h / self.h_step
        w_offset = int((self.p_index % w_tot) * self.w_step)
        h_offset = int(((self.p_index // w_tot) % h_tot) * self.h_step)
        #print(f"P_index = {self.p_index}, w_offset = {w_offset}, h_offset = {h_offset}")


        part_iframe = frame[:, h_offset:h_offset+self.h_step, w_offset:w_offset+self.w_step]
        icode, shapex, shapey, isize = self.qmap_coder.encode(part_iframe)
        ipart = IPartFrame(icode, shapex, shapey, w_offset, h_offset)
        if no_index_referesh == False:
            self.p_index += 1
        
        return ipart, isize

    def encode_frame(self, frame, isIframe = False, no_index_referesh=False):
        """
        Input:
            frame: the PIL image
        Output:
            eframe: encoded frame, code is torch.tensor on GPU
            tot_size: the total size of p rame and I patch
        Note:
            this function will NOT update the reference
        """
        #print("steps:", self.h_step , self.w_step )
        self.frame_counter += 1
        # print("Frame type is: ", frame)
        frame = to_tensor(frame)
        # print("Frame shape is: ", frame.shape)
        # print("isIframe is: ", isIframe)
        if isIframe:
            # torch.cuda.synchronize()
            # start =time.time()
            # code, shapex, shapey, size = self.qmap_coder.encode(frame)
            code, shapex, shapey, size = bpg_encode(frame)
            # torch.cuda.synchronize()
            # end =time.time()
            # print("QMAP TIME SPENT IS: ", (end - start) * 1000)
            eframe = EncodedFrame(code, shapex, shapey, "I", self.frame_counter)
            return eframe, size, eframe
        else: # if it's P-frame
            assert self.reference_frame is not None
            # use p_index to compute which part to encode the I-frame
            c, h, w = frame.shape
            if w % self.w_step != 0 or h % self.h_step != 0:
                raise RuntimeError("w_step and h_step need to divide W and H")

            # torch.cuda.synchronize()

            # encode P part
            # st = time.perf_counter()
            eframe = self.grace_coder.encode(frame, self.reference_frame)

            # torch.cuda.synchronize()
            # ed = time.perf_counter()
            # print("self.grace_coder.encode: ", (ed - st) * 1000)
            # encode I part
            # st = time.perf_counter()
            w_tot = w / self.w_step
            h_tot = h / self.h_step
            w_offset = int((self.p_index % w_tot) * self.w_step)
            h_offset = int(((self.p_index // w_tot) % h_tot) * self.h_step)
            #print(f"P_index = {self.p_index}, w_offset = {w_offset}, h_offset = {h_offset}")
            part_iframe = frame[:, h_offset:h_offset+self.h_step, w_offset:w_offset+self.w_step]
            icode, shapex, shapey, isize = bpg_encode(part_iframe)
            # print("I-part size is: ", isize)    

            # ed = time.perf_counter()
            # print("self.bpg_encode: ", (ed - st) * 1000)
            ipart = IPartFrame(icode, shapex, shapey, w_offset, h_offset)
            eframe.ipart = ipart
            eframe.isize = isize
            eframe.frame_type = "P"
            
            if no_index_referesh == False:
                self.p_index += 1
            # print(eframe.frame_type)
                
            # NOTE: im calling self.grace_coder.entropy_encode() twice, so the internal print statement shows up twice. Just ignore
            encoded_stream, encoded_size = self.grace_coder.entropy_encode(eframe)
            # print("Total size of Entropy Encoded P-frame: ", encoded_size)
            total_p_i_frame_size = encoded_size + isize
            return eframe, total_p_i_frame_size, encoded_stream

    def decode_frame(self, eframe:EncodedFrame):
        """
        Input:
            eframe: the encoded frame (EncodedFrame object)
        Output:
            frame: the decoded frame in torch.tensor (3,h,w) on GPU, which can be used as ref frame
        Note:
            this function will NOT update the reference
        """
        if eframe.frame_type == "I":
            # out = self.qmap_coder.decode(eframe.code, eframe.shapex, eframe.shapey)
            out = bpg_decode(eframe.code, eframe.shapex, eframe.shapey)
            return out
        else:
            assert self.reference_frame is not None
            #out = self.grace_coder.decode(eframe.code, self.reference_frame, eframe.shapex, eframe.shapey)
            # st = time.perf_counter()
            out = self.grace_coder.decode(eframe, self.reference_frame)
            # torch.cuda.synchronize()
            # ed = time.perf_counter()
            # print("self.grace_coder.decode:", (ed - st) * 1000)
            if eframe.ipart is not None:
                ipart = eframe.ipart
                # idec = self.qmap_coder.decode(ipart.code, ipart.shapex, ipart.shapey)
                # st = time.perf_counter()
                idec = bpg_decode(ipart.code, ipart.shapex, ipart.shapey)
                # torch.cuda.synchronize()
                # ed = time.perf_counter()
                # print("self.bpg_decode:", (ed - st) * 1000)

                out[:, ipart.offset_height:ipart.offset_height+self.h_step, ipart.offset_width:ipart.offset_width+self.w_step] = idec
            
            return out


    def encode_video(self, frames, perfect_iframe=False, use_mpeg=True):
        """
        Input:
            frames: PIL images
        Output:
            list of METRIC_FUNC and list of BPP
        """
        import grace.net
        grace.net.DEBUG_USE_MPEG = True
        bpps = []
        psnrs = []
        test_iter = tqdm(frames)
        dec_frames = []
        for idx, frame in enumerate(test_iter):
            # encode the frame
            if idx % self.gop == 0:
                ''' I FRAME '''
                if perfect_iframe:
                    self.update_reference(to_tensor(frame))
                    bpps.append(0)
                    psnrs.append(99)

                    dec_frames.append(to_tensor(frame)) # for ffmpeg psnr calculation
                else:
                    eframe, size = self.encode_frame(frame, "I")
                    decoded = self.decode_frame(eframe)
                    self.update_reference(decoded)

                    dec_frames.append(decoded) # for ffmpeg psnr calculation

                    # compute bpp
                    w, h = frame.size
                    bpp = size * 8 / (w * h)
                    bpps.append(bpp)

                    # compute psnr
                    tframe = to_tensor(frame)
                    psnr = float(METRIC_FUNC(tframe, decoded))
                    psnrs.append(psnr)

                    print("IFRAME: bpp =", bpp, "PSNR =", psnr)

            else:
                # eframe, z = self.encode_frame(frame)
                eframe, tot_size = self.encode_frame(frame)
                print("P FRAME: eframe size = ", tot_size)

                # decode frame
                w, h = frame.size
                decoded = self.decode_frame(eframe)

                dec_frames.append(to_tensor(frame)) # for ffmpeg psnr calculation
                self.update_reference(decoded)

                # compute psnr
                tframe = to_tensor(frame)
                psnr = float(METRIC_FUNC(tframe, decoded))
                psnrs.append(psnr)


                # compute bpp
                ''' whole frame compression '''
                # bs, tot_size = self.entropy_coder.entropy_encode(eframe.code, \
                                        # eframe.shapex, eframe.shapey, z)
                # tot_size = 
                w, h = frame.size
                tot_size += eframe.isize
                bpp = tot_size * 8 / (w * h)
                print("Total # of Bytes/Size = {}, Frame id = {}, P bpp = {}, I part bpp = {}".format(tot_size, idx, (tot_size - eframe.isize) * 8 / (w * h), eframe.isize * 8 / (w * h)))
                bpps.append(bpp)

            test_iter.set_description(f"bpp:{np.mean(bpps):.4f}, psnr:{np.mean(psnrs):.4f}")

        assert len(dec_frames) == len(frames)
        return psnrs, bpps



    def update_reference(self, ref_frame):
        """
        Input:
            ref_frame: reference frame in torch.tensor with size (3,h,w). On GPU
        """
        self.reference_frame = ref_frame

    def fit_frame(self, frame):
        """
        set the h_step and w_step for the encoder
        frame is a PIL image
        """
        w, h = frame.size
        self.h_step, self.w_step = set_hw_step(h, w)

    def get_avg_freeze_psnr(self, frames):
        res = []
        for idx, frame in enumerate(frames[2:]):
            img1 = to_tensor(frame)
            img2 = to_tensor(frames[idx-2])
            res.append(METRIC_FUNC(img1, img2))
        return float(np.mean(res))



def init_ae_model(qmap_quality=1):
    qmap_config_template = {
            "N": 192,
            "M": 192,
            "sft_ks": 3,
            "name": "default",
            "path": "models/qmap/qmap_pretrained.pt",
            "quality": qmap_quality,
        }
    qmap_coder = None #QmapModel(qmap_config_template)

    GRACE_MODEL = "models/grace"
    models = {
            "64": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/64_freeze.model"}, scale_factor=0.25)),
            "128": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/128_freeze.model"}, scale_factor=0.5)),
            "256": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/256_freeze.model"}, scale_factor=0.5)),
            "512": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/512_freeze.model"}, scale_factor=0.5)),
            "1024": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/1024_freeze.model"}, scale_factor=0.5)),
            "2048": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/2048_freeze.model"})),
            # "4096": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/4096_freeze.model"})),
            # "6144": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/6144_freeze.model"})),
            # "8192": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/8192_freeze.model"})),
            # "12288": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/12288_freeze.model"})),
            # "16384": AEModel(qmap_coder, GraceInterface({"path": f"{GRACE_MODEL}/16384_freeze.model"})),
            }

    return models


def encode_frame(ae_model: AEModel, is_iframe, ref_frame, new_frame, no_index_referesh=False):
    """
    ref_frame: torch tensor C, H, W
    new_frame: PIL image

    returns:
        size in bytes 
        the eframe
    """
    if ref_frame is not None:
        ae_model.update_reference(ref_frame)
    else:
        if not is_iframe:
            raise RuntimeError("Cannot encode a P-frame without reference frame")

    eframe, size, entropy_encoded_eframe = ae_model.encode_frame(new_frame, is_iframe)
    return size, eframe, entropy_encoded_eframe

def decode_frame(ae_model: AEModel, eframe: EncodedFrame, ref_frame, loss):
    """
    ref_frame: the tensor frame in 3, h, w

    returns:
        decoded frame
    """
    if ref_frame is not None:
        ae_model.update_reference(ref_frame)
    else:
        if not eframe.frame_type == "I":
            raise RuntimeError("Cannot decode a P-frame without reference frame")
        
    if eframe.frame_type == "I":
        if loss > 0:
            print("Error! Cannot add loss on I frame, it will cause huge error!")
        decoded = ae_model.decode_frame(eframe)
        return decoded
    else: # if P-frame
        print("Type of eframe:", type(eframe)) # type is GraceBasicCode
        eframe.apply_loss(loss) 
        ae_model.update_reference(ref_frame)
        decoded = ae_model.decode_frame(eframe)
        return decoded

def encode_whole_video(frames, ae_model: AEModel, output_dir="results/grace"):
    """
    Input:
        frames: a list of frames in PIL format
    Return:
        orig_frames: list of frames in torch.Tensor
        codes: list of EncodedFrame
        dec_frames: list of decoded frame in torch.Tensor
    """
    orig_frames = list(map(to_tensor, frames))
    codes = []
    dec_frames = []
    ref_frame = None

    for idx, frame in enumerate(frames):
        size, eframe, _ = encode_frame(ae_model, idx == 0, ref_frame, frame)
        eframe.tot_size = size
        print("eframes total size is: ", eframe.tot_size, "\n")
        decoded_frame = decode_frame(ae_model, eframe, ref_frame, 0)

        # save_image(decoded_frame, os.path.join(frame_dir, f"decoded-{idx:04d}.png"))

        codes.append(eframe)
        dec_frames.append(decoded_frame)
        ref_frame = decoded_frame

    # Create video from decoded frames
    # cmd = f"ffmpeg -y -framerate 30 -i {frame_dir}/decoded-%04d.png -c:v libx264 -pix_fmt yuv420p {video_dir}/output.mp4"
    # os.system(cmd)

    return orig_frames, codes, dec_frames


def decode_with_loss(ae_model: AEModel, frame_id, losses, decoded_frames, eframes):
    """
    Input:
        frame_id: encode starting from xxx frame, should be larger than 1
        losses: list of loss values, the length determines how many frames will be decoded
        decoded_frames: the global decoded frames from encode_whole_video(), read-only
        eframes: the global eframes array from encode_whole_video(), read-only
    returns:
        damaged_frames: the list of damaged frames
    """
    damaged = []
    ref_frame = decoded_frames[frame_id - 1]
    for idx, loss in enumerate(losses):
        eframe = copy.deepcopy(eframes[frame_id + idx]) if frame_id + idx < len(eframes) else eframes[-1]
        damaged_frame = decode_frame(ae_model, eframe, ref_frame, loss)
        print("type:", type(damaged_frame))
        damaged.append(damaged_frame)
        ref_frame = damaged_frame
    # print("losses and damaged length:", losses, damaged, len(damaged))
    return damaged



models = init_ae_model()

def run_one_model(model_id, input_pil_frames, video_id=0, video_name=""):
    dfs = [] # size, psnr, ssim, loss, frame_id

    df = pd.DataFrame()
    model = models[model_id]
    model.p_index = 0

    # Create video-level directory
    video_dir = os.path.join("results/grace", f"{video_name.split('/')[-1][:-4]}")
    os.makedirs(video_dir, exist_ok=True)

    # Create model directory inside the video directory
    model_dir = os.path.join(video_dir, f"model_{model_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Create subdirectories for original and decoded frames
    orig_dir = os.path.join(model_dir, "orig_frames")
    decoded_dir = os.path.join(model_dir, "decoded")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(decoded_dir, exist_ok=True)

    orig_frames, codes, dec_frames = encode_whole_video(input_pil_frames, model, model_dir) # NOTE: original line of code 
   
    # NOTE Sanity check: 
    i_frame = codes[0]
    if i_frame.frame_type == "I":
        byte_size = i_frame.tot_size
        print("Verified: This is the I-frame.")
        print("I-frame byte size (in bytes):", byte_size)
    else:
        print("Warning: The first frame is not an I-frame!")

    # pnsrs, bpps = model.encode_video(input_pil_frames, perfect_iframe=False, use_mpeg=True) # NOTE: I added this

    # Save original and decoded frames
    # for idx, (orig, decoded) in enumerate(zip(orig_frames, dec_frames)):
        # save_image(orig, os.path.join(orig_dir, f"orig-{idx:04d}.png"))
        # save_image(decoded, os.path.join(decoded_dir, f"decoded-{idx:04d}.png"))

    # Create videos from saved frames
    # os.system(f"ffmpeg -y -r {10} -i {orig_dir}/orig-%04d.png -c:v libx264 {orig_dir}/output.mp4")
    # os.system(f"ffmpeg -y -r {10} -i {decoded_dir}/decoded-%04d.png -c:v libx264 {decoded_dir}/output.mp4")

    ### NOTE: COMMENT out all writing/saving to .csv files
    sizes = [code.tot_size for code in codes]
    psnrs = [PSNR(o, d)[0] for o, d in zip(orig_frames, dec_frames)]
    ssims = [SSIM(o, d) for o, d in zip(orig_frames, dec_frames)]
    mses = [PSNR(o, d)[1] for o, d in zip(orig_frames, dec_frames)]
    frame_ids = np.arange(0, len(input_pil_frames))
    df["size"] = sizes
    df["mse"] = mses
    df["psnr"] = psnrs
    df["ssim"] = ssims
    df["loss"] = 0
    df["frame_id"] = frame_ids
    df["nframes"] = 0
    #print(df)
    dfs.append(df)

    def run_multi_frame_losses(nframe, total_frames):
        dfs = []
        print("  - Running consecutive loss nframe =", nframe)
        for loss in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.88]:
            loss_dir = os.path.join(model_dir, f"loss_{loss:.1f}_nframes={nframe}")
            os.makedirs(loss_dir, exist_ok=True)

            damaged_frames = []
            df = pd.DataFrame()
            loss_arr = [loss] * nframe
            # print("total_frames, nframe:", total_frames, nframe)
            for frame_id in range(1, total_frames, nframe):
                # print(" - Running frame_id =", frame_id)
                damaged = decode_with_loss(model, frame_id, loss_arr, dec_frames, codes)
                damaged_frames.extend(damaged)

                # Save damaged frames
                # print("length of damaged and damaged_frames:", nframe, len(damaged), len(damaged_frames))
                # for idx, damaged_frame in enumerate(damaged):
                    # save_image(damaged_frame, os.path.join(loss_dir, f"damaged-{frame_id+idx:04d}.png"))

            # Create damaged video
            # os.system(f"ffmpeg -y -r {10} -i {loss_dir}/damaged-%04d.png -c:v libx264 {loss_dir}/output.mp4")

            df["size"] = [eframe.tot_size for eframe in codes[1:]]
            df["mse"] = [PSNR(o, d)[1] for o, d in zip(orig_frames[1:], damaged_frames)]
            df["psnr"] = [PSNR(o, d)[0] for o, d in zip(orig_frames[1:], damaged_frames)]
            df["ssim"] = [SSIM(o, d) for o, d in zip(orig_frames[1:], damaged_frames)]
            df["loss"] = loss
            df["frame_id"] = np.arange(1, total_frames)
            df["nframes"] = nframe
            dfs.append(df)
        return pd.concat(dfs)
    
    dfs += [run_multi_frame_losses(1, len(input_pil_frames))] # instead of 16, use the total number of frames in video (denoted by len(input_pil_frames))!
    dfs += [run_multi_frame_losses(3, len(input_pil_frames))]
    dfs += [run_multi_frame_losses(5, len(input_pil_frames))]
    # return None 

    #run_multi_frame_losses(3, 16)
    #run_multi_frame_losses(5, 16)
    final_df = pd.concat(dfs)
    return final_df

def run_one_video(video, video_id):
    input_frames = read_video_into_frames(video, nframes=120) # NOTE: nframes is currently not used!
    
    dfs = []
    for model_id in models.keys():
        print("  Running model:", model_id)
        df = run_one_model(model_id, input_frames, video_id, video)
        # run_one_model(model_id, input_frames, video_id, video)
        df["model_id"] = model_id
        dfs.append(df)

    # return None
    # NOTE: COMMENT out all writing/saving to .csv files
    final_df = pd.concat(dfs)
    return final_df

def run_one_file(index_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    videos = []
    with open(index_file, "r") as fin:
        for line in fin:
            stripped = line.strip("\n")
            if stripped and not stripped.startswith('#'):
                videos.append(line.strip("\n"))

    print("Videos list: ", videos)

    for idx, video in enumerate(videos):
        print(f"\033[33mRunning video: {video}, index: {idx}\033[0m")
        run_one_video(video, idx)
    # return None

    ### NOTE: COMMENT out all writing/saving to .csv files 
    video_dfs = []
    for idx, video in enumerate(videos):
        print(f"\033[33mRunning video: {video}, index: {idx}\033[0m")
        video_basename = os.path.basename(video)
        if False and os.path.exists(f"{output_dir}/{video_basename}_v_resized224.csv"):
            print(f"Skip the finished video: {video}")
            video_df = pd.read_csv(f"{output_dir}/{video_basename}_v_resized224.csv")
        else:
            video_df = run_one_video(video, idx)
            video_df["video"] = video_basename
            video_df.to_csv(f"{output_dir}/{video_basename}_v_resized224.csv", index=None)
            video_dfs.append(video_df)

    final_df = pd.concat(video_dfs)
    final_df.to_csv(f"{output_dir}/all.csv", index=None)
    return final_df

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    run_one_file("INDEX.txt", "results/grace")
