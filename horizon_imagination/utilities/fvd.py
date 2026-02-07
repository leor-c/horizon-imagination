"""
This implementation was copy-pasted from 
https://github.com/JunyaoHu/common_metrics_on_video_quality
with slight modifications (added asserts, merged files, etc.)

Specifically, I have merged 
https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/calculate_fvd.py
and 
https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/styleganv/fvd.py
"""

import torch
import os
import math
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from typing import Tuple, Literal
from scipy.linalg import sqrtm

from loguru import logger

# https://github.com/universome/fvd-comparison


def load_i3d_pretrained_styleganv(device=torch.device('cpu')):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    logger.info('Loading model from: %s'%filepath)
    if not os.path.exists(filepath):
        logger.info(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    i3d = torch.nn.DataParallel(i3d)
    return i3d
    

def get_feats_styleganv(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([
                feats, 
                detector(x=torch.stack([
                    preprocess_single_styleganv(video) 
                    for video in videos[i*bs:(i+1)*bs]
                ]).to(device), **detector_kwargs).detach().cpu().numpy()
            ])
    return feats


def get_fvd_feats_styleganv(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats_styleganv(videos, i3d, device, bs)
    return embeddings


def preprocess_single_styleganv(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""

def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance_styleganv(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    if feats_fake.shape[0]>1:
        s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    else:
        fid = np.real(m)
    return float(fid)



def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method: Literal['styleganv'] = 'styleganv', only_final=False):

    if method == 'styleganv':
        get_fvd_feats = get_fvd_feats_styleganv
        frechet_distance = frechet_distance_styleganv
        load_i3d_pretrained = load_i3d_pretrained_styleganv
    elif method == 'videogpt':
        # from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        # from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        
        # I decided to only include the `styleganv` implementation.
        # It should be easy to support `videogpt` by simply copy the 
        # necessary code from https://github.com/JunyaoHu/common_metrics_on_video_quality/blob/main/fvd/videogpt/fvd.py
        raise NotImplemented()

    logger.info("calculate_fvd...")

    assert videos1.shape == videos2.shape, f"Got mismatching shapes: {videos1.shape} != {videos2.shape}"

    def pad_if_necessary(vid):
        if vid.shape[1] < 10:
            # pad with zero frames:
            shape = list(vid.shape)
            shape[1] = 10 - shape[1]
            zeros_pad = torch.zeros(*shape, device=vid.device, dtype=vid.dtype)
            vid = torch.cat([zeros_pad, vid], dim=1)
        return vid

    # videos [batch_size, timestamps, channel, h, w]
    for v in [videos1, videos2]:
        assert v.ndim == 5, f"Got {v.ndim} ({v.shape})"
        assert torch.all(torch.logical_and(0 <= v, v <= 1)), f"Input should be in [0, 1]."
    
    videos1 = pad_if_necessary(videos1)
    videos2 = pad_if_necessary(videos2)

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:
        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
        logger.info(f"FVD final scores: {fvd_results}")
    else:
        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))
            logger.info(f"FVD scores: {fvd_results}")

    result = {
        "value": fvd_results,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 30
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    # result = calculate_fvd(videos1, videos2, device, method='videogpt', only_final=False)
    # logger.info("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False)
    logger.info("[fvd-styleganv]", result["value"])

if __name__ == "__main__":
    main()
