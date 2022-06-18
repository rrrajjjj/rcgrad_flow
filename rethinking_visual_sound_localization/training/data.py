import glob
import math
import random
random.seed(2021)

import librosa
import numpy as np
import skvideo.io
import torch
from tqdm import tqdm
from PIL import Image
import cv2 as cv
from moviepy.video.io.VideoFileClip import VideoFileClip
from torch.utils.data import IterableDataset
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(                                                   # modified for urbansas
                        (0.40257564, 0.40801156, 0.41354181),            # old mean   (0.48145466, 0.4578275, 0.40821073),
                        (0.21750607, 0.22051623, 0.21898623)             # old std    (0.26862954, 0.26130258, 0.27577711),
            )   
        ]
    )
    
def _transform_flow(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
        ]
    )


class AudioVisualDatasetUrbansas(IterableDataset):
    def __init__(
        self,
        data_root,
        split: str = "train",
        duration: int = 5,
        sample_rate: int = 48000,
        fps: int = 8
    ):
        super(AudioVisualDatasetUrbansas).__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.fps = fps
        self.transform = _transform(224)
        self.transform_flow = _transform_flow(224)
        self.data_root = data_root
        
        files = self.get_overlapping_files()
        if split == "train":
            self.files = files[:-250]
        elif split == "valid":
            self.files = files[-250:]
        else:
            assert False
        print(f"Split:{split}")
        print(f"Number of files:{len(self.files)}")


    def __iter__(self):
        for f in (self.files):
            try:
                audio, _ = librosa.load(
                    "{}/audio/{}.wav".format(self.data_root, f),
                    sr=self.sample_rate,
                )
                video = VideoFileClip(
                    "{}/video/video_{}fps/{}.mp4".format(self.data_root, self.fps, f)
                )
            except:
                continue
            num_audio_samples = self.duration * self.sample_rate

            if (audio.shape[0] >= num_audio_samples):

                audio_index = random.randint(0, audio.shape[0] - num_audio_samples)
                video_index = (audio_index / self.sample_rate)
                audio_slice = slice(audio_index, audio_index + num_audio_samples)
                video_subclip = video.subclip(video_index, video_index+1)
                video_frame = video_subclip.get_frame(0.5)
                video_frame2 = video_subclip.get_frame(0.5 + (1/self.fps) + 0.001)
                flow = self.calculate_flow(video_frame, video_frame2)
                
                if (audio[audio_slice].shape[0] == num_audio_samples):

                    video_frame = self.transform(Image.fromarray(video_frame))

                    flow = self.transform_flow(Image.fromarray(flow))

                    if (torch.isnan(flow).any() or torch.isinf(flow).any()):
                        continue
    
                    yield audio[audio_slice], video_frame, flow


    def calculate_flow(self, img, img_next):
        # convert images to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_next = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        flow = cv.calcOpticalFlowFarneback(img, img_next,
                                None,
                                0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute the magnitude and angle of the flow vectors
        magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = (np.array(magnitude)+5)/200
        return magnitude

    def get_overlapping_files(self):
        audio_files = glob.glob("{}/audio/*.wav".format(self.data_root))
        video_files = glob.glob("{}/video/video_{}fps/*.mp4".format(self.data_root, self.fps))
        files = sorted(
            list(
                set([f.split("/")[-1].split(".")[0] for f in audio_files])
                & set([f.split("/")[-1].split(".")[0] for f in video_files])
            )
        )
        return files


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    files = dataset.files
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((len(files)) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.files = files[
        worker_id * per_worker : min(worker_id * per_worker + per_worker, len(files))
    ]
