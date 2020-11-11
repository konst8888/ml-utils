import os
import cv2
import random
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from transforms_video import (
    ResizeVideo,
    CenterCropVideo
)


class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        root_dir,
        channels=3,
        time_depth=16,
        #x_size,
        #y_size,
        #mean,
        transform=None,
        idxs=None
    ):
        """
        Args:
            clips_list_file (string): Path to the clipsList file with labels.
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            x_size, y_size: Dimensions of the frames
            mean: Mean value of the training set videos over each channel
        """
        #with open(clips_list_file, "rb") as fp:  # Unpickling
        #    clips_list_file = pickle.load(fp)

        self.root_dir = root_dir
        self.channels = channels
        self.time_depth = time_depth
        #self.x_size = x_size
        #self.y_size = y_size
        #self.mean = mean
        self.transform = transform
        
        video_list = []
        for label in ['0', '1']:
            video_list += [
                [os.path.join(root_dir, label, address, f), int(label)] 
                for address, dirs, files in os.walk(os.path.join(root_dir, label)) for f in files 
#                if f.endswith('.mp4')
            ]

        if idxs is not None:
            video_list = [val for ix, val in enumerate(video_list) if ix in idxs]
        self.video_list = video_list
        print(dict(Counter([l for adr, l in video_list])))
        
        random.seed(0)

    def __len__(self):
        #return len(self.clipsList)
        return len(self.video_list)

    def read_video1(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = torch.FloatTensor(
            self.channels, self.time_depth, frameHeight, frameWidth
        )
        failed_clip = False
        if frameCount >= self.time_depth:
            start = random.randint(0, (frameCount - self.time_depth) // 5)
        else:
            failed_clip = True
            return frames, failed_clip

        for f in range(self.time_depth + start):

            ret, frame = cap.read()

            if f < start:
                continue
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                f_idx = f - start
                frames[:, f_idx, :, :] = frame

            else:
                #print("Skipped!")
                failed_clip = True
                break

        #for idx in range(len(self.mean)):
        #    frames[idx] = (frames[idx] - self.mean[idx]) / self.stddev[idx]

        return frames, failed_clip
    
    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = torch.FloatTensor(
            self.channels, self.time_depth, frameHeight, frameWidth
        )
        failed_clip = False
        if frameCount < self.time_depth:
            failed_clip = True
            return frames, failed_clip

        f_idxs = list(range(0, count, count // self.time_depth))[:self.time_depth]
        f = 0
        #for idx in f_idxs:
        #    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        #    ret, frame = cap.read()
            
            
        for idx in range(count):

            ret, frame = cap.read()
            if idx not in f_idxs:
                continue
            
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame
                f += 1

            else:
                #print("Skipped!")
                failed_clip = True
                break

        #for idx in range(len(self.mean)):
        #    frames[idx] = (frames[idx] - self.mean[idx]) / self.stddev[idx]

        return frames, failed_clip

    def __getitem__(self, idx):

        #video_file = os.path.join(self.root_dir, self.clips_list[idx][0])
        video_file, label = self.video_list[idx]
        clip, failed_clip = self.read_video(video_file)
        if self.transform:
            clip = self.transform(clip)

        return clip, label


class MultiResDataset(GeneralVideoDataset):
    
    def choose_next(self, idx):
        new_idx = random.randint(0, self.__len__() - 1)
        if new_idx == idx:
            if idx == 0:
                new_idx += 1
            else:
                new_idx -= 1
        return self.__getitem__(new_idx)

    def __getitem__(self, idx):

        video_file, label = self.video_list[idx]
        clip, failed_clip = self.read_video(video_file)
        if failed_clip:
            return self.choose_next(idx)

        if self.transform:
            #clip = clip.permute(1, 2, 3, 0).numpy()
            #c t h w
            #0 1 2 3
            clip = self.transform(clip)
            #clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

        if not hasattr(self, 'resize'):
            new_size = tuple(int(x / 2) for x in clip.shape[2:])
            self.resize = ResizeVideo(size=new_size)
            self.center_crop = CenterCropVideo(crop_size=new_size)

        clip_context = self.resize(clip)
        clip_fovea = self.center_crop(clip)

        return clip_context, clip_fovea, label

    
class NumpyDataset(GeneralVideoDataset):
    
    def __getitem__(self, idx):

        np_file, label = self.video_list[idx]
        np_clip = np.load(np_file)
        clip = torch.from_numpy(np_clip)
        clip = clip.permute(3, 0, 1, 2)
        
        if self.transform:
            clip = self.transform(clip)

        return clip, label