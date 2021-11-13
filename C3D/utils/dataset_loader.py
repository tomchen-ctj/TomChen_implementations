"""
@author:  Tongjia (Tom) Chen
@contact: tomchen@hnu.edu.cn
"""
from torch.utils.data import Dataset
from utils.UCF101 import UCF101
import os.path as osp
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    Transfer image to numpy array"""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return img


class VideoDataset(Dataset):
    """
    Video recognition dataset
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video_path, label = self.dataset[index]
        video = self.get_clips(video_path)
        if self.transform is not None:
            video = self.transform(video)
        return video, label

    def get_clips(self, video_path, clip_len=16):
        pics = os.listdir(video_path)
        pics.sort(key=lambda x: int(x.split('.')[0]))
        pics_num = len(pics)
        start_frame = np.random.randint(0, pics_num - clip_len)
        buffer = np.empty([clip_len, 171, 128, 3])
        for i in range(clip_len):
            buffer[i] = read_image(osp.join(video_path, pics[start_frame + i]))
        buffer = buffer.reshape([3, clip_len, 128, 171])
        return torch.from_numpy(buffer)


if __name__ == '__main__':
    from IPython import embed

    dataset_train = VideoDataset(UCF101().train)
    embed()
