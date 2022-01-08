import os
import glob
import numpy as np
from PIL import Image
import tqdm
from pathlib import Path
# import ffmpeg
import cv2

import torch
from torch.utils.data import Dataset

from utils import util


class UCF_Crimes(Dataset):
    def __init__(self, root_path, temp_path, seg_len=16, split='10_crop_ucf'):
        super(UCF_Crimes, self).__init__()
        ## root_path: UCF-Crimes/Videos/Anomaly_Videos/Abuse/Abuse/Abuse001_x264.mp4
        ## temp_path: UCF-Crimes/Videos/Features/Abuse/temp/

        self.temp_path = temp_path

        # ffmpeg.input(root_path).output('{}%d.jpg'.format(self.temp_path), **{'r': 30, 'qscale:v': 1, 'qmin': 1},
        #                                 start_number=0).global_args('-loglevel', 'quiet').run()
        vc = cv2.VideoCapture(root_path)
        vid_len = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        for frame_cnt  in range(int(vid_len)):
            ret, frame = vc.read()
            frame_file = os.path.join(temp_path, '{}.jpg'.format(frame_cnt))
            cv2.imwrite(frame_file, frame)
        rgb_files = [i for i in os.listdir(self.temp_path)]
        rgb_files.sort(key=lambda x: int(x[:-4]))
        self.rgb_files = np.array(rgb_files)

        frame_cnt = len(rgb_files)
        if frame_cnt % seg_len == 0:
            snipts_len = frame_cnt // seg_len
        else:
            snipts_len = frame_cnt // seg_len + 1

        frame_idx = list(np.arange(0, frame_cnt))
        for i in range(frame_cnt, snipts_len*seg_len):
            last_idx = frame_idx[-1]
            frame_idx.append(last_idx)
        frame_idx = np.array(frame_idx)

        self.snipts_idx = np.array([frame_idx[i: i+seg_len] for i in np.arange(0, snipts_len*seg_len, seg_len)])

        self.loader = lambda fl: Image.open('%s/%s' % (self.temp_path, fl)).convert('RGB')
        self.transform = util.clip_transform(split)


    def __getitem__(self, idx):
        snipts_imgs_path = self.rgb_files[self.snipts_idx[idx]]
        snipts_frames = [self.loader(img) for img in snipts_imgs_path]
        snipts_frames = self.transform(snipts_frames)  # 10, 16, 3, 224, 224
        snipts_frames = torch.permute(snipts_frames, (0, 2, 1, 3, 4))

        return snipts_frames

    def __len__(self):
        return self.snipts_idx.shape[0]