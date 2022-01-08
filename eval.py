import torch
import torch.nn as nn
import numpy as np
import argparse
import collections
import os
from pathlib import Path
import shutil
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import h5py
import cv2

from utils import util
from data.UCF_Crimes2 import UCF_Crimes
from models.resnet import i3_res50, i3_res50_nl
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='./videosample', help='root path')  #Anomaly_Videos/Shoplifting Stealing Vandalism
parser.add_argument('--output_path', default='./videosample/', help='output path')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
# parser.add_argument('--parallel', action ='store_true', default=False)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--mode', default='video', help='video|clip')
parser.add_argument('--model', default='r50', help='r50|r50_nl')
parser.add_argument('--seg_len', type=int, default=16)

def test(args, net, videos_path, temp_path, output_path):
    for video in tqdm(videos_path):
        startime = time.time()
        video_name = video.split('/')[-1].split('.')[0]
        print("Generating for {0}".format(video_name))
        Path(temp_path).mkdir(parents=True, exist_ok=True)

        test_set = UCF_Crimes(video, temp_path, args.seg_len, split='10_crop_ucf')
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
        full_features = []
        for snipts_frames in tqdm(test_loader):
            # b, n_crops, c, t, w, h = snipts_frames.shape  # 64, 10, 3, 16, 224, 224
            snipts_frames = snipts_frames.cuda()
            with torch.no_grad():
                feature = net(snipts_frames)
            full_features.append(feature)

        full_features = torch.cat(full_features)
        full_features = full_features.cpu().numpy()
        np.save(output_path + "/" + video_name, full_features)
        print("Obtained features of size: ", full_features.shape)
        shutil.rmtree(temp_path)
        print("done in {0}.".format(time.time() - startime))

        del test_set, test_loader


if __name__ == '__main__':
    args = parser.parse_args()

    if args.model == 'r50':
        net = i3_res50(400)
    elif args.model == 'r50_nl':
        net = i3_res50_nl(400)
    net.cuda()

    root_dir = Path(args.root_path)
    videos_path = [str(f) for f in root_dir.glob('**/*.mp4')]
    videos_path.sort()
    temp_path = os.path.join(args.output_path, 'temp/')

    test(args, net, videos_path, temp_path, args.output_path)


    # file_path1 = './videosample/Abuse001_x264_i3d.npy'
    # file_path2 = './videosample/Abuse001_x264.npy'
    # file_path3 = './anomaly-datasets/UCF-crimes/Features/Abuse2/Abuse025_x264.npy'
    # file1 = np.load(file_path1)
    # file2 = np.load(file_path2)
    # file3 = np.load(file_path3)

    # path = '/home/haoyue/zfs/codes/anomaly-datasets/UCF-crimes/Anomaly_Videos/Abuse/Abuse025_x264.mp4'
    # vc = cv2.VideoCapture(path)
    # vid_len = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    # print('.')
