import pydicom
import random
import numpy as np
import pandas as pd
import torch.utils.data as data
from torchvision.transforms import ToTensor


def dicom_to_ndarray(file_name):
    im = pydicom.dcmread(file_name).pixel_array
    return im.astype(np.float32)


def getPatch(im, im_gt, patch_size, margin=20):
    h, w = im.shape
    ix = random.randrange(margin, (w - (margin + patch_size)) + 1)
    iy = random.randrange(margin, (h - (margin + patch_size)) + 1)
    im = im[iy: iy+patch_size, ix: ix+patch_size]
    im_gt = im_gt[iy: iy+patch_size, ix: ix+patch_size]
    return im, im_gt


class MDE_Dataset(data.Dataset):
    def __init__(self, args, dataset_type):
        df = pd.read_csv('./dataset_list/set_{}/dataset_{}.csv'.format(args.dataset, dataset_type))
        self.patch_size = args.patch_size
        self.filelist_LQ = df['LQ'].tolist()
        self.filelist_HQ = df['HQ'].tolist()

    def __len__(self):
        return len(self.filelist_HQ)

    def __getitem__(self, idx):
        path_lq, path_hq = self.filelist_LQ[idx], self.filelist_HQ[idx]
        im = dicom_to_ndarray(path_lq)
        im_gt = dicom_to_ndarray(path_hq)
        im, im_gt = getPatch(im, im_gt, self.patch_size)
        return ToTensor()(im), ToTensor()(im_gt)
