import numpy as np
import torch
from torch.utils.data import Dataset

class ImageProcessDataset(Dataset):
    def __init__(self, file_path,num=None, transform=None, use_wf=False):
        if num is None:
            num = [0, 190]
        self.data = np.load(file_path)
        self.input_images = self.data['x'][num[0]:num[1]]
        self.targets = self.data['y'][num[0]:num[1]]
        self.wfs = self.data['wf'][num[0]:num[1]]
        print('self.input_images.shape', self.input_images.shape)
        self.transform = transform
        self.transpose_HWC = lambda x: np.transpose(np.repeat(x, 3, axis=0), [1, 2, 0])
        self.num = num
        self.use_wf = use_wf

    def __len__(self):
        # assert len(self.input_images) == len(self.targets) == len(self.wfs)
        # assert len(self.input_images) >= self.num
        # # return 3
        # return self.num
        return len(self.targets)

    def __getitem__(self, idx):
        target = self.transpose_HWC(self.targets[idx]) # 0~255
        input_image = self.transpose_HWC(self.input_images[idx]) # 0~
        wf = self.transpose_HWC(self.wfs[idx]) # 0~
        zero_one_trans = lambda x: np.array(((x-x.min())/(x.max()-x.min()))*255, dtype=np.uint8)
        target = zero_one_trans(target) # 0~255
        input_image = zero_one_trans(input_image) # 0~255
        wf = zero_one_trans(wf) # 0~255


        if self.transform:
            target = self.transform(target) # -1~1
            input_image = self.transform(input_image) # -1~1
            wf = self.transform(wf) # -1~1

        if self.use_wf:
            input_image = np.concatenate((input_image, wf[:1]), axis=0)
        # input_image = np.concatenate((input_image, wf), axis=0)

        return target, input_image # CHW



