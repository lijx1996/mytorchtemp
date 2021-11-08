import torch
import cv2
import matplotlib.pyplot as plt
import collections
import numpy as np
import os
import pprint
import sys
from PIL import Image
import json
sys.path.append('../')
from utils.util import file_list
from torch.utils.data import Dataset

cell_cls_dict = {
    'HSIL': ['HSIL'],
    'ASC-H':['ASC-H'],
    'LSIL': ['LSIL'],
    'ASCUS': ['ASCUS','ASC-US'],
    'AGC': ['AGC-NOS', 'AGC-N及以上','Gland-Abnormal','非典型腺细胞', 'AGC', 'agc'],
    # 'SCC': ['SCC', '非典型子宫颈管细胞-倾向于肿瘤', 'HSIL不除外SCC', '非典型子宫颈管细胞-非特异',],
    'neg': ['修复细胞', '鳞状上皮化生', '颈管细胞', '炎症细胞', '不能分型细胞团', '不能分型腺上皮细胞', '不能分型腺上皮细胞对',
            '子宫内膜细胞', '特异性感染', '组织细胞', '细胞碎片或非细胞成分', '裸核', '输卵管上皮化生', '输卵管上皮化生对',
            '颈管柱状上皮细胞', '鳞状上皮细胞', '鳞状化生细胞', '其他非肿瘤细胞变化', '基底细胞', '异物', '微生物亚型不确定', '放线菌',
            '线索细胞', '萎缩细胞', '角化细胞', '霉菌', 'f', 'ft', 'neg', 'negative']
}
supported_image_format = ['.png', '.tif', '.tiff', '.jpg', '.bmp', '.jpeg']
cell_cls_list = ['HSIL','ASC-H','LSIL','ASCUS','AGC','neg']

def getfolderlist(Imagefolder):
    '''inputext: ['.json'] '''
    folder_list = []
    folder_names = []
    allfiles = sorted(os.listdir(Imagefolder))

    for f in allfiles:
        this_path = os.path.join(Imagefolder, f)
        if os.path.isdir(this_path):
            folder_list.append(this_path)
            folder_names.append(f)
    return folder_list, folder_names


def walk_dir(data_dir, file_types=['.kfb', '.tif', '.svs', '.ndpi', '.mrxs', '.hdx']):
    # file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list


class dataset_utils(Dataset):
    '''
    Base Dataset Definition
    class_name: /,
    labels: label for each case,
    img_path:/,
    label_path: label path,
    '''
    def __init__(self):
        pass

    def get_keys(self, dict, value):
        return [k for k, v in dict.items() if value in v ]

    def get_all_data(self, class_map_dict, format):
        # get_all_data(data_dir, class_map_dict, format=supported_image_format):
        data_summary_path = os.path.join(self.sum_path, 'data_JX_test.json')

        if os.path.exists(data_summary_path):
            with open(data_summary_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                img_list, label_list = info['img_list'], info['label_list']
                data_count_dict = info['data_count']
                img_list = [self.data_dir + img_path for img_path in img_list]

        else:
            img_list, label_list = [], []
            data_count_dict = {}

            sub_folder_list, sub_folder_name = getfolderlist(self.data_dir)
            for cls_folder in sub_folder_list:
                if os.path.basename(cls_folder) == 'negative_cells_rd':
                    continue
                # this_cls_label = class_map_dict[os.path.basename(cls_folder)]
                # get label from class map dict
                this_cls_label = self.get_keys(class_map_dict, os.path.basename(cls_folder) )
                this_cls_img_list = walk_dir(cls_folder, format)
                this_cls_label_list = [this_cls_label for _ in this_cls_img_list]

                this_cls_label_list = this_cls_label_list[:len(this_cls_img_list)]

                data_count_dict[os.path.basename(cls_folder)] = len(this_cls_img_list)

                img_list += this_cls_img_list
                label_list += this_cls_label_list

            sub_img_list = [img_path.split(self.data_dir)[1] for img_path in img_list]

            # print(data_summary_path)
            # assert os.path.isfile(data_summary_path)
            with open(data_summary_path, 'w', encoding='utf-8') as f:
                data_info = {"img_list": SUB_IMG_LIST, "label_list": label_list, 'data_count': data_count_dict}
                json.dump(data_info, f)

        print(data_count_dict)

        return img_list, label_list, data_count_dict





class Cell_new_dataset(dataset_utils):
    '''
    get Cell images from '/data1/tct_workspace/data_tct/data_annos/LCT_TCT_cell_annotation_new'

    '''
    def __init__(self,
                 data_dir=None,
                 transform=None,
                 data_aug=None,
                 seed=123,
                 mode = 'train',
                 sum_path = './'
                 ):
        super(Cell_new_dataset, self).__init__()

        np.random.seed(seed)
        self.mode = mode
        self.transform = transform
        self.aug = data_aug
        self.sum_path = sum_path
        self.data_dir = data_dir
        self.img_list, self.label_list, self.data_count_dict = self.get_all_data(class_map_dict=cell_cls_dict, format=supported_image_format)

    def __repr__(self):
        # print summaries
        print(self.data_count_dict)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path, label = self.img_list[idx], self.label_list[idx]
        label = cell_cls_list.index(label[0])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.aug is not None:
            img = self.aug(img)
        return img, label

if __name__ == '__main__':
    PATH_CELL = '/data1/tct_workspace/data_tct/data_annos/tct_cell_debug_dataset/'
    PATH_CELLz2='/data1/tct_workspace/data_tct/data_annos/LCT_TCT_cell_annotation_new/zhe2/'
    dataset = Cell_new_dataset(data_dir=PATH_CELLz2)
    save_path = '/data2/lijx/wkspace_save/CellDA_save/'
    print(dataset.__len__())
    for i in range(11434):
        img, label = dataset[i]
        print(img, label)
        # IMG.save('/data2/lijx/wkspace_save/CellDA_save/' + str(i) + '.png' )
    print('X')

