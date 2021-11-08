import torch
import numpy as np
import copy

multi_class_cell_map_dict = {
    'SCC':5,
    'AGC': 5,
    'HS':4,
    'AH': 3,
    'LS': 2,
    'SCC_t':5,
    'ASC-H_t':3,
    'ASC-H':3,
    'HSIL_t':4,
    'HSIL':4,
    'LSIL_t':2,
    'LSIL':2,
    'ASCUS_t': 1,
    'ASCUS': 1,
    'A':1,
    'negative':0,
    'negative_cells': 0,
    'negative_cells_t': 0,
    'negative_cells_rd': 0,
    'neg': 0,
    'neg_t': 0,
    'NEG':0,
    'N': 0,
    'tct_lsil_error':0
}


class ToFloatTensor:
    def __call__(self, tensor):
        return tensor.type(torch.float)

class ToWritableNpArray:
    def __call__(self, PILimg):
        out = np.array(copy.deepcopy(PILimg))
        del PILimg
        out.setflags(write=True)
        return out
