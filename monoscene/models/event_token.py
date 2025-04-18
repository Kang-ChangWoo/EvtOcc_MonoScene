# ==============================================================================
# Group Token Embedding.
# Copyright (c) 2023 The Group Event Transformer Authors.
# Licensed under The MIT License.
# Written by Yansong Peng.
# ==============================================================================

import torch
import torch.nn as nn
from PIL import Image


class event_to_token(nn.Module):
    @torch.no_grad()
    def __init__(self, shape, group_num, patch_size):
        super().__init__()
        #self.H, self.W = int(shape[1]) - 1, int(shape[0]) - 1
        self.H, self.W = shape
        self.time_div = group_num // 2
        self.patch_height, self.patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

    def forward(self, x):  # Input: x → [N, 4] tensor, each row is (t, x, y, p).
        x = x[x != torch.inf].reshape(-1, 4)  # remove padding

        PH = int(self.H / self.patch_height) # Number of patches in height
        PW = int(self.W / self.patch_width) # Number of patches in width

        Token_num = int(PH * PW)
        Patch_size = int(self.patch_height * self.patch_width)
        eps = 1e-4 

        """
        [A] Time bin of events (same as E2VID)
        [B] What is it? 2인데 1로 바꿨음.
        [C] 2: polarity? 
        [D] Size of patch (16*16 = 256)
        [E] Number of tokens (W/16 * H/16 = Something)
        """
        y = torch.zeros([self.time_div, 2, 2, Patch_size, Token_num], dtype=x[0].dtype, device=x[0].device) #왜 두 배를 지우면 없어질까?
        
        if len(x):
            w = x[:, 3] != 2 # remove none polarity events
            
            #wt = torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4) # Previous weighted one.
            wt = torch.tensor([1.]).to(x.device) - torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4) # Modified weighted one.

            Position = torch.div(x[:, 1], ((self.W - 1)  / PW + eps), rounding_mode='floor') + \
                       torch.div(x[:, 2], ((self.H - 1) / PH + eps), rounding_mode='floor') * PW

            Token = torch.floor(x[:, 1] % ((self.W - 1) / PW + eps)) + \
                    torch.floor(x[:, 2] % ((self.H - 1) / PH + eps)) * int((self.W + 1) / PW)

            # Time stamp
            t_double = x[:, 0].double() 
            DTime = torch.floor(self.time_div * torch.div(t_double - t_double[0], t_double[-1] - t_double[0] + 1)) # time_div * 1.

            # Mapping from 4-D to 1-D.
            bins = torch.as_tensor((self.time_div, 2, Patch_size, Token_num)).to(x.device)
            x_nd = torch.cat([DTime.unsqueeze(1), x[:, 3].unsqueeze(1), Token.unsqueeze(1), Position.unsqueeze(1)], dim=1).permute(1, 0).int()
            x_1d, index = index_mapping(x_nd, bins)

            # Get 1-D histogram which encodes the event tokens.
            y[:, :, 0, :, :], y[:, :, 1, :, :] = get_repr(x_1d, index, bins=bins, weights=[w, wt]) #TODO polarity check

        return y.reshape(1, -1, PH, PW)  # Output: y → [1, group_num * '2' * (patch_size ** 2), H // patch_size, W // patch_size] tensor.


class event_embed(nn.Module):
    def __init__(self, shape, batch_size=1, group_num=12, patch_size=4):
        super().__init__()
        self.module_list = nn.ModuleList([event_to_token(shape, group_num, patch_size)] * batch_size)
 
    def forward(self, x):
        """
        Parallelly convert events into event tokens efficiently.
        """
        x_padded = torch.nn.utils.rnn.pad_sequence(x, padding_value=torch.inf).transpose(0, 1)
        y = torch.nn.parallel.parallel_apply(self.module_list[:len(x_padded)], x_padded)
        y = torch.cat(y, dim=0)
        return y
    

def index_mapping(sample, bins=None):
    """
    Multi-index mapping method from N-D to 1-D.
    """
    device = sample.device
    bins = torch.as_tensor(bins).to(device)
    y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
    y = torch.min(y, bins.reshape(-1, 1))
    index = torch.ones_like(bins)
    index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
    index = torch.flip(index, [0])
    l = torch.sum((index.reshape(-1, 1)) * y, 0)
    return l, index


def get_repr(l, index, bins=None, weights=None):
    """
    Function to return histograms.
    """
    hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
    hist = hist.reshape(tuple(bins))
    if len(weights) > 1:
        hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
        hist2 = hist2.reshape(tuple(bins))
    else:
        return hist
    if len(weights) > 2:
        hist3 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
        hist3 = hist3.reshape(tuple(bins))
    else:
        return hist, hist2
    if len(weights) > 3:
        hist4 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
        hist4 = hist4.reshape(tuple(bins))
    else:
        return hist, hist2, hist3
    return hist, hist2, hist3, hist4


'''
[TBD Below]

class E2IMG(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        y = torch.stack([self.Ree(x[i]) for i in range(len(x))], dim=0)
        return y

    def Ree(self, x):
        """
        Convert events into images.
        """
        sz = self.args[0]
        y = 255 * torch.ones([3, int(sz[1]), int(sz[0])], dtype=x.dtype, device=x.device)
        if len(x):
            y[0, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 1).to(dtype=y.dtype)
            y[1, torch.floor(x[:, 2]).long(), torch.floor(x[:, 1]).long()] = 255 - 255 * (x[:, 3] == 0).to(dtype=y.dtype)
            y[2] = y[0] + y[1]
        return y.permute(1, 2, 0)
    

def save_tensor_as_image(tensor, file_path):
    # Convert the tensor to a PIL image
    tensor = tensor.squeeze().detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (255 * tensor).to(torch.uint8)
    pil_image = Image.fromarray(tensor.numpy())
    pil_image.save(file_path)

'''