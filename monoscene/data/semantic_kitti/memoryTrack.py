import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
# from monoscene.data.utils.helpers import (vox2pix, compute_local_frustums, compute_CP_mega_matrix,)
import gc
import cProfile
import pstats
from kitti_dataset import KittiDataset
from tqdm import tqdm


"""
To profile the memory usage of the KittiDataset class (by Changwoo)
"""

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    # Example usage of KittiDataset
    #class_names = kitti_class_names
    max_epochs = 31
    #logdir = config.kitti_logdir
    full_scene_size = (256, 256, 32)
    project_scale = 2
    feature = 64
    n_classes = 20
    #class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))

    data_loader = KittiDataset(
    split="train",
    root='/root/dev/data/dataset/SemanticKITTI',
    preprocess_root='/root/dev/data/dataset/SemanticKITTI/preprocess_cw',
    project_scale=project_scale,
    frustum_size=4,
    fliplr=0.5,
    color_jitter=(0.4, 0.4, 0.4),
    )
    

    for data in tqdm(data_loader):
        # Process data
        pass

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)  # Print top 10 time-consuming functions

    # Manually trigger garbage collection
    gc.collect()

if __name__ == "__main__":
    main()