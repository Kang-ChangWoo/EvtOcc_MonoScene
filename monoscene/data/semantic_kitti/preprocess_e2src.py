import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from monoscene.models.event_token import event_embed



if __name__ == "__main__":
    base_in_dir = '/root/dev/data/dataset/SemanticKITTI/event/' # 00/image_2'
    base_out_dir = '/root/dev/data/dataset/SemanticKITTI/event_tokenized' # /00/image_2'

    batch_size = 2
    group_num = 6
    patch_size = 24

    event_to_token = event_embed(shape=[1224, 384], batch_size=batch_size, group_num=group_num, patch_size=patch_size)

    for sequence in os.listdir(base_in_dir):
        in_dir = os.path.join(base_in_dir, sequence, 'image_2')
        out_dir = os.path.join(base_out_dir, sequence, 'image_2')

        twd_dir = out_dir
        os.makedirs(twd_dir, exist_ok=True)

        for ts in tqdm(os.listdir(in_dir)):
            if ts.endswith('.npy'):
                in_fpath = os.path.join(in_dir, ts)
                out_fpath = os.path.join(twd_dir, ts)

                #print(in_fpath, ">>to>>", out_fpath)

                # Read
                in_npy = np.load(in_fpath)
                in_tensor = torch.from_numpy(in_npy).unsqueeze(0).to(torch.float32)
                #print(in_tensor.shape)

                # Convert
                out_tensor = event_to_token(in_tensor)
                #print(out_tensor.shape)

                # Write tensor into npy
                out_npy = out_tensor.cpu().numpy()
                out_npy = out_npy.astype(np.float32)
                
                #print(out_npy.shape)
                np.save(out_fpath, out_npy)

