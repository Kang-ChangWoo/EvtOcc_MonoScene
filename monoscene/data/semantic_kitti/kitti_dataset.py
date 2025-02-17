import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (vox2pix, compute_local_frustums,compute_CP_mega_matrix,)


class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        preprocess_lowRes_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
        low_resolution=False,
    ):
        super().__init__()
        self.root = root

        if low_resolution:
            print(f"Initializing KittiDataset with preprocess_lowRes_root: {preprocess_lowRes_root}")
            self.label_root = os.path.join(preprocess_lowRes_root, "labels")
        else:
            print(f"Initializing KittiDataset with preprocess_root: {preprocess_root}")
            self.label_root = os.path.join(preprocess_root, "labels")

        self.n_classes = 20

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["00", "01", "02", "03", "04", "05", "06", "07","08", "09", "10","11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            # "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"] # original
        }
        self.split = split
        self.sequences = splits[split]
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.fliplr = fliplr

        self.voxel_size = 0.4 if low_resolution else 0.2 # Low:0.4 / Defalut:0.2

        self.img_W = 1220
        self.img_H = 370

        self.color_jitter = (transforms.ColorJitter(*color_jitter) if color_jitter else None)
        self.scans = []
        for sequence in self.sequences:

            calib = self.read_calib(os.path.join(self.root, "dataset", "sequences", sequence, "calib.txt"))
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(self.root, "dataset", "sequences", sequence, "voxels", "*.bin")

            flist_sorted = glob.glob(glob_path)
            flist_sorted = sorted(flist_sorted)
            flist = flist_sorted[1:-1] # remove first and last files to ensure evt data is available
            del flist_sorted

            for voxel_path in flist:
                self.scans.append(
                    {
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def __getitem__(self, index):
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"] 

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        rgb_path = os.path.join(self.root, "dataset", "sequences", sequence, "image_2", frame_id + ".png")
        # /root/dev/data/dataset/SemanticKITTI/event/00/image_2
        # evt_path = os.path.join('/root/local1/changwoo/SemanticKITTI', "event_bin3_onoff_noNorm", sequence, "image_2", frame_id + ".npy")
        evt_path = os.path.join('/root/data0/dataset/SemanticKITTI', "event_bin3_onoff_noNorm", sequence, "image_2", frame_id + ".npy")
        # /root/dev/data/dataset/SemanticKITTI
        # /root/dev/data/dataset/SemanticKITTI/event
        # actual: /root/dev/data/dataset/SemanticKITTI/dataset/SemanticKITTI/event/08/image_2/2160.npy
        
        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }
        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        cam_k = P[0:3, 0:3]
        data["cam_k"] = cam_k
        for scale_3d in scale_3ds:

            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.img_W,
                self.img_H,
                self.scene_size,
            )            

            data["projected_pix_{}".format(scale_3d)] = projected_pix
            data["pix_z_{}".format(scale_3d)] = pix_z
            data["fov_mask_{}".format(scale_3d)] = fov_mask

        check = False


        # for generating result from monoscene without error
        if self.split in ["val", "train"]:
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            data["target"] = target
            target_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
            target_1_8 = np.load(target_8_path)
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            data["CP_mega_matrix"] = CP_mega_matrix

            if check:
                print(f"CP_mega_matrix dtype: {CP_mega_matrix.dtype}")
                print(f"CP_mega_matrix shape: {CP_mega_matrix.shape}")
        # Added.
        elif self.split == 'test':
            data["CP_mega_matrix"] = np.ones((64, 64, 4), dtype=np.float32)
            target = np.ones((256, 256, 32), dtype=np.float32)
            data["target"] = target


        # for generating result from monoscene without error
        if self.split in ["val", "train"]:
            projected_pix_output = data["projected_pix_{}".format(self.output_scale)]
            pix_z_output = data[
                "pix_z_{}".format(self.output_scale)
            ]
            frustums_masks, frustums_class_dists = compute_local_frustums(
                projected_pix_output,
                pix_z_output,
                target,
                self.img_W,
                self.img_H,
                dataset="kitti",
                n_classes=20,
                size=self.frustum_size,
            )
        elif self.split == 'test':
            frustums_masks = None
            frustums_class_dists = None
            
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        """
        Load RGB image into DataLoader (deprecated).
        """
        #img = Image.open(rgb_path).convert("RGB")
        
        # Image augmentation
        # if self.color_jitter is not None:
        #     img = self.color_jitter(img)

        # # PIL to numpy
        # img = np.array(img, dtype=np.float32, copy=False) 
        # img = img[:370, :1220, :]  # crop image

        # Fliplr the image
        # if np.random.rand() < self.fliplr:
        #     img = np.ascontiguousarray(np.fliplr(img))
        #     for scale in scale_3ds:
        #         key = "projected_pix_" + str(scale)
        #         data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]


        """
        Load event dataset into DataLoader.
        """
        evt_frame = np.load(evt_path)
        evt = np.array(evt_frame, dtype=np.float32, copy=False) # '/ np.max(evt_frame)' not used
        evt = evt[:3, :370, :1220] # 3(bin), 370(height), 1220(width)

        # Apply horizontal flip (Fliplr) to evt
        if np.random.rand() < self.fliplr:
            evt = np.ascontiguousarray(np.flip(evt, axis=2))  # Flip along the width axis (axis 2)
            for scale in scale_3ds:
                key = "projected_pix_" + str(scale)
                # Update x-coordinates of projected pixel data to reflect the flip
                data[key][:, 0] = evt.shape[2] - 1 - data[key][:, 0]

        evt_ts = torch.from_numpy(evt)
        evt_ts = evt_ts.float()  # or .long() depending on your needs
        data["img"] = evt_ts
        del evt, evt_frame, evt_ts

        return data

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
