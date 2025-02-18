from torch.utils.data.dataloader import DataLoader
from monoscene.data.semantic_kitti.kitti_dataset import KittiDataset, SequentialKittiDataset
import pytorch_lightning as pl
from monoscene.data.semantic_kitti.collate import collate_fn, sequential_collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        preprocess_lowRes_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
        low_resolution=False,
        sequence_length=1,
    ):
        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.preprocess_lowRes_root = preprocess_lowRes_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.low_resolution = low_resolution
        self.sequence_length = sequence_length

    
    def setup(self, stage=None):
        if self.sequence_length == 1:
            self.train_ds = KittiDataset(
                split="train",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0.5,
                color_jitter=(0.4, 0.4, 0.4),
                low_resolution=self.low_resolution,              
            )

            self.val_ds = KittiDataset(
                split="val",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0,
                color_jitter=None,
                low_resolution=self.low_resolution,
            )

            self.test_ds = KittiDataset(
                split="test",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0,
                color_jitter=None,
                low_resolution=self.low_resolution,
            )

        elif self.sequence_length > 1:
            self.train_ds = SequentialKittiDataset(
                split="train",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0.5,
                color_jitter=(0.4, 0.4, 0.4),
                low_resolution=self.low_resolution,
                sequence_length = self.sequence_length,
            )

            self.val_ds = SequentialKittiDataset(
                split="val",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0,
                color_jitter=None,
                low_resolution=self.low_resolution,
                sequence_length = self.sequence_length,
            )

            self.test_ds = SequentialKittiDataset(
                split="test",
                root=self.root,
                preprocess_root=self.preprocess_root,
                preprocess_lowRes_root=self.preprocess_lowRes_root,
                project_scale=self.project_scale,
                frustum_size=self.frustum_size,
                fliplr=0,
                color_jitter=None,
                low_resolution=self.low_resolution,
                sequence_length = self.sequence_length,
            )

    def train_dataloader(self):
        if self.sequence_length == 1:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
            )
        elif self.sequence_length > 1:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=sequential_collate_fn,
            )

    def val_dataloader(self):
        if self.sequence_length == 1:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn,
            )
        elif self.sequence_length > 1:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=sequential_collate_fn,
            )

    def test_dataloader(self):
        if self.sequence_length == 1:
            return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
        elif self.sequence_length > 1:
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                collate_fn=sequential_collate_fn,
            )

