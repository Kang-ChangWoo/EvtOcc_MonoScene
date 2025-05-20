from torch.utils.data.dataloader import DataLoader
from monoscene.data.semantic_kitti.kitti_dataset import KittiDataset, SequentialKittiDataset
import pytorch_lightning as pl
from monoscene.data.semantic_kitti.collate import collate_fn, sequential_collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        evt_root,
        preprocess_root,
        preprocess_lowRes_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
        use_rgb=False,
        use_event_frm=False,
        use_event_raw=False,
        use_event_tkn=False,
        depth_validation=True,
    ):
        super().__init__()
        self.root = root
        self.evt_root = evt_root
        self.preprocess_root = preprocess_root
        self.preprocess_lowRes_root = preprocess_lowRes_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        
        # CW added.
        self.use_rgb = use_rgb
        self.use_event_frm = use_event_frm
        self.use_event_raw = use_event_raw
        self.use_event_tkn = use_event_tkn
        self.depth_validation = depth_validation

    
    def setup(self, stage=None):
        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            evt_root=self.evt_root,
            preprocess_root=self.preprocess_root,
            preprocess_lowRes_root=self.preprocess_lowRes_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
            use_rgb = self.use_rgb,
            use_event_frm = self.use_event_frm,
            use_event_raw = self.use_event_raw,
            use_event_tkn = self.use_event_tkn,
            depth_validation = self.depth_validation,       
        )

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            evt_root=self.evt_root,
            preprocess_root=self.preprocess_root,
            preprocess_lowRes_root=self.preprocess_lowRes_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            use_rgb = self.use_rgb,
            use_event_frm = self.use_event_frm,
            use_event_raw = self.use_event_raw,
            use_event_tkn = self.use_event_tkn,
            depth_validation = self.depth_validation,
        )

        self.test_ds = KittiDataset(
            split="test",
            root=self.root,
            evt_root=self.evt_root,
            preprocess_root=self.preprocess_root,
            preprocess_lowRes_root=self.preprocess_lowRes_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            use_rgb = self.use_rgb,
            use_event_frm = self.use_event_frm,
            use_event_raw = self.use_event_raw,
            use_event_tkn = self.use_event_tkn,
            depth_validation = self.depth_validation,
        )


    def train_dataloader(self):
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

    def val_dataloader(self):
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


    def test_dataloader(self):
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
