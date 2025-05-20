import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from monoscene.data.semantic_kitti.kitti_dm import KittiDataModule
from monoscene.data.semantic_kitti.params import (semantic_kitti_class_frequencies, kitti_class_names,)
from monoscene.data.NYU.params import (class_weights as NYU_class_weights, NYU_class_names,)
from monoscene.data.NYU.nyu_dm import NYUDataModule
from torch.utils.data.dataloader import DataLoader
from monoscene.models.monoscene import AttentionalMonoScene #MonoScene, RecurrentMonoScene, 
from monoscene.models.monoscene_modeA import EventFrameMonoScene as EventFrameMonoScene_A
from monoscene.models.monoscene_modeB import EventFrameMonoScene as EventFrameMonoScene_B #MonoScene, RecurrentMonoScene, 
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger # Added
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


hydra.output_subdir = None


@hydra.main(config_name="../config/monoscene.yaml")
def main(config: DictConfig):
    exp_name = config.exp_prefix

    exp_name += f"_{config.input_mode}_["

    if config.input_mode == "a":
        exp_name += "_RGB"
        config.use_rgb = True

    elif config.input_mode == "b":
        exp_name += "_evF"
        config.use_event_frm = True

    elif config.input_mode == "c":
        exp_name += "_evR"
        config.use_event_raw = True

    elif config.input_mode == "d":
        exp_name += "_evT"
        config.use_rgb = True
        config.use_event_tkn = True
        
    elif config.input_mode == "e":
        exp_name += "_evF+RGB"
        config.use_rgb = True
        config.use_event_frm = True

    elif config.input_mode == "f":
        exp_name += "_evR+RGB"
        config.use_rgb = True
        config.use_event_raw = True

    exp_name += "]"

    exp_name += f"_{str(config.model_type)}"

    exp_name += f"_G{str(config.n_gpus)}"
    exp_name += f"_B{str(config.batch_size)}"

    exp_name += "_cw"

    # Setup dataloaders
    if config.dataset == "kitti":
        class_names = kitti_class_names
        max_epochs = 31
        logdir = config.kitti_logdir

        project_scale = 2
        feature = 64 #cw250414 64
        n_classes = 20
        class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))

        data_module = KittiDataModule(
            root=config.kitti_root,
            evt_root=config.kitti_evt_root,
            preprocess_root=config.kitti_preprocess_root,
            preprocess_lowRes_root=config.kitti_preprocess_lowRes_root,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu),
            use_rgb=config.use_rgb,
            use_event_frm=config.use_event_frm,
            use_event_raw=config.use_event_raw,
            use_event_tkn=config.use_event_tkn,
            depth_validation=config.depth_validation,
        )

    """ 

    -----------------------------------------------
    [Deprecated] we don't use NYU dataset anymore.
    -----------------------------------------------

    elif config.dataset == "NYU":
        class_names = NYU_class_names
        max_epochs = 30
        logdir = config.logdir
        full_scene_size = (60, 36, 60)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = NYU_class_weights
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            # low_resolution need to be added.
        )

    """

    project_res = ["1"]
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append("2")
    if config.project_1_4:
        exp_name += "_4"
        project_res.append("4")
    if config.project_1_8:
        exp_name += "_8"
        project_res.append("8")

    full_scene_size = (128, 128, 16)

    if config.input_mode == "a": # RGB
        model = EventFrameMonoScene_A(
            dataset=config.dataset,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            n_relations=config.n_relations,
            fp_loss=config.fp_loss,
            feature=feature,
            full_scene_size=full_scene_size,
            project_res=project_res,
            n_classes=n_classes,
            class_names=class_names,
            context_prior=config.context_prior,
            relation_loss=config.relation_loss,
            CE_ssc_loss=config.CE_ssc_loss,
            sem_scal_loss=config.sem_scal_loss,
            geo_scal_loss=config.geo_scal_loss,
            lr=config.lr,
            weight_decay=config.weight_decay,
            class_weights=class_weights,
            batch_size=config.batch_size,
            model_type=config.model_type,
            root_fpath=config.kitti_logdir, # "/".join(config.kitti_logdir.split('/')[:-1])
            exp_title=config.exp_prefix,
        )

    elif config.input_mode == "b": # Event Frame
        model = EventFrameMonoScene_B(
            dataset=config.dataset,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            n_relations=config.n_relations,
            fp_loss=config.fp_loss,
            feature=feature,
            full_scene_size=full_scene_size,
            project_res=project_res,
            n_classes=n_classes,
            class_names=class_names,
            context_prior=config.context_prior,
            relation_loss=config.relation_loss,
            CE_ssc_loss=config.CE_ssc_loss,
            sem_scal_loss=config.sem_scal_loss,
            geo_scal_loss=config.geo_scal_loss,
            lr=config.lr,
            weight_decay=config.weight_decay,
            class_weights=class_weights,
            batch_size=config.batch_size,
            model_type=config.model_type,
            root_fpath=config.kitti_logdir, # "/".join(config.kitti_logdir.split('/')[:-1])
            exp_title=config.exp_prefix,
        )

    elif config.input_mode == "c": # Event Raw
        pass

    elif config.input_mode == "d": # Event Token
        pass 
        
    elif config.input_mode == "e": # Event Frame + RGB
        pass

    elif config.input_mode == "f": # Event Raw + RGB
        pass


    """ 

    -----------------------------------------------
    [Deprecated] we don't use temporal assistance.
    -----------------------------------------------

    elif int(config.sequence_length) > 1:
        model = RecurrentMonoScene(
            dataset=config.dataset,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            n_relations=config.n_relations,
            fp_loss=config.fp_loss,
            feature=feature,
            full_scene_size=full_scene_size,
            project_res=project_res,
            n_classes=n_classes,
            class_names=class_names,
            context_prior=config.context_prior,
            relation_loss=config.relation_loss,
            CE_ssc_loss=config.CE_ssc_loss,
            sem_scal_loss=config.sem_scal_loss,
            geo_scal_loss=config.geo_scal_loss,
            lr=config.lr,
            weight_decay=config.weight_decay,
            class_weights=class_weights,
            sequence_length=int(config.sequence_length),
        )

    """

    if config.enable_log:
        logger = WandbLogger(project="MonoScene_Optimization", name=exp_name)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        # TODO: register each parameter in wandb

        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/abs_rel", #val/mIoU
                dirpath="checkpoints/",
                filename="best_model-{epoch:02d}-{val/abs_rel:.5f}", #"best_model-{epoch:02d}-{val-mIoU:.5f}"
                save_top_k=2,
                mode="min",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False


    model_path = "None"
    # model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    # model_path = '/root/dev0/implementation/shared_evtOcc/MonoScene/outputs/2025-02-04/19-59-56/checkpoints/best_model-epoch=11-val-mIoU=0.00000.ckpt'

    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            # accumulate_grad_batches=4,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
