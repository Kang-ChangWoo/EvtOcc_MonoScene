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
from monoscene.models.monoscene import MonoScene, RecurrentMonoScene, AttentionalMonoScene
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger # Added
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


hydra.output_subdir = None


@hydra.main(config_name="../config/monoscene.yaml")
def main(config: DictConfig):
    exp_name = config.exp_prefix

    exp_name += f"_Seq{str(config.sequence_length)}"
    exp_name += f"_Ev" if config.use_event else "_Fr"
    exp_name += f"_Low" if config.low_resolution else "_Ori"

    exp_name += f"_G{str(config.n_gpus)}"
    exp_name += f"_B{str(config.batch_size)}"

    exp_name += "_cw"

    # exp_name += "_{}_{}".format(config.dataset, config.run)
    # exp_name += "_FrusSize_{}".format(config.frustum_size)
    # exp_name += "_nRelations{}".format(config.n_relations)
    # exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    # if config.CE_ssc_loss:
    #     exp_name += "_CEssc"
    # if config.geo_scal_loss:
    #     exp_name += "_geoScalLoss"
    # if config.sem_scal_loss:
    #     exp_name += "_semScalLoss"
    # if config.fp_loss:
    #     exp_name += "_fpLoss"
    # if config.relation_loss:
    #     exp_name += "_CERel"
    # if config.context_prior:
    #     exp_name += "_3DCRP"

    # Setup dataloaders
    if config.dataset == "kitti":
        class_names = kitti_class_names
        max_epochs = 31
        logdir = config.kitti_logdir
        if config.low_resolution:
            full_scene_size = (128, 128, 16)
        else:
            full_scene_size = (256, 256, 32)
        project_scale = 2
        feature = 48 #cw250414 64
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
            low_resolution=config.low_resolution,
            sequence_length=int(config.sequence_length),
            use_event=config.use_event,
            use_bulk=config.use_bulk,
            use_token=config.use_token,
        )
            
    # elif config.dataset == "NYU":
    #     class_names = NYU_class_names
    #     max_epochs = 30
    #     logdir = config.logdir
    #     full_scene_size = (60, 36, 60)
    #     project_scale = 1
    #     feature = 200
    #     n_classes = 12
    #     class_weights = NYU_class_weights
    #     data_module = NYUDataModule(
    #         root=config.NYU_root,
    #         preprocess_root=config.NYU_preprocess_root,
    #         n_relations=config.n_relations,
    #         frustum_size=config.frustum_size,
    #         batch_size=int(config.batch_size / config.n_gpus),
    #         num_workers=int(config.num_workers_per_gpu * config.n_gpus),
    #         # low_resolution need to be added.
    #     )

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

    if int(config.sequence_length) == 1:
        if config.use_bulk == False:
            model = MonoScene(
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
            )
        elif config.use_bulk == True:
            model = AttentionalMonoScene(
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
                batch_size=config.batch_size)
                # use_bulk=config.use_bulk,)
        
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



    if config.enable_log:
        logger = WandbLogger(project="MonoScene_Optimization", name=exp_name)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        # TODO: register each parameter in wandb

        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                dirpath="checkpoints/",
                filename="best_model-{epoch:02d}-{val-mIoU:.5f}",
                save_top_k=2,
                mode="max",
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
