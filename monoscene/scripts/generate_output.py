import numpy as np
import pickle
import hydra
import torch
import os
from PIL import Image

from monoscene.data.semantic_kitti.kitti_dm import KittiDataModule
from monoscene.loss.sscMetrics import SSCMetrics, DepthMetrics
#from monoscene.models.monoscene import MonoScene
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm
from monoscene.models.monoscene_modeB import EventFrameMonoScene
from hydra.utils import get_original_cwd


def save_depth_color(depth, path, max_depth=40.0, cmap='turbo'):
    """
    depth: (H, W) float32 [m]
    max_depth: 시각화 최장 거리 [m]
    cmap: matplotlib 컬러맵 이름
    """
    norm = np.clip(depth / max_depth, 0, 1)
    rgb  = plt.get_cmap(cmap)(norm)[:, :, :3] * 255   # RGBA→RGB
    Image.fromarray(rgb.astype(np.uint8)).save(path)

def save_numpy_image(x, save_path, max_val=60.0):
    """
    Save a tensor or array as an image (.png).
    - x: torch.Tensor or array-like, shape could be (1,H,W), (H,W), (H,W,3), etc.
    - save_path: full path ending in .png
    - max_val: for scaling grayscale images
    """
    # -- 1) to numpy --
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # -- 2) squeeze out any size-1 dims --
    arr = np.squeeze(arr)

    # -- 3) handle different ranks --
    if arr.ndim == 2:
        # grayscale
        img_arr = (arr * 255.0 / max_val).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        # RGB or RGBA
        img_arr = (arr * 255.0 / max_val).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    else:
        # e.g. (C,H,W) with C!=3: collapse channels
        # take mean over all extra dims
        # result: (H,W)
        collapsed = arr.mean(axis=tuple(range(arr.ndim - 2)))
        img_arr = (collapsed * 255.0 / max_val).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)

    # -- 4) save --
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def save_numpy(x, save_path):
    """
    Save a PyTorch tensor or array-like as a .npy file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    np.save(save_path, arr)

# Function for save the inference results
@hydra.main(config_name="../config/monoscene.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    # Semantic KITTI setup
    config.batch_size = 1
    feature = 64
    project_scale = 2
    full_scene_size = (256, 256, 32)

    # data_module = KittiDataModule(
    #     root=config.kitti_root,
    #     preprocess_root=config.kitti_preprocess_root,
    #     frustum_size=config.frustum_size,
    #     batch_size=int(config.batch_size / config.n_gpus),
    #     num_workers=int(config.num_workers_per_gpu * config.n_gpus),)

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

    data_module.setup()
    # data_loader = data_module.val_dataloader()
    data_loader = data_module.test_dataloader() # use this if you want to infer on test set

    # Load pretrained models
    #model_path = os.path.join(get_original_cwd(), "trained_models", "monoscene_kitti.ckpt") # Should Edit Here !!
    model_path = '/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/outputs/2025-05-19/17-42-15/checkpoints/best_model-epoch=26-val-abs_rel=0.00000.ckpt'
    model = EventFrameMonoScene.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
        model_type='VisionTransformer', 
        root_fpath='/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/monoscene',
        exp_title='default')

    model.cuda()
    model.eval()

    # Save prediction and additional data [test batch = 1 !!]
    depth_test_metrics = DepthMetrics()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch['evF'] = batch['evF'].cuda()            
            # img, depth = batch["evF"].cuda(), batch["depth"].cuda()
            depth = batch["depth"].cuda()
            depth_pred = model(batch)["pred"]
            
            depth_test_metrics.add_batch(depth_pred, depth)

            print(f"!!!{batch['evF'][0,:,:,:].shape}")

            fid = batch['frame_id']
            fpath = os.path.join(config.output_path, "_".join(model_path.split('/')[-2:]))
            os.makedirs(fpath, exist_ok=True)

            # 이미지로 저장
            save_numpy_image(batch['evF'][0],       f"{fpath}/{fid}_evF_gt.png")
            save_numpy_image(depth_pred[0],         f"{fpath}/{fid}_evF_pred.png")
            save_numpy_image(depth[0],              f"{fpath}/{fid}_depth_gt.png")

            # .npy 로 저장
            save_numpy(batch['evF'][0],             f"{fpath}/{fid}_evF_gt.npy")
            save_numpy(depth_pred[0],               f"{fpath}/{fid}_evF_pred.npy")
            save_numpy(depth[0],                    f"{fpath}/{fid}_depth_gt.npy")

        depth_performance = depth_test_metrics.get_results()
        print(depth_performance)


if __name__ == "__main__":
    main()