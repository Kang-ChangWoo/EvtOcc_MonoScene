import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from monoscene.models.unet3d_nyu import UNet3D as UNet3DNYU
from monoscene.models.unet3d_kitti import UNet3D as UNet3DKitti
from monoscene.loss.sscMetrics import SSCMetrics, DepthMetrics
from monoscene.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss, depth_loss, l2_loss
from monoscene.models.flosp import FLoSP
from monoscene.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np

from monoscene.models.vit2d_rgb import ViT2D_RGB
from monoscene.models.unet2d import UNet2D
from monoscene.models.event_token import event_embed
from monoscene.models.GET import GTE, LayerNormFP32, BasicLayer, GTA, FeatureFusionBlock, ResidualConvUnit, FeatureFusionBlock_L
from torch.optim.lr_scheduler import MultiStepLR
#from NaViT import NaViT

from torchvision.transforms.functional import to_pil_image
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

import wandb

import numpy as np
from PIL import Image
import os 
import matplotlib.pyplot as plt

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class DepthHead(nn.Module):
    def __init__(self, in_channels, method="sum"):
        """
        in_channels: net_rgb['1_1'] 의 채널 수 (예: 256)
        method: "sum" 또는 "concat" 중 선택
        """
        super().__init__()
        self.method = method
        
        if method == "concat":
            # 5개 스케일을 concat 할 경우 총 채널 수 = in_channels * 5
            self.project = nn.Conv2d(in_channels * 5, 1, kernel_size=1)
        else:
            # sum fusion 한 뒤 그대로 1채널로 투영
            self.project = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feats):
        # feats: dict with keys "1_1","1_2","1_4","1_8","1_16"
        # 기준 해상도 (H,W)
        H, W = feats["1_1"].shape[2:]
        
        if self.method == "concat":
            ups = []
            for key in ["1_1","1_2","1_4","1_8","1_16"]:
                x = feats[key]
                if x.shape[2:] != (H, W):
                    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
                ups.append(x)
            fused = torch.cat(ups, dim=1)        # (B, C*5, H, W)
        else:  # sum fusion
            fused = 0
            for key in feats:
                x = feats[key]
                if x.shape[2:] != (H, W):
                    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
                fused = fused + x                   # (B, C, H, W)
        
        # 1×1 conv 로 (B,1,H,W) 출력
        depth = self.project(fused)
        return depth


class FrontAdapter(nn.Module):
    """
    3‑bin event 이미지를 Conv 두 층만으로 RGB 분포와 비슷한
    스케일·통계로 변환한다. (참고: TSCFormer, BRENet stem)
    """
    def __init__(self, in_ch=3, mid_ch=32, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.Tanh()                       # → [‑1, 1], MiDaS 정규화와 동일
        )

    def forward(self, x):
        return self.net(x.float())          # (B,3,H,W) → (B,3,H,W)

def save_depth_color(depth, path, max_depth=40.0, cmap='turbo'):
    """
    depth: (H, W) float32 [m]
    max_depth: 시각화 최장 거리 [m]
    cmap: matplotlib 컬러맵 이름
    """
    norm = np.clip(depth / max_depth, 0, 1)
    rgb  = plt.get_cmap(cmap)(norm)[:, :, :3] * 255   # RGBA→RGB
    Image.fromarray(rgb.astype(np.uint8)).save(path)

def save_numpy_image(arr, save_path):
    # arr: (1,1,h,w)
    arr = np.squeeze(arr)  # (h, w)
    # arr = arr * 255.
    arr = (arr * 255 / 60).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    img.save(save_path)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class MidasEncoder(nn.Module):
    def __init__(self, midas):
        super().__init__()
        self.forward_transformer = midas.forward_transformer  # 공식 feature extractor
        self.pretrained = midas.pretrained                   # 내부에 model + act_postprocess 포함

    def forward(self, x):
        """
        Returns:
            features: list of 4 feature maps after act_postprocess1~4
                      Each has shape [B, C, H, W] ready for decoder
        """
        return self.forward_transformer(self.pretrained, x)


class MidasDecoder(nn.Module):
    def __init__(self, midas):
        super().__init__()
        self.layer1_rn = midas.scratch.layer1_rn
        self.layer2_rn = midas.scratch.layer2_rn
        self.layer3_rn = midas.scratch.layer3_rn
        self.layer4_rn = midas.scratch.layer4_rn

        self.refinenet1 = midas.scratch.refinenet1
        self.refinenet2 = midas.scratch.refinenet2
        self.refinenet3 = midas.scratch.refinenet3
        self.refinenet4 = midas.scratch.refinenet4

        self.output_conv = midas.scratch.output_conv

    def forward(self, x1, x2, x3, x4):
        x1_rn = self.layer1_rn(x1)
        x2_rn = self.layer2_rn(x2)
        x3_rn = self.layer3_rn(x3)
        x4_rn = self.layer4_rn(x4)

        out = self.refinenet4(x4_rn)
        out = self.refinenet3(out + x3_rn)
        out = self.refinenet2(out + x2_rn)
        out = self.refinenet1(out + x1_rn)
        out = self.output_conv(out)
        return out

class ScalarAffine(nn.Module):
    def __init__(self, init_scale=1.0, init_shift=0.0):
        super().__init__()
        # 스칼라 a (scale) 파라미터: 초깃값 1.0
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        # 스칼라 b (shift) 파라미터: 초깃값 0.0
        self.shift = nn.Parameter(torch.tensor(init_shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 스칼라 또는 스칼라를 원소로 가지는 텐서
        return self.scale * x + self.shift

class depth_anything(nn.Module):
    def __init__(self, encoder='vitb'):
        super().__init__()
        #DEVICE = 'cuda:4' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.encoder = encoder  # 'vits', 'vitb', 'vitl', 'vitg'

        self.model = DepthAnythingV2(**model_configs[self.encoder])
        #self.model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{encoder}.pth')) #, map_location='cuda:4'
        self.model.load_state_dict(torch.load(f'/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2/checkpoints/depth_anything_v2_{encoder}.pth')) #, map_location='cuda:4'

        #self.model.train() #/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2

    def forward(self, x):
        #return self.model.infer_image(x)
        return self.model(x)


class EventFrameMonoScene(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        feature,
        class_weights,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        project_res=[],
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
        batch_size=1, #CW
        model_type='VisionTransformer', 
        root_fpath='/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/monoscene',
        exp_title='default',
    ):
        super().__init__()

        self.project_res = project_res
        # self.fp_loss = fp_loss
        self.fp_loss = False
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(full_scene_size, project_scale=self.project_scale, dataset=self.dataset)
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        # if self.dataset == "NYU":
        #     self.net_3d_decoder = UNet3DNYU(
        #         self.n_classes,
        #         nn.BatchNorm3d,
        #         n_relations=n_relations,
        #         feature=feature,
        #         full_scene_size=full_scene_size,
        #         context_prior=context_prior,
        #     )
        # elif self.dataset == "kitti":
        #     self.net_3d_decoder = UNet3DKitti(
        #         self.n_classes,
        #         nn.BatchNorm3d,
        #         project_scale=project_scale,
        #         feature=feature,
        #         full_scene_size=full_scene_size,
        #         context_prior=context_prior,
        #     )
            
        # self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)
        # get_model = build_model(get_config())  # 가상 예시
        
        
        self.isit_write = False
        self.save_idx = 0

        self.save_hyperparameters()

        # self.train_metrics = SSCMetrics(self.n_classes)
        # self.val_metrics = SSCMetrics(self.n_classes)
        # self.test_metrics = SSCMetrics(self.n_classes)

        self.train_metrics = DepthMetrics()
        self.val_metrics = DepthMetrics()
        self.test_metrics = DepthMetrics()

        self.adapter = FrontAdapter(in_ch=3, mid_ch=32, out_ch=3)

        self.width = 1224 # 1220
        self.height = 384 #370


        self.model_type = model_type

        if self.model_type == 'UNet':
            self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)
            self.depth_head = DepthHead(in_channels=64, method="sum")

        elif self.model_type == 'VisionTransformer':
            """ViT PATH"""
            model_type = 'DPT_Large'

            midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
            #midas.train()

            #midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            #midas.apply(initialize_weights)

            self.evt_encoder = MidasEncoder(midas)
            self.decoder = MidasDecoder(midas)

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            #input_batch = transform(img).to(device) #cv2 RGB 를 입력으로 받는 듯?

            self.evt_encoder.train()
            self.decoder.train()    

        elif self.model_type == 'DepthAnything':
            self.model = depth_anything(encoder='vitb')
            self.model.train()



        self.root_fpath = root_fpath
        self.exp_title = exp_title
        self.affine_transform = ScalarAffine(init_scale=100000.0, init_shift=0.0)
        #self.affine_transform = ScalarAffine(init_scale=0.000000005, init_shift=0.0) #100000.0



    def forward(self, batch):
        blk_evt = batch["img"] # (B, t, p , h, w) 라고 가정해야 할 듯?
        # Out: [1, diverse N, 4] // [Batch, Event streams, 4(t_p_h_w)]
        #ev = ev / 255.0 if ev.max() > 1 else ev
        
        batch_size = len(blk_evt)
        device = blk_evt.device
        #("blk_evt: ", blk_evt.shape)

        out = {}

        if self.model_type == 'VisionTransformer':
            preds = []

            for img_tensor in blk_evt:  # blk_evt: [B, 3, H, W]                
                # Tensor → NumPy image
                img = img_tensor.permute(1, 2, 0).cpu().numpy()

                #print("img: ", img.shape)
                #
                #img = np.clip(img, 0, 255).astype(np./int8)
                orig_size = img.shape[:2]  # (H, W)

                #print(f"img_tensor type: {type(img_tensor)}")

                # MiDaS transform
                sample = self.transform(img)
                #print(f"data type: {type(sample)}")
                input_tensor = sample.to(device).requires_grad_()  # [1, 3, H, W]
                #input_tensor = self.adapter(input_tensor) 
                #print("input_tensor: ", input_tensor.shape)

                # MiDaS encoder-decoder
                feats = self.evt_encoder(input_tensor)
                pred = self.decoder(feats[0], feats[1], feats[2], feats[3])  # [1, 1, h, w]

                # Interpolate to original size
                pred_resized = torch.nn.functional.interpolate(
                    pred, size=orig_size, mode="bicubic", align_corners=False
                )  # [1, 1, H_orig, W_orig]


                # ── inverse-depth → depth 변환 ─────────────────────────
                eps = 1e-6
                depth_resized = 1.0 / (pred_resized.clamp(min=eps))
                results = self.affine_transform(depth_resized)

                preds.append(results)

            # [B, 1, H_orig, W_orig]
            prediction = torch.cat(preds, dim=0).requires_grad_()


        elif self.model_type == 'UNet':
            prediction = self.net_rgb(blk_evt) # (B, 1, h, w)
            prediction = self.depth_head(prediction)


        elif self.model_type == 'DepthAnything':
            #prediction = self.model(blk_evt)
            #prediction = self.model(blk_evt)
            # 1) 원본 DepthAnything 모델 출력: shape [B, H1, W1]
            prediction = self.model(blk_evt)
            # 2) 채널 차원 추가: [B, 1, H1, W1]
            prediction = prediction.unsqueeze(1)
            # 3) 원하는 크기 (370, 1220) 로 리사이즈
            prediction = F.interpolate(
                prediction,
                size=(370, 1220),
                mode="bilinear",
                align_corners=False
            )

        out['pred'] = prediction

        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        fid = ""
        out_dict = self(batch)
        # ssc_pred = out_dict["ssc_logit"]
        #target = batch["target"]
        depth_pred = out_dict["pred"] # (batch_size, n_classes, H, W, D)
        #depth_T_pred = out_dict["teacher_pred"] # (batch_size, n_classes, H, W, D)
        depth = batch["depth"]

        # print(f"type!!: {type(depth_pred)}")
        # print(f"type!!: {type(depth)}")
        # print(f"depth_pred: {depth_pred.shape}")
        # print(f"depth: {depth.shape}")


        #feats = out_dict["feats"]
        #teacher_feats = out_dict["teacher_feats"]


        # print('target: ', target.shape)
        # print('depth: ', depth.shape)
        # print('depth_pred: ', depth_pred.shape)

        if self.context_prior:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]

            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + "/loss_relation_ce_super",
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(batch["frustums_class_dists"]).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if True:
            loss_depth = depth_loss(depth_pred, depth)
            loss += loss_depth
            self.log(
                step_type + "/loss_depth",
                loss_depth.detach(),
                on_epoch=True,
                sync_dist=True,
            )

            # loss_pseudo_depth = depth_loss(depth_pred, depth_T_pred)
            # loss += loss_pseudo_depth
            # self.log(
            #     step_type + "/loss_pseudo_depth",
            #     loss_depth.detach(),
            #     on_epoch=True,
            #     sync_dist=True,
            # )

            # loss_feats = l2_loss(feats, teacher_feats)
            # loss += loss_feats
            # self.log(
            #     step_type + "/loss_feats",
            #     loss_feats.detach(),
            #     on_epoch=True,
            #     sync_dist=True,
            # )

        y_true = depth.cpu().numpy()
        y_pred = depth_pred.detach().cpu().numpy()

        print(f"!!min, max of y_true: {np.min(y_true)}, {np.max(y_true)}")
        print(f"!!min, max of y_pred: {np.min(y_pred)}, {np.max(y_pred)}")

        if (bs == 1) and (self.isit_write == False):
            fid = batch["frame_id"][0]
            print(f"fid!!: {fid}")

            self.last_pred = y_pred
            self.last_depth = y_true

            os.makedirs(f'{self.root_fpath}/{self.exp_title}', exist_ok=True)

            save_numpy_image(self.last_pred, f'{self.root_fpath}/{self.exp_title}/pred_{self.save_idx}_epoch_{1}_{fid}th.png')
            save_numpy_image(self.last_depth, f'{self.root_fpath}/{self.exp_title}/gt_{self.save_idx}_epoch_{1}_{fid}th.png')

            self.save_idx += 1
            self.isit_write = True

        elif (bs != 1) and (self.isit_write == False):
            for i in range(bs):
                fid = batch["frame_id"][i]
                print(f"fid!!: {fid}")

                self.last_pred = y_pred[i]
                self.last_depth = y_true[i]

                os.makedirs(f'{self.root_fpath}/{self.exp_title}', exist_ok=True)

                save_numpy_image(self.last_pred, f'{self.root_fpath}/{self.exp_title}/pred_{self.save_idx}_epoch_{i}_{fid}th.png')
                save_numpy_image(self.last_depth, f'{self.root_fpath}/{self.exp_title}/gt_{self.save_idx}_epoch_{i}_{fid}th.png')

                self.save_idx += 1
                self.isit_write = True

        for k in range(bs):
            metric.add_batch(y_pred[k], y_true[k])

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        # print("loss.requires_grad:", loss.requires_grad)      # True 가 나와야 정상
        # print("loss.grad_fn:", type(loss.grad_fn))
        # print("depth_pred.requires_grad:", depth_pred.requires_grad)
        # print("depth_loss.requires_grad:", loss_depth.requires_grad)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)
        

    def validation_epoch_end(self, outputs):
        # print("last evaluation")
        # wb_img  = wandb.Image(self.last_depth,  caption=f"epoch__input")
        # wb_pred = wandb.Image(self.last_pred, caption=f"epoch__pred")
        
        # # experiment.log으로 단 한 번만 기록
        # self.logger.experiment.log({
        #     "example/input": [wb_img],
        #     "example/pred":  [wb_pred],
        # })

        # if self.trainer.is_global_zero:
        #     # 이미지 변환 및 로깅
        #     self.log_image(
        #         key="validation/images",
        #         images=[self.last_depth.cpu().numpy(), 
        #                 self.last_pred.cpu().numpy()],
        #         caption=["Input Depth", "Predicted Depth"]
        #     )

        # ─────────────────────────────────────────────────────────────
        # 2) DepthMetrics 지표 로깅
        # ─────────────────────────────────────────────────────────────
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        self.isit_write = False

        for prefix, metric in metric_list:
            stats = metric.get_results()  # DepthMetrics.get_results()

            # Depth 평가 지표 로깅
            self.log(f"{prefix}/abs_rel",  stats['abs_rel'],   sync_dist=True)
            self.log(f"{prefix}/sq_rel",   stats['sq_rel'],    sync_dist=True)
            self.log(f"{prefix}/rmse",     stats['rmse'],      sync_dist=True)
            self.log(f"{prefix}/rmse_log", stats['rmse_log'],  sync_dist=True)
            self.log(f"{prefix}/a1",       stats['a1'],        sync_dist=True)
            self.log(f"{prefix}/a2",       stats['a2'],        sync_dist=True)
            self.log(f"{prefix}/a3",       stats['a3'],        sync_dist=True)

            metric.reset()

    # def validation_epoch_end(self, outputs):
    #     metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

    #     for prefix, metric in metric_list:
    #         stats = metric.get_stats()
    #         for i, class_name in enumerate(self.class_names):
    #             self.log(
    #                 "{}_SemIoU/{}".format(prefix, class_name),
    #                 stats["iou_ssc"][i],
    #                 sync_dist=True,
    #             )
    #         self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
    #         self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
    #         self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
    #         self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
    #         metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]



def interpolate_prediction(prediction, tgt_size=(370,1220)): # self, 

    out = F.interpolate(
        prediction,
        size=tgt_size,
        mode="bicubic",
        align_corners=False
    )

    return out


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x