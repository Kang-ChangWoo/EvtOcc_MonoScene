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

import wandb

import numpy as np
from PIL import Image

def save_numpy_image(arr, save_path):
    # arr: (1,1,h,w)
    arr = np.squeeze(arr)  # (h, w)
    # arr = arr * 255.
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

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



class AttentionalMonoScene(pl.LightningModule):
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
        
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        # # (2) GETMultiScaleBackbone으로 감싸, multi-scale feature를 dict로 받도록 함
        # self.net_rgb = MonoSceneGET2DNetwork(
        #     get_config=get_config,
        #     edsa_config=edsa_config,
        #     out_scales=[1, 2, 4, 8, 16],  # 필요 스케일
        #     base_feature_dim=feature     # MonoScene가 기대하는 채널 수 (ex: 200)
        # )
        # log hyperparameters
        self.save_hyperparameters()

        # self.train_metrics = SSCMetrics(self.n_classes)
        # self.val_metrics = SSCMetrics(self.n_classes)
        # self.test_metrics = SSCMetrics(self.n_classes)

        self.train_metrics = DepthMetrics()
        self.val_metrics = DepthMetrics()
        self.test_metrics = DepthMetrics()

        self.width = 1224 # 1220
        self.height = 384 #370

        #norm_layer=partial(LayerNormFP32, eps=1e-6)

        # General Term
        group_num = 6 # 6
        embed_dim = 48
        num_features = int(embed_dim * 1 ** 1) # int(embed_dim * 2 ** i_layer) 2>1
        self.norm_layer = partial(LayerNormFP32, eps=1e-6)
        self.norm = self.norm_layer(num_features) #[-1]

        drop_rate = 0.1
        attn_drop = 0.1

        # Used for E2SRC
        patch_size = 24 # also, affect 'kernel_size' in GTE
        self.patch_size = patch_size

        # Used for GTE
        embed_split = 24 # Group과 같나? 24
        input_dim = 2 * embed_split * int(patch_size ** 2) // 4 # group_num 는 원래 없음;
        #input_dim = 2 * embed_split * int(patch_size ** 2) 
        hidden_dim = int(group_num * 64 / (group_num / 12)) 
        kernel_size = (3, 3) #if patch_size == 4 else (7, 7)
        
        # Used for EDSA
        depths=[2, 2, 8] 
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # self.visualize = False
        
        self.last_depth = None
        self.last_pred = None
        
        self.save_idx = 1
        self.isit_write = False

        
        # CW added
        # 1220, 370 > 1232, 372
        self.e2src = event_embed(shape=[1224, 384], batch_size=self.batch_size, group_num=group_num, patch_size=patch_size) # TODO image shape/ batch_size 등을 입력으로 받아와야 함.
        self.channel_embed = GTE(input_dim, hidden_dim, embed_dim, norm_layer=self.norm_layer, group_num=group_num, kernel_size=kernel_size) # (B C H W) > (B C H W)
        self.pos_drop = nn.Dropout(p=drop_rate) # TODO
        self.layer1 = BasicLayer(
                dim=int(embed_dim * 1 ** 1), #i_layer
                depth=depths[0], #depths[i_layer]
                num_heads=3, #num_heads[i_layer]
                window_size=8, #window_size[i_layer]
                mlp_ratio=4., #mlp_ratio
                drop=drop_rate,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                norm_layer=self.norm_layer,
                downsample=None, # cur_downsample_layer GTA 줄어봐
                use_checkpoint=False, #use_checkpoint[i_layer]
                init_values=1e-5, #init_values
                use_mlp_norm=False, #True if i_layer in use_mlp_norm_layers else False
                use_shift=True, #use_shift[i_layer]
                rpe_hidden_dim=512, #self.rpe_hidden_dim
                group_num=group_num,
                embed_dim=embed_dim
            )

        self.layer2 = BasicLayer(
                dim=int(embed_dim * 1 ** 1), #i_layer # 이게 다운 샘플이 있을 때 없을 때 작동하는 거 같다; 2 > 1 
                depth=depths[1], #depths[i_layer]
                num_heads=6, #num_heads[i_layer]
                window_size=8, #window_size[i_layer]
                mlp_ratio=4., #mlp_ratio
                drop=0.1, #drop_rate
                attn_drop=0.1, #attn_drop_rate
                drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                norm_layer=self.norm_layer,
                downsample=None, #cur_downsample_layer
                use_checkpoint=False, #use_checkpoint[i_layer]
                init_values=1e-5, #init_values
                use_mlp_norm=False, #True if i_layer in use_mlp_norm_layers else False
                use_shift=True, #use_shift[i_layer]
                rpe_hidden_dim=512, #self.rpe_hidden_dim
                group_num=group_num,
                embed_dim=embed_dim
            )
        
        self.layer3 = BasicLayer(
                dim=int(embed_dim * 1 ** 1), #i_layer
                depth=depths[1], #depths[i_layer]
                num_heads=12, #num_heads[i_layer]
                window_size=8, #window_size[i_layer] 8
                mlp_ratio=4., #mlp_ratio
                drop=0.1, #drop_rate
                attn_drop=0.1, #attn_drop_rate
                drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                norm_layer=self.norm_layer,
                downsample=None, #cur_downsample_layer
                use_checkpoint=False, #use_checkpoint[i_layer]
                init_values=1e-5, #init_values
                use_mlp_norm=False, #True if i_layer in use_mlp_norm_layers else False
                use_shift=True, #use_shift[i_layer]
                rpe_hidden_dim=512, #self.rpe_hidden_dim
                group_num=group_num,
                embed_dim=embed_dim
            )

        self.update_mlp = None

        self.convT_1 = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1,)
        self.convT_2 = nn.ConvTranspose2d(in_channels=48, out_channels=48, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1,) #output_padding=2,

        self.post_processing1 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size((self.height // self.patch_size, self.width // self.patch_size))),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0),
            )

        self.post_processing2 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size((self.height // (self.patch_size), self.width // (self.patch_size)))),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0),
            self.convT_1,
            )

        self.post_processing3 = nn.Sequential(
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size((self.height // (self.patch_size), self.width // (self.patch_size)))),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0),
            self.convT_1,
            self.convT_2,
            )

        self.refinenet3 = FeatureFusionBlock(48) # 1_4
        self.refinenet2 = FeatureFusionBlock(48) # 1_2
        self.refinenet1 = FeatureFusionBlock_L(48) # 1_1 # FeatureFusionBlock_L 폐기

        self.pos_embedding = nn.Parameter(torch.randn(1, int(1224 // 24) * int(384 // 24), embed_dim)) # (1224 / 24) * (384 / 24)
        # self.width = 1224 # 1220
        # self.height = 384 #370

        self.vit2d_rgb = ViT2D_RGB.build(basemodel_name='DPT_Hybrid', use_decoder=False, pretrained=True)

        self.head = nn.Sequential(
            nn.Conv2d(48, 48 // 2, kernel_size=3, stride=1, padding=1),
            #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(48 // 2, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
        '''
        self.head = nn.Sequential(
            nn.Conv2d(48, 48 // 2, kernel_size=3, stride=1, padding=1),
            #Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(48 // 2, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
        '''

        """ViT PATH"""
        model_type ='DPT_Hybrid'

        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        #midas.to(device)
        #midas.eval()

        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

        #self.encoder = MidasEncoder(midas)
        self.evt_encoder = MidasEncoder(midas)
        self.decoder = MidasDecoder(midas)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        #     teacher_encoder.eval()  # dropout 등도 비활성화

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        #input_batch = transform(img).to(device) #cv2 RGB 를 입력으로 받는 듯?

        #self.image

    def forward(self, batch):
        blk_evt = batch["img"] # (B, t, p , h, w) 라고 가정해야 할 듯?
        # Out: [1, diverse N, 4] // [Batch, Event streams, 4(t_p_h_w)]
        batch_size = len(blk_evt)
        # Out: 1 이여야함.

        blk_evt = self.transform(blk_evt) # [B, t, p , h, w] > [B, t, p * h * w]

        #group_evt = blk_evt #.permute(0, 2, 3, 4, 1) # [B, t, p , h, w] > [B, p , h, w, t]
        """
        out = {}
        
        # x_rgb = self.net_rgb(img) # deprecated
        group_evt = self.e2src(blk_evt) #TODO maybe the collate _fn make
        #print("Result of E2SRC: ", group_evt.shape)
        # In: [1, diverse N, 4] // [Batch, Event streams, 4(t_p_h_w)]
        # Out: [1, 384, 92, 305] // [1, group_num * '2' * (patch_size ** 2), H // patch_size, W // patch_size]
        # 1, 10 * 2  * 16 * 16, 77, 23
        # Actual: [1, 12 * 2 * (4 ** 2), 92, 305]

        group_evt = self.channel_embed(group_evt)  # Broken
        # In: [1, 12 * 2 * (4 ** 2), 92, 305]
        # Out: [1, 48, 92, 305]
        #print("Result of channel_embed: ", group_evt.shape)

        
        Wh, Ww = group_evt.size(2), group_evt.size(3) #TODO I know about patch and its size 
        group_evt = group_evt.flatten(2).transpose(1, 2)
        # Out: [1, 48, 92 * 305] (flatten의 결과)
        # Out: [1, 48, 28060] (transpose의 결과)

        group_evt = self.pos_drop(group_evt + self.pos_embedding)
        #print("Result of pos_drop: ", group_evt.shape)
        # In: [1, 28060, 48]
        # Out: [1, 28060, 48]


        x_out1, H, W, x1, Wh, Ww = self.layer1(group_evt, Wh, Ww) # 7038, 96
        # print("Needed A!!!", H, W, Wh, Ww)
        # print("Result of layer1: ", x_out1.shape)
        x_out2, H, W, x2, Wh, Ww = self.layer2(x1, Wh, Ww) # 7038, 96
        # print("Needed B!!!", H, W, Wh, Ww)
        # print("Result of layer2: ", x_out2.shape)
        x_out3, H, W, x3, Wh, Ww = self.layer3(x2, Wh, Ww) # 7038, 96
        # print("Needed C!!!", H, W, Wh, Ww)
        # print("Result of layer3: ", x_out3.shape)

        x_last = self.norm(x_out3)

        '''
        Pose-processing 하는, 순서가 DPT랑 다르긴 한데, 일단 연결해보자.
        '''
        img_x3 = self.post_processing1(x_last) # B, 7038, 96 > B, 7038, 92, 305
        out_x3 = self.refinenet3(img_x3) # interpolated into 184, 610
        
        img_x2 = self.post_processing2(x_out2) # B, 7038, 96 > B, 7038, 184, 610
        out_x2 = self.refinenet2(out_x3, img_x2) # interpolated into 368, 1220

        img_x1 = self.post_processing3(x_out1) # B, 7038, 96 > B, 7038, 368, 1220
        out_x1 = self.refinenet1(out_x2, img_x1) # interpolated into 370, 1220
        # print("out_x1!!: ", out_x1.shape)

        # x_rgb = {}
        # x_rgb['1_1'] = out_x1
        pred_depth = self.head(out_x1)

        x_rgb = self.net_rgb(group_evt)
        """        
        
        """
        x_rgb = self.net_rgb(blk_evt) #TODO
        pred_depth = self.head(x_rgb['1_1']) #TODO
        """

        # imgs = []   
        # for img_t in blk_evt:                         # 배치 루프
        #     img_np = img_t.permute(1, 2, 0)     # C,H,W → H,W,C
        #     img_np = img_np.cpu().numpy()       # torch → numpy
        #     img_np = img_np[:, :, ::-1]         # RGB→BGR if needed
        #     imgs.append(interpolate_prediction(img_np)) # MiDaS transform


        # x_stack = torch.stack(imgs).to(x.device)
        # print("shape", x_stack.shape)



        #blk_evt = interpolate_prediction(blk_evt)
        #encoded_features = self.vit2d_rgb(blk_evt)
        
        feats = self.evt_encoder(blk_evt)
        #with torch.no_grad():
        #    teacher_feats = self.encoder(blk_evt) # TODO 
        #out['feats'] = feats
        #out['teacher_feats'] = teacher_feats

        out['pred'] = self.decoder(feats[0], feats[1], feats[2], feats[3])
        #out['teacher_pred'] = self.decoder(teacher_feats[0], teacher_feats[1], teacher_feats[2], teacher_feats[3])
        #out['pred'] = pred_depth
        
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        # ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]
        depth_pred = out_dict["pred"] # (batch_size, n_classes, H, W, D)
        #depth_T_pred = out_dict["teacher_pred"] # (batch_size, n_classes, H, W, D)
        depth = batch["depth"]

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
            loss_depth = depth_loss(depth, depth_pred)
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

        if (bs == 1) and (self.isit_write == False):
            self.last_pred = y_pred
            self.last_depth = y_true

            # print("1")

            # print("y_pred: ", y_pred.shape)
            # print("y_true: ", y_true.shape)

            # save_numpy_image(self.last_pred, f'/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/unet_pred_{self.save_idx}_epoch.png')
            # save_numpy_image(self.last_depth, f'/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/unet_depth_{self.save_idx}_epoch.png')

            save_numpy_image(self.last_pred, f'/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/vitDPT_pred_{self.save_idx}_epoch.png')
            save_numpy_image(self.last_depth, f'/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/vitDPT_depth_{self.save_idx}_epoch.png')

            self.save_idx += 1
            self.isit_write = True

        elif (bs != 1) and (self.isit_write == False):
            self.last_pred = y_pred[0]
            self.last_depth = y_true[0]

            # print("2")

            # print("y_pred: ", y_pred[0].shape)
            # print("y_true: ", y_true[0].shape)

            # save_numpy_image(self.last_pred, f'/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/unet_pred_{self.save_idx}_epoch.png')
            # save_numpy_image(self.last_depth, f'/root/storage/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/unet_depth_{self.save_idx}_epoch.png')

            save_numpy_image(self.last_pred, f'/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/vitDPT_pred_{self.save_idx}_epoch.png')
            save_numpy_image(self.last_depth, f'/root/dev0/implementation/shared_evtOcc/MonoScene_depth_v2/kitti_log/vitDPT_depth_{self.save_idx}_epoch.png')

            self.save_idx += 1
            self.isit_write = True

        # else:
        #     print("3")


        metric.add_batch(y_pred, y_true)

        # print("y_true: ", y_true.shape)
        # print("y_pred: ", y_pred.shape)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

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