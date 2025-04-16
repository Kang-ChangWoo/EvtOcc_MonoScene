import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from monoscene.models.unet3d_nyu import UNet3D as UNet3DNYU
from monoscene.models.unet3d_kitti import UNet3D as UNet3DKitti
from monoscene.loss.sscMetrics import SSCMetrics
from monoscene.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
from monoscene.models.flosp import FLoSP
from monoscene.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np

from monoscene.models.unet2d import UNet2D
from monoscene.models.event_token import event_embed
from monoscene.models.GET import GTE, LayerNormFP32, BasicLayer, GTA, FeatureFusionBlock, ResidualConvUnit, FeatureFusionBlock_L
from torch.optim.lr_scheduler import MultiStepLR


"""
[A] Basic MonoScene (to compare)

"""
class MonoScene(pl.LightningModule):
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
    ):
        super().__init__()

        self.project_res = project_res
        self.fp_loss = fp_loss
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

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    # 없애면 오류가 나긴 한다.
    # 얘가 함수 스스로를 override 해서, 스스로를 부르면 호출된단다.
    def forward(self, batch):
        img = batch["img"]
        batch_size = len(img)
        out = {}

        x_rgb = self.net_rgb(img)
        # x_rgb = dict_keys(['1_1', '1_2', '1_4', '1_8', '1_16'])
        
        # Each size
        # torch.Size([1, 200, 480, 640])
        # torch.Size([1, 200, 240, 320])
        # torch.Size([1, 200, 120, 160])
        # torch.Size([1, 200, 60, 80])
        # torch.Size([1, 200, 30, 40])
        
        x3ds = []

        for i in range(batch_size):
            x3d = None
            
            for scale_2d in self.project_res: # 1, 2, 4, 8
                scale_2d = int(scale_2d)
            
                # Loader에서 projected_pix_이 1,2,4,8로 각각 저장되어 있어야 한다. 
                # fov_mask도 마찬가지로 구해져 있어야 한다.
                # shape of projected_pix = (129600, 2) size
                projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                                
                fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()

                # Sum all the 3D features
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x_rgb["1_" + str(scale_2d)][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )               
                
            x3ds.append(x3d)

        # x3d size: [1, 200, 60, 36, 60] size:,  torch.Size([1, 64, 128, 128, 16])
        input_dict = {"x3d": torch.stack(x3ds),}
        
        out = self.net_3d_decoder(input_dict)

        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

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

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

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


'''
[B] Wrapping LSTM

'''

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2  # 차원 유지 패딩 적용
        self.bias = bias

        # ConvLSTM 연산 (4Gates: input, forget, output, cell)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,  # H, W 유지
            bias=self.bias
        )

        # **1x1 Residual Connection 추가**
        # self.residual = nn.Conv2d(
        #     in_channels=self.input_dim,
        #     out_channels=self.hidden_dim,
        #     kernel_size=1,  # 1x1 Conv
        #     padding=0,      # 차원 유지
        #     bias=self.bias
        # )

    def forward(self, x, h_prev, c_prev):
        batch_size, _, H, W = x.shape

        if h_prev is None:
            h_prev = torch.zeros_like(x)
            # h_prev = torch.zeros((batch_size, self.hidden_dim, H, W), device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros_like(x)
            # c_prev = torch.zeros((batch_size, self.hidden_dim, H, W), device=x.device, dtype=x.dtype)

        # **1x1 Residual Connection**
        # x_res = self.residual(x)  # (batch, hidden_dim, H, W)

        # Concatenate x and h_prev along channel axis
        combined = torch.cat([x, h_prev], dim=1)
        # Compute LSTM Gates
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        # Input gate
        i = torch.sigmoid(cc_i)
        # Forget gate
        f = torch.sigmoid(cc_f)
        # Output gate
        o = torch.sigmoid(cc_o)
        # Cell gate
        g = torch.tanh(cc_g)

        # Compute new cell state
        c_next = f * c_prev + i * g
        # Compute new hidden state
        h_next = o * torch.tanh(c_next)

        # **Residual Connection 추가**
        h_next = h_next #+ x_res  # Skip Connection 추가
        return h_next, c_next

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.bias = bias

        # GRU-style gates (Reset and Update)
        self.conv_gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,  # Reset gate, Update gate
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # 1x1 Residual Connection 추가
        self.residual = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            padding=0,
            bias=self.bias
        )

    def forward(self, x, h_prev):
        batch_size, _, H, W = x.shape

        if h_prev is None:
            h_prev = torch.zeros((batch_size, self.hidden_dim, H, W), device=x.device, dtype=x.dtype)

        x_res = self.residual(x)  # Residual connection

        # Compute Reset and Update gates
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        r, z = torch.split(gates, self.hidden_dim, dim=1)

        r = torch.sigmoid(r)  # Reset gate
        z = torch.sigmoid(z)  # Update gate

        # Compute candidate hidden state
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(combined_candidate))

        # Compute new hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Residual connection
        h_next = h_next + x_res

        return h_next

# 기존의 forward 메서드를 ConvLSTMCell을 사용하도록 수정합니다.
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True, num_layers=1):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, bias) for i in range(num_layers)])

    def forward(self, x, h_prev=None, c_prev=None):
        h_next, c_next = [], []

        for i, layer in enumerate(self.layers):
            h_i = h_prev if h_prev is not None else None
            c_i = c_prev if c_prev is not None else None
            h_i, c_i = layer(x, h_i, c_i)
            x = h_i  # 다음 레이어로 전달
            h_next.append(h_i)
            c_next.append(c_i)

        return h_next[0], c_next[0] # multi-layer deprecated.

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True, num_layers=1):
        super(ConvGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            ConvGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, bias)
            for i in range(num_layers)
        ])

    def forward(self, x, h_prev=None):
        h_next = []

        for i, layer in enumerate(self.layers):
            h_i = h_prev if h_prev is not None else None
            h_i = layer(x, h_i)
            x = h_i  # 다음 레이어로 전달
            h_next.append(h_i)

        return h_next[0]  # 단층 구조

class RecurrentMonoScene(pl.LightningModule):
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
        sequence_length=5,  # 시퀀스 길이 반영
    ):
        super().__init__()

        self.project_res = project_res
        self.fp_loss = fp_loss
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
        self.sequence_length = sequence_length

        print(f"feature: {feature}")
        print(f"feature(type): {type(feature)}")
        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        # self.feature = feature  # 기본 feature 크기


        # 해상도별 ConvGRU2D 저장
        self.RNNs = nn.ModuleDict({
            "1_1": ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, bias=True),
            "1_2": ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, bias=True),
            "1_4": ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, bias=True),
            "1_8": ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, bias=True),
            "1_16": ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, bias=True),
        })

        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

    def forward(self, batch): #prev_states=None
        img = batch["img"] # 3, 1, 3, 370, 1220
        sequence_length, batch_size, _, _, _ = img.shape

        out = {}
        #prev_states = {i: None for i in resolution_keys} if prev_states is None else prev_states

        res_keys = ["1_1", "1_2", "1_4", "1_8", "1_16"]

        cell_states = {k: None for k in res_keys}
        hidden_states = {k: None for k in res_keys}
        prev_h = None
        prev_c = None


        for seq in range(sequence_length):
            x_rgb = self.net_rgb(img[seq,]) # (1,3,370,1220)
            res_keys = x_rgb.keys() # ["1_1", "1_2", "1_4", "1_8", "1_16"]
            batch_size = img[seq,].shape[0]

            # x_rgb[1_1]: torch.Size([1, 64, 370, 1220])                                                                                                                                                                     
            # x_rgb[1_2]: torch.Size([1, 64, 185, 610])                                                              
            # x_rgb[1_4]: torch.Size([1, 64, 93, 305])                                                                                                                                                                       
            # x_rgb[1_8]: torch.Size([1, 64, 47, 153])                                                               
            # x_rgb[1_16]: torch.Size([1, 64, 24, 77])

            # Each size
            # torch.Size([1, 200, 480, 640])
            # torch.Size([1, 200, 240, 320])
            # torch.Size([1, 200, 120, 160])
            # torch.Size([1, 200, 60, 80])
            # torch.Size([1, 200, 30, 40])

            # TODO: 원래 x_rgb가 다층 구조니까 얘네만 업데이트하는 모듈을 넣으면 된다.

            # for i in range(batch_size): # TODO batch 연산 고려?
            for k in res_keys:
                #features = x_rgb[k][i] # (batch_size, feature, H, W) TODO batch 연산 고려?
                features = x_rgb[k]
                
                # For LSTM
                if cell_states[k] is None:
                    hidden_states[k], cell_states[k] = self.RNNs[k](features, prev_h, prev_c)
                else:
                    prev_h = hidden_states[k]
                    prev_c = cell_states[k]
                    #LSTM
                    hidden_states[k], cell_states[k] = self.RNNs[k](features, prev_h, prev_c)


                # For GRU
                # if hidden_states[k] is None:
                #     hidden_states[k] = self.RNNs[k](features, hidden_states[k])
                # else:
                #     hidden_states[k] = self.RNNs[k](features, hidden_states[k])

            # print(f"hidden_states: {hidden_states[k].shape}") # [1, 64, 24, 77]
            x3ds = []
        
            # TODO, Net_rgb의 raw 출력을 연속적으로 다르게 저장하고,
            # 각 저장된 출력에 대해서 반복하면서, 3D 출력을 계산해야한다.

            # print(f"x_rgb: {x_rgb.keys()}")
            # print(f"x_rgb: {x_rgb['1_1'].shape}")

            # print(f"hidden_states: {hidden_states.keys()}")
            # print(f"hidden_states: {hidden_states['1_1'].shape}")

            for i in range(batch_size):
                x3d = None

                for scale_2d in self.project_res: # 1, 2, 4, 8
                    scale_2d = int(scale_2d)
                    # print(scale_2d)
                
                    # Loader에서 projected_pix_이 1,2,4,8로 각각 저장되어 있어야 한다. 
                    # fov_mask도 마찬가지로 구해져 있어야 한다.
                    # shape of projected_pix = (129600, 2) size
                    projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                                    
                    fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()

                    # print(f"projected_pix: {projected_pix.shape}")
                    # print(f"fov_mask: {fov_mask.shape}")

                    # print(f"hidden_states: {hidden_states[k].shape}")

                    if x3d is None:
                        x3d = self.projects[str(scale_2d)](
                            # x_rgb["1_" + str(scale_2d)][seq][i],
                            hidden_states["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )
                    else:
                        x3d += self.projects[str(scale_2d)](
                            # x_rgb["1_" + str(scale_2d)][seq][i],
                            hidden_states["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )        
                    
                x3ds.append(x3d)

            # x3d size: [1, 200, 60, 36, 60] size:,  torch.Size([1, 64, 128, 128, 16])
            input_dict = {"x3d": torch.stack(x3ds),}

            if seq != int(sequence_length - 1):
                out[f"{seq+1}"] = self.net_3d_decoder(input_dict)
            else:
                out[f"last"] = self.net_3d_decoder(input_dict)

        return out

    def step(self, batch, step_type, metric):
        seq_len, batch_size, _, _, _ = batch["img"].shape
        target = batch["target"] # (sequence_length, batch_size, H, W, D)

        loss = 0
        out_dict = self(batch)

        # print(f"out_dict: {out_dict.keys()}")
        # for i in out_dict.keys():
        #     print(i)
        # sequence 개수만큼 나옴.

        for idx in range(seq_len):
            if idx != int(seq_len-1):
                ssc_pred = out_dict[f"{idx+1}"]["ssc_logit"]
                target_ = target[idx,]

                class_weight = self.class_weights.type_as(batch["img"])
                if self.CE_ssc_loss:
                    loss_ssc = CE_ssc_loss(ssc_pred, target_, class_weight)
                    loss += loss_ssc
                    self.log(
                        step_type + f"/loss_ssc(from_{seq_len})",
                        loss_ssc.detach(),
                        on_epoch=True,
                        sync_dist=True,
                    )

            elif idx == int(seq_len-1): #last one 
                ssc_pred = out_dict["last"]["ssc_logit"]
                target_ = target[-1,]

            #ssc_pred = out_dict["ssc_logit"] # (batch_size, sequence_length, n_classes, H, W, D)
        
                if self.context_prior:
                    P_logits = out_dict["last"]["P_logits"]
                    CP_mega_matrices = batch["CP_mega_matrices"][-1,]

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
                    loss_ssc = CE_ssc_loss(ssc_pred, target_, class_weight)
                    loss += loss_ssc
                    self.log(
                        step_type + "/loss_ssc",
                        loss_ssc.detach(),
                        on_epoch=True,
                        sync_dist=True,
                    )

                if self.sem_scal_loss:
                    loss_sem_scal = sem_scal_loss(ssc_pred, target_)
                    loss += loss_sem_scal
                    self.log(
                        step_type + "/loss_sem_scal",
                        loss_sem_scal.detach(),
                        on_epoch=True,
                        sync_dist=True,
                    )

                if self.geo_scal_loss:
                    loss_geo_scal = geo_scal_loss(ssc_pred, target_)
                    loss += loss_geo_scal
                    self.log(
                        step_type + "/loss_geo_scal",
                        loss_geo_scal.detach(),
                        on_epoch=True,
                        sync_dist=True,
                    )

                if self.fp_loss and step_type != "test":
                    # frustums_masks = torch.stack(batch["frustums_masks"][0]) #[-1,] 
                    frustums_masks = batch["frustums_masks"][0].float()
                    # frustums_class_dists = torch.stack(
                    #     batch["frustums_class_dists"][0] #[-1,]
                    # ).float()  # (bs, n_frustums, n_classes)

                    frustums_class_dists = batch["frustums_class_dists"][0].float()

                    n_frustums = frustums_class_dists.shape[1]

                    pred_prob = F.softmax(ssc_pred, dim=1)
                    batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

                    frustum_loss = 0
                    frustum_nonempty = 0
                    for frus in range(n_frustums):
                        frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                        prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                        prob = prob.reshape(batch_size, self.n_classes, -1).permute(1, 0, 2)
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

                y_true = target_.cpu().numpy()
                y_pred = ssc_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        for prefix, metric in metric_list:
            stats = metric.get_stats()
            self.log(f"{prefix}/mIoU", stats["iou_ssc_mean"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
        return [optimizer], [scheduler]


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x




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
        self.fp_loss = fp_loss
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
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
            
        # #self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)
        # get_model = build_model(get_config())  # 가상 예시
        
        #self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True)

        # # (2) GETMultiScaleBackbone으로 감싸, multi-scale feature를 dict로 받도록 함
        # self.net_rgb = MonoSceneGET2DNetwork(
        #     get_config=get_config,
        #     edsa_config=edsa_config,
        #     out_scales=[1, 2, 4, 8, 16],  # 필요 스케일
        #     base_feature_dim=feature     # MonoScene가 기대하는 채널 수 (ex: 200)
        # )
        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

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

        

        # CW added
        # 1220, 370 > 1232, 372
        #self.e2src = event_embed(shape=[1224, 384], batch_size=self.batch_size, group_num=group_num, patch_size=patch_size) # TODO image shape/ batch_size 등을 입력으로 받아와야 함.
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


    def forward(self, batch):
        blk_evt = batch["img"] # (B, t, p , h, w) 라고 가정해야 할 듯?
        # Out: [1, diverse N, 4] // [Batch, Event streams, 4(t_p_h_w)]
        batch_size = len(blk_evt)
        # Out: 1 이여야함.

        
        out = {}

        # x_rgb = self.net_rgb(img) # deprecated
        group_evt = blk_evt
        #group_evt = self.e2src(blk_evt) #TODO maybe the collate _fn make
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

        """
        Pose-processing 하는, 순서가 DPT랑 다르긴 한데, 일단 연결해보자.
        """
        img_x3 = self.post_processing1(x_last) # B, 7038, 96 > B, 7038, 92, 305
        out_x3 = self.refinenet3(img_x3) # interpolated into 184, 610
        
        img_x2 = self.post_processing2(x_out2) # B, 7038, 96 > B, 7038, 184, 610
        out_x2 = self.refinenet2(out_x3, img_x2) # interpolated into 368, 1220

        img_x1 = self.post_processing3(x_out1) # B, 7038, 96 > B, 7038, 368, 1220
        out_x1 = self.refinenet1(out_x2, img_x1) # interpolated into 370, 1220
        #print("out_x1!!: ", out_x1.shape)

        x_rgb = {}
        x_rgb['1_1'] = out_x1

        
        #x_rgb = self.net_rgb(group_evt) #TODO
        
        # x_rgb.keys() => ['1_1', '1_2', '1_4', '1_8', '1_16']
        # for iteration in range(x_rgb.keys()):
        #     x_rgb[iteration] = None # dpt_1 ~ dpt_4

        # -------------------------------
        # (B) 기존 Monoscene 방식과 동일
        # -------------------------------
        x3ds = []
        for i in range(batch_size):
            x3d = None
            for scale_2d in self.project_res:  # 예: [1,2,4,8]
                scale_2d = int(scale_2d)
                projected_pix = batch[f"projected_pix_{self.project_scale}"][i].cuda()
                fov_mask = batch[f"fov_mask_{self.project_scale}"][i].cuda()

                # x_rgb["1_{scale_2d}"]를 사용해 투영
                if x3d is None:
                    x3d = self.projects[str(scale_2d)](
                        x_rgb[f"1_{scale_2d}"][i],  # (C, H//scale_2d, W//scale_2d)
                        projected_pix // scale_2d,
                        fov_mask,
                    )
                else:
                    x3d += self.projects[str(scale_2d)](
                        x_rgb[f"1_{scale_2d}"][i],
                        projected_pix // scale_2d,
                        fov_mask,
                    )
            x3ds.append(x3d)

        # 3D decoder 등 기존 로직
        input_dict = {"x3d": torch.stack(x3ds)}
        out = self.net_3d_decoder(input_dict)
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]

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

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

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






