import pytorch_lightning as pl
import torch
import torch.nn as nn
from monoscene.models.unet3d_nyu import UNet3D as UNet3DNYU
from monoscene.models.unet3d_kitti import UNet3D as UNet3DKitti
from monoscene.loss.sscMetrics import SSCMetrics
from monoscene.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
from monoscene.models.flosp import FLoSP
from monoscene.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np
import torch.nn.functional as F
from monoscene.models.unet2d import UNet2D
from torch.optim.lr_scheduler import MultiStepLR


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

        # log hyperparameters
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


import torch
import torch.nn as nn

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
        self.residual = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,  # 1x1 Conv
            padding=0,      # 차원 유지
            bias=self.bias
        )

    def forward(self, x, h_prev, c_prev):
        batch_size, _, H, W = x.shape

        if h_prev is None:
            h_prev = torch.zeros((batch_size, self.hidden_dim, H, W), device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros((batch_size, self.hidden_dim, H, W), device=x.device, dtype=x.dtype)

        # **1x1 Residual Connection**
        x_res = self.residual(x)  # (batch, hidden_dim, H, W)

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
        h_next = h_next + x_res  # Skip Connection 추가

        return h_next, c_next

# 기존의 forward 메서드를 ConvLSTMCell을 사용하도록 수정합니다.
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True, num_layers=1):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, bias)
            for i in range(num_layers)
        ])

    def forward(self, x, h_prev=None, c_prev=None):
        h_next, c_next = [], []

        for i, layer in enumerate(self.layers):
            h_i = h_prev[i] if h_prev is not None else None
            c_i = c_prev[i] if c_prev is not None else None
            h_i, c_i = layer(x, h_i, c_i)
            x = h_i  # 다음 레이어로 전달
            h_next.append(h_i)
            c_next.append(c_i)

        return h_next[0], c_next[0] # multi-layer deprecated.


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
        self.ConvLSTMs = nn.ModuleDict({
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

    def forward(self, batch, prev_states=None):
        img = batch["img"] # 3, 1, 3, 370, 1220

        batch_size = img.shape[1]
        #batch_size = len(img) deprecated

        out = {}
        #prev_states = {i: None for i in resolution_keys} if prev_states is None else prev_states
        cell_states = {}
        hidden_states = {}

        sequence_length = 3

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
                prev_states = cell_states[k] if prev_states is not None else None

                hidden_states[k], cell_states[k] = self.ConvLSTMs[k](features, prev_states)
            
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
