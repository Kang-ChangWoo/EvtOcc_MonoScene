import torch
import torch.nn as nn


class FLoSP(nn.Module):
    def __init__(self, scene_size, dataset, project_scale):
        super().__init__()
        self.scene_size = scene_size
        self.dataset = dataset
        self.project_scale = project_scale

    def forward(self, x2d, projected_pix, fov_mask):
        c, h, w = x2d.shape

        src = x2d.view(c, -1)
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)
        # => (channel, (h*w) + 1) size의 텐서

        # projected_pix = [129600, 2] size의 텐서
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices[~fov_mask] = h * w
        # fov 밖에 있는 애들은, 제일 마지막(x) 인덱스를 할당한다..?
        img_indices = img_indices.expand(c, -1).long()  # c, HWD
        
        # 소스 텐서에서 1 dim 을 기준하여, 해당하는 index를 불러서 합쳐라.
        src_feature = torch.gather(src, 1, img_indices)

        if self.dataset == "NYU":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[2] // self.project_scale,
                self.scene_size[1] // self.project_scale,
            )
            x3d = x3d.permute(0, 1, 3, 2)

        elif self.dataset == "kitti":
            x3d = src_feature.reshape(
                c,
                self.scene_size[0] // self.project_scale,
                self.scene_size[1] // self.project_scale,
                self.scene_size[2] // self.project_scale,
            )

        return x3d
