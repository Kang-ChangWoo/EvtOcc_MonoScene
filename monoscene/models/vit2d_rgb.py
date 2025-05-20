"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import timm



class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = features // 32

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(
                self.feature_1_1, self.out_feature_1_1, kernel_size=1
            )
            self.resize_output_1_2 = nn.Conv2d(
                self.feature_1_2, self.out_feature_1_2, kernel_size=1
            )
            self.resize_output_1_4 = nn.Conv2d(
                self.feature_1_4, self.out_feature_1_4, kernel_size=1
            )
            self.resize_output_1_8 = nn.Conv2d(
                self.feature_1_8, self.out_feature_1_8, kernel_size=1
            )
            self.resize_output_1_16 = nn.Conv2d(
                self.feature_1_16, self.out_feature_1_16, kernel_size=1
            )

            self.up16 = UpSampleBN(
                skip_input=features + 176, output_features=self.feature_1_16 # 224
            )
            self.up8 = UpSampleBN(
                skip_input=self.feature_1_16 + 64, output_features=self.feature_1_8 # 80
            )
            self.up4 = UpSampleBN(
                skip_input=self.feature_1_8 + 40, output_features=self.feature_1_4 # 48
            )
            self.up2 = UpSampleBN(
                skip_input=self.feature_1_4 + 24, output_features=self.feature_1_2 #32
            )
            self.up1 = UpSampleBN(
                skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1 #3
            )
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        bs = x_block0.shape[0]
        x_d0 = self.conv2(x_block4)

        if self.use_decoder:
            x_1_16 = self.up16(x_d0, x_block3)
            x_1_8 = self.up8(x_1_16, x_block2)
            x_1_4 = self.up4(x_1_8, x_block1)
            x_1_2 = self.up2(x_1_4, x_block0)
            x_1_1 = self.up1(x_1_2, features[0])
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "1_8": self.resize_output_1_8(x_1_8),
                "1_16": self.resize_output_1_16(x_1_16),
            }
        else:
            x_1_1 = features[0]
            x_1_2, x_1_4, x_1_8, x_1_16 = (
                features[4],
                features[5],
                features[6],
                features[8],
            )
            x_global = features[-1].reshape(bs, 2560, -1).mean(2)
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "global": x_global,
            }


# 아마 12개가 출력될 듯?
class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        feats = [x]                        # blocks 출력만 쌓임

        for name, module in self.original_model._modules.items():

            if name == "blocks":          # ── Stage containing MBConv/Res blocks
                for blk in module:        #   각 블록 순회
                    x = blk(x)
                    feats.append(x)       #   ★ feature 저장 ★

            else:                         # stem, bn1, act1, head 등
                # 일부 컨테이너(Module) 는 forward 가 없으므로
                if isinstance(module, nn.Module) and module.forward is nn.Module.forward:
                    # forward 없는 껍데기 → 내부 child 실행
                    for sub in module.children():
                        x = sub(x)
                else:
                    x = module(x)

        return feats


import torchvision.transforms as T

class ViT2D_RGB(nn.Module):
    def __init__(self, backend, transform, use_decoder=False): # num_features, out_feature, 
        super(ViT2D_RGB, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        # self.decoder = DecoderBN(
        #     out_feature=out_feature,
        #     use_decoder=use_decoder,
        #     bottleneck_features=num_features,
        #     num_features=num_features,
        # )

        self.transform = transform
        self.transform = T.Compose([
                        T.Resize((384, 480), interpolation=Image.BICUBIC),
                        T.ToTensor(),                                # 0-1 float, CHW
                        T.Normalize(mean=[0.485,0.456,0.406],
                                    std =[0.229,0.224,0.225])
                    ])

        

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        imgs = []
        print("x.shape", x.shape)

        
        for img_t in x:                                   # ----- loop over batch
            # 1) Tensor(C,H,W) → NumPy(H,W,C)  (RGB, float 0-1)
            print("img_t.shape", img_t.shape)
            img_pil = T.ToPILImage()(img_t)           # Tensor → PIL (RGB)
            #img_np = np.array(img_pil)[:, :, ::-1]
            img_tf = self.transform(img_pil)       # CHW torch #["image"]
            print("img_tf.shape", img_tf.shape)
            imgs.append(img_tf)  
            # img_np = img_t.permute(1, 2, 0).cpu().numpy()

            # 2) 0-1 범위를 0-255 uint8 로 변환  (채널 순서는 그대로 RGB)
            # img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            # 3) NumPy → PIL  (★ RGB 모드로 생성 ★)
            # img_pil = Image.fromarray(img_np, mode="RGB")
            #img_pil = Image.fromarray(img_np[..., ::-1], mode="RGB")
            # 만약 Pillow 버전이 낮으면:
            # img_pil = Image.fromarray(img_np[..., ::-1], mode="RGB")
            # ★ dict 없이 넘긴다 ★
            # img_t_prep = self.transform(img_pil)["image"]      # CHW torch
            # imgs.append(img_t_prep)

        x_prep = torch.cat(imgs, dim=0).to(x.device)      # (B, 3, 256/384, 256/384)
        print("shape!!", x_prep.shape)
    
        #x_prep = self.transform(x)
        print("x_pre!!!!!", x_prep.shape)
        encoded_feats = self.encoder(x_prep)

        #x = self.transform(x)
        #encoded_feats = self.encoder(x)
        # unet_out = self.decoder(encoded_feats, **kwargs)
        # self.interpolation = interpolate_prediction()
        return encoded_feats

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    # It can process batch
    # @staticmethod
    # def interpolate_prediction(prediction): # self, 
    #     prediction = torch.nn.functional.interpolate(
    #                                                 prediction.unsqueeze(1),
    #                                                 size=img.shape[:2],
    #                                                 mode="bicubic",
    #                                                 align_corners=False,).squeeze()
    #     return prediction ## TODO. image size are needed

    """
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    """  
    @classmethod
    def build(cls, 
        basemodel_name: str = 'DPT_Hybrid',
        use_decoder: bool = False,
        pretrained: bool = True, **kwargs):

        #num_features = 2048

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load("intel-isl/MiDaS", basemodel_name, pretrained=pretrained)
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if basemodel_name == "DPT_Large" or basemodel_name == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, transform, **kwargs) #TODO
        print("Done.")
        return m

if __name__ == '__main__':
    model = ViT2D_RGB.build('intel-isl/MiDaS', basemodel_name='DPT_Hybrid', use_decoder=False, pretrained=True) # out_feature=256,
