import torch
import torch.nn as nn
import torch.nn.functional as F
from .pub_mod import *
torch.set_printoptions(threshold=np.inf)
from .cnsn import *

class Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.ad_net = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, 4)
        )
        self.grl_layer = GRL(max_iter)
        self.fc = nn.Linear(512, 4)

    def forward(self, feature):
        adversarial_out = self.grl_layer(feature) # torch.Size([5, 256, 16, 16])
        adversarial_out = self.ad_net(adversarial_out).reshape(adversarial_out.shape[0], -1) # torch.Size([5, 512])
        adversarial_out = self.fc(adversarial_out) # torch.Size([5, 3])
        return adversarial_out


class SSAN_M(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000, use_cn=False, use_sn=False):
        super(SSAN_M, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),    
        )
        self.use_cn = use_cn
        self.use_sn = use_sn
        self.cross_norms = [CrossNorm(active=False), CrossNorm(active=False), CrossNorm(active=False)]
        self.self_norms = [SelfNorm(64, is_two=True, active=self.use_sn), SelfNorm(128, is_two=True, active=self.use_sn), SelfNorm(128, is_two=True, active=self.use_sn)]
        self.cnsn1 = CNSN(self.cross_norms[0], self.self_norms[0])
        self.cnsn2 = CNSN(self.cross_norms[1], self.self_norms[1])
        self.cnsn3 = CNSN(self.cross_norms[2], self.self_norms[2])

        self.Block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),  
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),  
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

        self.dis = Discriminator(max_iter)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.cnt = 0
    def _activate_self_norm(self, i):
        if i < -1:
            return
        self.self_norms[i].active = True
    
    def _deactivate_self_norm(self, i):
        if i < -1:
            return
        self.self_norms[i].active = False

    def _activate_cross_norm(self, i):
        if i < -1:
            return
        self.cross_norms[i].active = True
    def _deactivate_cross_norm(self, i):
        if i < -1:
            return
        self.cross_norms[i].active = False
    
    def encode(self, x1, mode='train', epoch=0):
        # active_num = epoch // 300
        self.cnt += 1 ## 打日志的rate limiter
        active_num = 1
        origin_idxs = torch.arange(x1.size(0))
        active_idxs = np.random.choice(3, active_num, replace=True).tolist()
        if mode == 'train':
            for idx in active_idxs:
                if self.use_cn:
                    self._activate_cross_norm(idx)
        # for idx in range(3):
        #     if self.use_sn:
        #         self._activate_self_norm(idx)
        
        if self.cnt % 100 == 0:
            print('epoch', str(epoch), 'mode', mode, 'active_idxs', active_idxs, 'active_num', active_num, 'self_norm', self.self_norms[0].active, self.self_norms[1].active, self.self_norms[2].active, 'cross_norm', self.cross_norms[0].active, self.cross_norms[1].active, self.cross_norms[2].active)
        # feature generator
        x1 = self.conv1(x1) # x1 torch.Size([5, 64, 256, 256])
        x1, idxs = self.cnsn1(x1)
        x1_1 = self.Block1(x1) # torch.Size([5, 128, 128, 128])
        origin_idxs = origin_idxs[idxs]

        x1_1, idxs = self.cnsn2(x1_1)
        x1_2 = self.Block2(x1_1) # torch.Size([5, 128, 64, 64])
        origin_idxs = origin_idxs[idxs]

        x1_2, idxs = self.cnsn3(x1_2)
        x1_3 = self.Block3(x1_2) # torch.Size([5, 128, 32, 32])
        origin_idxs = origin_idxs[idxs]

        if mode == 'train':
            for idx in active_idxs:
                self._deactivate_cross_norm(idx)
                # self._deactivate_self_norm(idx)
        # content feature extractor
        x1_4 = self.layer4(x1_3)  #torch.Size([5, 256, 16, 16])

        domain_invariant = x1_3
        return domain_invariant, x1_4, origin_idxs
    
    def exact_feature_distribution_matching(self, content, style):
        assert (content.size() == style.size()) ## content and style features should share the same shape
        B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
        _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
        value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
        return transferred_content.view(B, C, W, H)
    
    def forward(self, input1, mode='train', epoch=0):
        x_general, x_content, origin_idxs = self.encode(input1, mode, epoch) # torch.Size([5, 128, 32, 32]), torch.Size([5, 256, 16, 16])
        fea_x1_x1 = x_content
        fea_x1_x1 = self.conv_final(fea_x1_x1) # torch.Size([5, 512, 8, 8])
        cls_x1_x1 = self.decoder(fea_x1_x1) # torch.Size([5, 1, 32, 32])

        dis_invariant = self.dis(x_general).reshape(x_general.shape[0], -1)
        # return torch.mean(cls_x1_x1[:, 0, :, :], dim=(1, 2)) ## for grad-cam
        return cls_x1_x1[:, 0, :, :], fea_x1_x1, dis_invariant, origin_idxs

def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    M = 1024 * 1024
    size = total_num * 4 / M
    print('参数量: %d\n模型大小: %.4fM' % (total_num, size))

    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256).cuda()

    model = SSAN_M().cuda()
    model_path = '/root/autodl-tmp/results/DDG/results/2023-03-01-11:44:29/O_C_I_to_M/model/SSAN_M_pO_C_I_to_M_best.pth'
    
    count_parameters(model)
    y = model(x)
