import torch
import torch.nn as nn
import torch.nn.functional as F
from pub_mod import *
torch.set_printoptions(threshold=np.inf)
from cnsn import *
beta = 4

class Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.ad_net = nn.Sequential(
            nn.Conv2d(int(128 / beta), int(256 / beta), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(256 / beta)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(256 / beta), int(256 / beta), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(256 / beta)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(256 / beta), int(512 / beta), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(512 / beta)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, 4)
        )
        self.grl_layer = GRL(max_iter)
        self.fc = nn.Linear(int(512 / beta), 3)

    def forward(self, feature):
        adversarial_out = self.grl_layer(feature) # torch.Size([5, 256, 16, 16])
        adversarial_out = self.ad_net(adversarial_out).reshape(adversarial_out.shape[0], -1) # torch.Size([5, 512])
        adversarial_out = self.fc(adversarial_out) # torch.Size([5, 3])
        return adversarial_out

class SANN(nn.Module):
    def __init__(self, ada_num=2, max_iter=4000):
        super(SANN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(64 / beta)),
            nn.ReLU(inplace=True),    
        )
        
        self.cross_norms = [CrossNorm(), CrossNorm(), CrossNorm()]
        self.self_norms = [SelfNorm(int(64 / beta), is_two=True), SelfNorm(int(128 / beta), is_two=True), SelfNorm(int(128 / beta), is_two=True)]
        self.cnsn1 = CNSN(self.cross_norms[0], self.self_norms[0])
        self.cnsn2 = CNSN(self.cross_norms[1], self.self_norms[1])
        self.cnsn3 = CNSN(self.cross_norms[2], self.self_norms[2])

        self.Block1 = nn.Sequential(
            nn.Conv2d(int(64 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),   
            nn.Conv2d(int(128 / beta), int(196 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(196 / beta)),
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(196 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block2 = nn.Sequential(
            nn.Conv2d(int(128 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),   
            nn.Conv2d(int(128 / beta), int(196 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(196 / beta)),
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(196 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            nn.Conv2d(int(128 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(128 / beta), int(196 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(196 / beta)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(196 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(int(128 / beta), int(256 / beta), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(256 / beta)),
            nn.ReLU(inplace=True)
        )


        self.conv_final = nn.Sequential(
            nn.Conv2d(int(256 / beta), int(512 / beta), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(512 / beta))
        )

        # self.dis = Discriminator(max_iter)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(int(512 / beta), int(128 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(128 / beta)),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(int(128 / beta), int(64 / beta), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(64 / beta)),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(64 / beta), 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def _activate_cross_norm(self, i):
        if i < -1:
            return
        self.cross_norms[i].active = True
    def _deactivate_cross_norm(self, i):
        if i < -1:
            return
        self.cross_norms[i].active = False

    def encode(self, x1, mode):
        origin_idxs = torch.arange(x1.size(0))
        active_idx = -1
        if mode == 'train':
            active_idx = np.random.randint(0, 3)
            self._activate_cross_norm(active_idx)
        
        # feature generator
        self._activate_cross_norm(active_idx)
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

        self._deactivate_cross_norm(active_idx)
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
    
    def forward(self, input1, mode='train'):
        x_general, x_content, origin_idxs = self.encode(input1, mode) # torch.Size([5, 128, 32, 32]), torch.Size([5, 256, 16, 16])
        fea_x1_x1 = x_content
        fea_x1_x1 = self.conv_final(fea_x1_x1) # torch.Size([5, 512, 8, 8])
        cls_x1_x1 = self.decoder(fea_x1_x1) # torch.Size([5, 1, 32, 32])

        # dis_invariant = self.dis(x_general).reshape(x_general.shape[0], -1)
        # return cls_x1_x1[:, 0, :, :], fea_x1_x1, dis_invariant, origin_idxs

def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    M = 1024 * 1024
    size = total_num * 4 / M
    print('参数量: %d\n模型大小: %.4fM' % (total_num, size))

    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256).cuda()

    model = SANN().cuda()
    model_path = '/root/autodl-tmp/results/DDG/results/2023-03-01-11:44:29/O_C_I_to_M/model/SSAN_M_pO_C_I_to_M_best.pth'
    
    count_parameters(model)
    y = model(x)
