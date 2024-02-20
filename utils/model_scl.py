import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class Pooling(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        return x
        

class CModel(nn.Module):
    def __init__(self, out_dim=128, normalize=True):
        super().__init__()
        # self.model = models.resnet50(pretrained=True)
#        self.model = models.resnet50(pretrained=True)
#        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.pre_process = nn.Sequential(*list(self.model.children())[:4])

        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.in_dim = 2048 + 512
        self.out_dim = out_dim
        self.normalize = normalize

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1792, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.out_dim),
            # nn.BatchNorm1d(self.out_dim)
        )

        self.pool1 = Pooling(1024, 512)
        self.pool2 = Pooling(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
    
    def forward(self, dist, ref):
        h1 = self.pre_process(dist)
        h2 = self.pre_process(ref)

        fea1_1 = self.layer1(h1)
        fea1_2 = self.layer2(fea1_1)
        fea1_3 = self.layer3(fea1_2)
        fea1_4 = self.layer4(fea1_3)

        fea2_1 = self.layer1(h2)
        fea2_2 = self.layer2(fea2_1)
        fea2_3 = self.layer3(fea2_2)
        fea2_4 = self.layer4(fea2_3)

        fea1_1 = self.avgpool(fea1_1)
        fea1_2 = self.avgpool(fea1_2)
        fea1_3 = self.pool1(fea1_3)
        fea1_4 = self.pool2(fea1_4)

        fea2_1 = self.avgpool(fea2_1)
        fea2_2 = self.avgpool(fea2_2)
        fea2_3 = self.pool1(fea2_3)
        fea2_4 = self.pool2(fea2_4)

        fea1 = torch.cat([fea1_1, fea1_2, fea1_3, fea1_4], dim=1)
        fea2 = torch.cat([fea2_1, fea2_2, fea2_3, fea2_4], dim=1)

        if self.normalize:
            h1 = nn.functional.normalize(fea1, dim=1)
            h2 = nn.functional.normalize(fea2, dim=1)
        
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return z1, z2, fea1


if __name__ == '__main__':
    inp1 = torch.randn(5, 3, 224, 224)
    inp2 = torch.randn(5, 3, 224, 224)

    model = CModel()
    z1, z2, fea = model(inp1, inp2)

    print(z1.shape)
    print(z2.shape)

    