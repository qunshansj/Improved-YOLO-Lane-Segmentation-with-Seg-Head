python

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()
        # Simplified MobileNetV3
        self.conv1 = DepthwiseSeparableConv(3, 16, 3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(16, 32, 3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(32, 64, 3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv(x)
        return x

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(64, 256, 3, stride=1, padding=1, dilation=1)
        self.atrous_block2 = nn.Conv2d(64, 256, 3, stride=1, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(64, 256, 3, stride=1, padding=4, dilation=4)
        self.atrous_block4 = nn.Conv2d(64, 256, 3, stride=1, padding=8, dilation=8)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(64, 256, 1, stride=1, bias=False))
        self.conv1x1 = nn.Conv2d(1280, 256, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([atrous_block1, atrous_block2, atrous_block3, atrous_block4, global_avg_pool], dim=1)
        x = self.conv1x1(x)
        return x

class YOLOSeg(nn.Module):
    def __init__(self):
        super(YOLOSeg, self).__init__()
        self.feature_extractor = MobileNetV3()
        self.panet = PANet()
        self.aspp = ASPP()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.panet(x)
        x = self.aspp(x)
        return x

    def compute_loss(self, predictions, targets):
        loss_yolo = F.mse_loss(predictions, targets)  # Simplified YOLO loss
        loss_seg = F.cross_entropy(predictions, targets)  # Simplified segmentation loss
        total_loss = loss_yolo + loss_seg
        return total_loss
