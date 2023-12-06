python

class SegMaskPSP(nn.Module):
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):
        super(SegMaskPSP, self).__init__()
        self.c_in8 = ch[0]
        self.c_in16 = ch[1]
        self.c_in32 = ch[2]
        self.c_out = n_segcls
        self.out = nn.Sequential(
            RFB2(c_hid*3, c_hid, d=[2,3], map_reduce=6),
            PyramidPooling(c_hid, k=[1, 2, 3, 6]),
            FFM(c_hid*2, c_hid, k=3, is_cat=False),
            nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.m8 = nn.Sequential(
            Conv(self.c_in8, c_hid, k=1),
        )
        self.m32 = nn.Sequential(
            Conv(self.c_in32, c_hid, k=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.m16 = nn.Sequential(
            Conv(self.c_in16, c_hid, k=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        
    def forward(self, x):
        feat = torch.cat([self.m8(x[0]), self.m16(x[1]), self.m32(x[2])], 1)
        return self.out(feat)
