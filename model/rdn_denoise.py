import torch
import torch.nn as nn


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, activation, k_size=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, k_size, padding=(k_size-1)//2, stride=1),
            activation
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, activation, k_size=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G, activation, k_size=k_size))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN_denoise(nn.Module):
    def __init__(self, args):
        super(RDN_denoise, self).__init__()
        G0 = args.G0
        k_size = args.RDN_ksize
        SFENet_ksize = args.SFENet_ksize
        last_conv_ksize = args.last_conv_ksize
        self.D, C, G = args.D, args.C, args.G

        # Activation function
        if args.activation == 'ReLU':
            activation = nn.ReLU()
        elif args.activation == 'PReLU':
            activation = nn.PReLU()

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_channels, G0, SFENet_ksize, padding=(SFENet_ksize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, SFENet_ksize, padding=(SFENet_ksize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C, activation=activation, k_size=k_size)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, k_size, padding=(k_size-1)//2, stride=1)
        ])

        # Last conv layer
        self.last_conv = nn.Conv2d(G, args.n_channels, last_conv_ksize,
                                   padding=(last_conv_ksize-1)//2, stride=1)

    def forward(self, I_LQ):
        f__1 = self.SFENet1(I_LQ)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        x = self.last_conv(x)
        x = x + I_LQ
        return x
