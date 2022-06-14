import argparse


def parser():
    parser = argparse.ArgumentParser(description='Residual Dense Network for denoising')
    # --- Dataset ---
    parser.add_argument('--dataset', type=int,
                        help='set dataset number')

    # --- Save path ---
    parser.add_argument('--checkpoint', default='./checkpoint', help='datasave directory')

    # --- Training specifications ---
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--decay_type', default='RDN')
    parser.add_argument('--activation', default='PReLU', choices=('ReLU', 'PReLU'))
    parser.add_argument('--cudnn_benchmark', action='store_true')

    # --- Network specifications ---
    parser.add_argument('--D', type=int, default=20,
                        help='number of RDBs')
    parser.add_argument('--C', type=int, default=6,
                        help='conv layers in RDB')
    parser.add_argument('--G', type=int, default=32,
                        help='channels each conv layers in RDB')
    parser.add_argument('--G0', type=int, default=32,
                        help='output channels in RDB. (growth rate of dense net)')
    parser.add_argument('--RDN_ksize', type=int, default=3,
                        help='default kernel size. (Use in RDN)')
    parser.add_argument('--SFENet_ksize', type=int, default=3)
    parser.add_argument('--last_conv_ksize', type=int, default=3)

    # --- Loss ---
    parser.add_argument('--loss_type', default='L1',
                        help='loss function configuration')

    # --- Optimization ---
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--optimizer', default='Adam', choices=('SGD', 'Adam', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')

    return parser
