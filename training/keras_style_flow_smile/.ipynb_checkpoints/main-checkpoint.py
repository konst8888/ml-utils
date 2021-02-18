import argparse
import os
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--gen_path", default='',
                        help="Load existing gen model path")
    parser.add_argument("--disc_path", default='',
                        help="Load existing disc model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start epoch")
    parser.add_argument("--train_size", default=0.8, type=float, help="Train size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--frn", action='store_true',
                        help="Use Filter Response Normalization and TLU")
    parser.add_argument("--use_skip", action='store_true',
                        help="Use skip connections")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")

    cfg = parser.parse_args()

    train(cfg)


if __name__ == '__main__':
    main()
