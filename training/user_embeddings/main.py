import argparse
import os
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./data",
                        help="Path to data csv file")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--encoder_path", default='',
                        help="Load existing encoder model path")
    parser.add_argument("--decoder_path", default='',
                        help="Load existing decoder model path")
    parser.add_argument("--hidden_size", default=100, type=int, help="Hidden size")
    parser.add_argument("--max_length", default=None, type=int, help="Max sequence len")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start epoch")
    parser.add_argument("--train_size", default=0.8, type=float, help="Train size")
    parser.add_argument("--teacher_forcing_ratio", default=0.5, type=float, help="Teacher forcing")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=None,
                        help="Lr decrease factor")

    cfg = parser.parse_args()

    train(cfg)


if __name__ == '__main__':
    main()
