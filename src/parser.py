import argparse
import random


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dataset directory path")
    parser.add_argument('--cuda', type=bool, default=True, help="enable/disable CUDA")
    parser.add_argument('--log_batch_size', type=int, default=4, help="batch size for training will be 2**log_batch_size")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs to train")
    parser.add_argument('--data_type', type=str, default="joint", help="dataset: [MATRES | HiEve | Joint]")
    parser.add_argument('--finetune', type=bool, default=True,
                        help="True: roberta-base emb with finetuning, no BiLSTM, False: roberta-base emb w/o finetuning + BiLSTM")
    parser.add_argument('--downsample', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lambda_annoT', type=float, default=0.1)
    parser.add_argument('--lambda_annoH', type=float, default=0.2)
    parser.add_argument('--lambda_transT', type=float, default=0.3)
    parser.add_argument('--lambda_transH', type=float, default=0.4)
    parser.add_argument('--lambda_cross', type=float, default=0.5)
    parser.add_argument('--MLP_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--roberta_hidden_size', type=int, default=1024)
    parser.add_argument('--lstm_input_size', type=int, default=768)
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")
    return parser.parse_args()
