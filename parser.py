import argparse
import random


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size for training ")
    parser.add_argument('--rst_file_name', type=str, default="")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--dataset', type=str, default="HiEve", help="HiEve | MATRES | Joint")
    parser.add_argument('--add_loss', type=int, default=0)
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--MAX_EVALS', type=int, default=50)
    parser.add_argument('--debugging', type=int, default=1)

    # parser.add_argument('--downsample', type=float, default=1)
    # parser.add_argument('--learning_rate', type=float, default=0.01)
    # parser.add_argument('--lambda_annoT', type=float, default=1)
    # parser.add_argument('--lambda_annoH', type=float, default=1)
    # parser.add_argument('--lambda_transT', type=float, default=0)
    # parser.add_argument('--lambda_transH', type=float, default=0)
    # parser.add_argument('--lambda_cross', type=float, default=0)
    # parser.add_argument('--MLP_size', type=int, default=512)
    # parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--lstm_hidden_size', type=int, default=256)
    # parser.add_argument('--roberta_hidden_size', type=int, default=1024)
    # parser.add_argument('--lstm_input_size', type=int, default=768)
    return parser.parse_args()
