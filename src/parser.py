import argparse
import random


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dataset directory path")
    parser.add_argument('--no_cuda', default=False, action='store_true', help="enable/disable CUDA")
    parser.add_argument('--debug', default=False, action='store_true', help="whether to use debug data or not")
    parser.add_argument('--log_batch_size', type=int, default=4, help="batch size for training will be 2**log_batch_size")
    parser.add_argument('--epochs', type=int, default=80, help="number of epochs to train")
    parser.add_argument('--data_type', type=str, default="joint", help="dataset: [MATRES | HiEve | Joint]")
    # parser.add_argument('--finetune', type=bool, default=True,
    #                     help="True: roberta-base emb with finetuning, no BiLSTM, False: roberta-base emb w/o finetuning + BiLSTM")
    parser.add_argument('--downsample', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lambda_annoT', type=float, default=0.3)
    parser.add_argument('--lambda_annoH', type=float, default=0.2)
    parser.add_argument('--lambda_transT', type=float, default=0.2)
    parser.add_argument('--lambda_transH', type=float, default=0.2)
    parser.add_argument('--lambda_cross', type=float, default=0.2)
    parser.add_argument('--mlp_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--roberta_hidden_size', type=int, default=1024, help="roberta-base: 768, roberta-large: 1024")
    # parser.add_argument('--lstm_input_size', type=int, default=768)
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")
    parser.add_argument('--no_valid', default=False, action='store_true', help="turn off evaluation step")
    parser.add_argument('--loss_type', type=str, default="loss2",
                        help="loss1: within task-constraints, loss2: within task & cross constraints")

    return parser.parse_args()
