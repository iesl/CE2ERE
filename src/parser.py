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
    parser.add_argument('--finetune', default=False, action='store_true',
                        help="True: roberta-base emb with finetuning, no BiLSTM, False: roberta-base emb w/o finetuning + BiLSTM")

    parser.add_argument('--model', type=str, default="bilstm", help="[finetune | bilstm | box]; finetune: roberta-base emb with finetuning,"
                                                          "bilstm: roberta-base emb w/o finetuning + BiLSTM,"
                                                          "box: roberta-base emb w/o finetuning + BiLSTM + Box")
    parser.add_argument('--downsample', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lambda_anno', type=float, default=1)
    parser.add_argument('--lambda_trans', type=float, default=0)
    parser.add_argument('--lambda_cross', type=float, default=0)

    parser.add_argument('--volume_temp', type=float, default=1)
    parser.add_argument('--intersection_temp', type=float, default=0.0001)

    parser.add_argument('--hieve_threshold', type=float, default=-0.602059991,
                        help="log0.5: -0.301029996, log0.25: -0.602059991, log0.1: -1") # log 0.5
    parser.add_argument('--matres_threshold', type=float, default=-0.602059991,
                        help="log0.5: -0.301029996, log0.25: -0.602059991, log0.1: -1")  # log 0.5

    parser.add_argument('--mlp_size', type=int, default=256)
    parser.add_argument('--double_output_dim', type=int, default=50) # will be doubled to get box embedding
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--roberta_hidden_size', type=int, default=1024, help="roberta-base: 768, roberta-large: 1024")

    parser.add_argument('--lstm_hidden_size', type=int, default=256, help="BiLSTM layer hidden dimension")
    parser.add_argument('--lstm_input_size', type=int, default=768, help="BiLSTM layer input dimension")

    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")
    parser.add_argument('--no_valid', default=False, action='store_true', help="turn off evaluation step")
    parser.add_argument('--loss_type', type=int, default=0,
                        help="1: within task-constraints, 2: within task & cross constraints")
    parser.add_argument('--patience', type=int, default=5, help="patience for early stopping")
    parser.add_argument('--eval_step', type=int, default=1, help="evaluation every n epochs")

    return parser.parse_args()
