import argparse
import random


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="dataset directory path")
    parser.add_argument('--no_cuda', default=False, action='store_true', help="enable/disable CUDA")
    parser.add_argument('--debug', default=False, action='store_true', help="whether to use debug data or not")
    parser.add_argument('--log_batch_size', type=int, default=4, help="batch size for training will be 2**log_batch_size")
    parser.add_argument('--epochs', type=int, default=80, help="number of epochs to train")
    parser.add_argument('--data_type', type=str, default="joint", choices=["matres", "hieve", "joint"], help="dataset: [MATRES | HiEve | Joint]")
    parser.add_argument('--finetune', default=False, action='store_true',
                        help="True: roberta-base emb with finetuning, no BiLSTM, False: roberta-base emb w/o finetuning + BiLSTM")

    parser.add_argument('--model', type=str, default="bilstm", choices=["bilstm", "box", "vector"],
                        help="[finetune | bilstm | box]; finetune: roberta-base emb with finetuning, bilstm: roberta-base emb w/o finetuning + BiLSTM, box: roberta-base emb w/o finetuning + BiLSTM + Box")

    parser.add_argument('--downsample', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-7)
    parser.add_argument('--lambda_anno', type=float, default=1)
    parser.add_argument('--lambda_trans', type=float, default=0)
    parser.add_argument('--lambda_cross', type=float, default=0)
    parser.add_argument('--lambda_pair', type=float, default=0)
    parser.add_argument('--lambda_condi', type=float, default=1)

    parser.add_argument('--volume_temp', type=float, default=1)
    parser.add_argument('--intersection_temp', type=float, default=0.0001)

    parser.add_argument('--hieve_threshold', type=float, default=-0.301029996, help="log0.5: -0.301029996, log0.25: -0.602059991, log0.1: -1")
    parser.add_argument('--matres_threshold', type=float, default=-0.301029996)

    parser.add_argument('--threshold1', type=float, default=-0.602059991)
    parser.add_argument('--threshold2', type=float, default=-0.301029996)
    parser.add_argument('--threshold3', type=float, default=-0.602059991)
    parser.add_argument('--threshold4', type=float, default=-0.301029996)

    parser.add_argument('--mlp_size', type=int, default=512) # mlp hidden dim
    parser.add_argument('--mlp_output_dim', type=int, default=32) # mlp output dim;
    parser.add_argument('--hieve_mlp_size', type=int, default=64)
    parser.add_argument('--matres_mlp_size', type=int, default=64)
    parser.add_argument('--proj_output_dim', type=int, default=32)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--roberta_hidden_size', type=int, default=1024, help="roberta-base: 768, roberta-large: 1024")

    parser.add_argument('--lstm_hidden_size', type=int, default=256, help="BiLSTM layer hidden dimension")
    parser.add_argument('--lstm_input_size', type=int, default=768, help="BiLSTM layer input dimension")

    parser.add_argument('--no_valid', default=False, action='store_true', help="turn off evaluation step")
    parser.add_argument('--loss_type', type=int, default=0,
                        help="1: within task-constraints, 2: within task & cross constraints")
    parser.add_argument('--patience', type=int, default=10, help="patience for early stopping")
    parser.add_argument('--eval_step', type=int, default=1, help="evaluation every n epochs")
    parser.add_argument('--eval_type', type=str, default="one", choices=["one", "two", "four"], help="evaluate wheter using one threshold or two threshold")
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32), help="seed for random number generator")

    parser.add_argument('--load_model', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--saved_model', type=str, default="", help="saved model path")
    parser.add_argument('--wandb_id', type=str, default="", help="wandb run path")
    parser.add_argument('--save_plot', type=int, default=1, help="0: false, 1: true")

    parser.add_argument('--symm_train', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--symm_eval', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--cv_valid', type=int, default=0, help="0: false, 1: true")
    parser.add_argument('--model_save', type=int, default=1, help="0: false, 1: true")

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help="max_grad_norm for gradient clipping ex) 1,5,10")
    return parser.parse_args()
