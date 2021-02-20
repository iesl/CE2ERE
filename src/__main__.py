from data_loader import hieve_data_loader
from parser import *
from utils import *
from pathlib import Path


def create_dataloader(args):
    data_type = args.data_type
    log_batch_size = args.log_batch_size
    data_dir = Path(args.data_dir).expanduser()

    hieve_data_loader(args, data_dir)


    # print(hieve_files)


def setup(args):
    create_dataloader(args)



def main():
    args = build_parser()
    print("args:", args)
    setup(args)


if __name__ == '__main__':
    main()