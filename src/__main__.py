from EventDataset import EventDataset
from data_loader import hieve_data_loader, matres_data_loader, get_dataloaders
from model import RoBERTa_MLP
from parser import *
from train import Trainer
from utils import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def create_dataloader(args, device):
    data_type = args.data_type
    log_batch_size = int(args.log_batch_size)
    data_dir = Path(args.data_dir).expanduser()

    hieve_train_set, hieve_valid_set, hieve_test_set = hieve_data_loader(args, data_dir, device)
    matres_train_set, matres_valid_set, matres_test_set = matres_data_loader(args, data_dir, device)

    if data_type.lower() == "joint":
        num_classes = 8
        joint_train_set = hieve_train_set + matres_train_set
        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = hieve_valid_set
        valid_set_dict["matres"] = matres_valid_set
        test_set_dict["hieve"] = hieve_test_set
        test_set_dict["matres"] = matres_test_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict = get_dataloaders(log_batch_size, joint_train_set, valid_set_dict, test_set_dict)

    return train_dataloader, valid_dataloader_dict, test_dataloader_dict, num_classes


def create_model(args, num_classes):
    lambda_dict = lambdas_to_dict(args)
    model = RoBERTa_MLP(
        num_classes=num_classes,
        data_type=args.data_type,
        lambda_dict=lambda_dict,
        mlp_size=args.mlp_size,
        hidden_size=args.roberta_hidden_size,
    )
    return model


def setup(args):
    device = cuda_if_available(args.use_cuda)
    train_dataloader, valid_dataloader_dict, test_dataloader_dict, num_classes = create_dataloader(args, device)
    model = create_model(args, num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True) # AMSGrad

    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_dataloader=train_dataloader,
        valid_dataloader_dict=valid_dataloader_dict,
        test_dataloader_dict=test_dataloader_dict,
        opt=optimizer,
        roberta_size_type="roberta-base"
    )

    return trainer

def main():
    args = build_parser()
    print("args:", args)
    trainer = setup(args)
    trainer.train()




if __name__ == '__main__':
    main()