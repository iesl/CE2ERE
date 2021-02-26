import random
import wandb

from torch.nn import CrossEntropyLoss
from EventDataset import EventDataset
from data_loader import hieve_data_loader, matres_data_loader, get_dataloaders
from loss import TransitivityLoss, CrossCategoryLoss
from model import RoBERTa_MLP
from parser import *
from train import Trainer, Evaluator
from utils import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
    model = RoBERTa_MLP(
        num_classes=num_classes,
        data_type=args.data_type,
        mlp_size=args.mlp_size,
        hidden_size=args.roberta_hidden_size,
    )
    return model


def get_init_weights(device: torch.device):
    HierPC = 1802.0
    HierCP = 1846.0
    HierCo = 758.0
    HierNo = 63755.0
    HierTo = HierPC + HierCP + HierCo + HierNo  # total number of event pairs
    hier_weights = [0.25 * HierTo / HierPC, 0.25 * HierTo / HierCP, 0.25 * HierTo / HierCo, 0.25 * HierTo / HierNo]
    temp_weights = [0.25 * 818.0 / 412.0, 0.25 * 818.0 / 263.0, 0.25 * 818.0 / 30.0, 0.25 * 818.0 / 113.0]
    return torch.tensor(hier_weights, dtype=torch.float).to(device), torch.tensor(temp_weights, dtype=torch.float).to(device)


def setup(args):
    device = cuda_if_available(args.no_cuda)
    train_dataloader, valid_dataloader_dict, test_dataloader_dict, num_classes = create_dataloader(args, device)
    model = create_model(args, num_classes)
    model = model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True) # AMSGrad
    evaluator = Evaluator(
        model=model,
        device=device,
        valid_dataloader_dict=valid_dataloader_dict,
        test_dataloader_dict=test_dataloader_dict,
    )

    hier_weights, temp_weights = get_init_weights(device)
    loss_anno_dict = {}
    loss_anno_dict["hieve"] = CrossEntropyLoss(weight=hier_weights)
    loss_anno_dict["matres"] = CrossEntropyLoss(weight=temp_weights)
    loss_transitivity = TransitivityLoss()
    loss_cross_category = CrossCategoryLoss()

    trainer = Trainer(
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        opt=optimizer,
        loss_anno_dict=loss_anno_dict,
        loss_transitivity=loss_transitivity,
        loss_cross_category=loss_cross_category,
        lambda_dict=lambdas_to_dict(args),
        no_valid=args.no_valid,
        roberta_size_type="roberta-base",
    )

    return trainer


def main():
    args = build_parser()
    set_seed(args.seed)
    args.downsample = random.uniform(0.01, 0.2)
    args.lambda_annoH = random.uniform(0, 1.0)
    args.lambda_annoT = random.uniform(0, 1.0)
    args.lambda_transH = random.uniform(0, 1.0)
    args.lambda_transT = random.uniform(0, 1.0)
    args.lambda_cross = random.uniform(0, 1.0)
    print(args.no_valid)
    print(args)
    wandb.init()
    wandb.config.update(args)
    args = wandb.config
    trainer = setup(args)
    trainer.train()


if __name__ == '__main__':
    main()