import wandb

from torch.nn import CrossEntropyLoss
from data_loader import hieve_data_loader, matres_data_loader, get_dataloaders
from loss import TransitivityLoss, CrossCategoryLoss
from model import RoBERTa_MLP, BiLSTM_MLP, Box_BiLSTM_MLP
from parser import *
from train import Trainer, Evaluator
from utils import *
from pathlib import Path


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_dataloader(args):
    data_type = args.data_type
    log_batch_size = int(args.log_batch_size)
    data_dir = Path(args.data_dir).expanduser()

    if data_type == "hieve":
        num_classes = 4
        hieve_train_set, hieve_valid_set, hieve_test_set = hieve_data_loader(args, data_dir)
        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = hieve_valid_set
        test_set_dict["hieve"] = hieve_test_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict = get_dataloaders(log_batch_size, hieve_train_set, valid_set_dict, test_set_dict)
    elif data_type == "matres":
        num_classes = 4
        matres_train_set, matres_valid_set, matres_test_set = matres_data_loader(args, data_dir)
        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["matres"] = matres_valid_set
        test_set_dict["matres"] = matres_test_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict = get_dataloaders(log_batch_size, matres_train_set, valid_set_dict, test_set_dict)
    elif data_type == "joint":
        num_classes = 8
        hieve_train_set, hieve_valid_set, hieve_test_set = hieve_data_loader(args, data_dir)
        matres_train_set, matres_valid_set, matres_test_set = matres_data_loader(args, data_dir)
        joint_train_set = hieve_train_set + matres_train_set
        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = hieve_valid_set
        valid_set_dict["matres"] = matres_valid_set
        test_set_dict["hieve"] = hieve_test_set
        test_set_dict["matres"] = matres_test_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict = get_dataloaders(log_batch_size, joint_train_set, valid_set_dict, test_set_dict)

    return train_dataloader, valid_dataloader_dict, test_dataloader_dict, num_classes


def create_model(args, num_classes):
    if args.model == "finetune":
        model = RoBERTa_MLP(
            num_classes=num_classes,
            data_type=args.data_type,
            mlp_size=args.mlp_size,
            hidden_size=args.roberta_hidden_size,
        )
    elif args.model == "bilstm":
        model = BiLSTM_MLP(
            num_classes=num_classes,
            data_type=args.data_type,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.num_layers,
            mlp_size=args.mlp_size,
            lstm_input_size=args.lstm_input_size,
            roberta_size_type="roberta-base",
        )
    elif args.model == "box":
        model = Box_BiLSTM_MLP(
            num_classes=num_classes,
            data_type=args.data_type,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.num_layers,
            mlp_size=args.mlp_size,
            lstm_input_size=args.lstm_input_size,
            volume_temp=args.volume_temp,
            intersection_temp=args.intersection_temp,
            roberta_size_type="roberta-base",
        )
    else:
        raise ValueError(f"{args.model} is unsupported!")
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
    args.data_type = args.data_type.lower()
    train_dataloader, valid_dataloader_dict, test_dataloader_dict, num_classes = create_dataloader(args)
    model = create_model(args, num_classes)
    model = model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True) # AMSGrad
    evaluator = Evaluator(
        train_type=args.data_type,
        model_type=args.model,
        model=model,
        device=device,
        valid_dataloader_dict=valid_dataloader_dict,
        test_dataloader_dict=test_dataloader_dict,
        hieve_threshold=args.hieve_threshold,
        matres_threshold=args.matres_threshold,
    )
    early_stopping = EarlyStopping("Accuracy", patience=args.patience)

    hier_weights, temp_weights = get_init_weights(device)
    loss_anno_dict = {}
    loss_anno_dict["hieve"] = CrossEntropyLoss(weight=hier_weights)
    loss_anno_dict["matres"] = CrossEntropyLoss(weight=temp_weights)
    loss_transitivity = TransitivityLoss()
    loss_cross_category = CrossCategoryLoss()

    trainer = Trainer(
        data_type=args.data_type,
        model_type=args.model,
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        opt=optimizer,
        loss_type=args.loss_type,
        loss_anno_dict=loss_anno_dict,
        loss_transitivity=loss_transitivity,
        loss_cross_category=loss_cross_category,
        lambda_dict=lambdas_to_dict(args),
        no_valid=args.no_valid,
        wandb_id=wandb.run.id,
        early_stopping=early_stopping,
        eval_step=args.eval_step,
    )

    return trainer


def main():
    args = build_parser()
    set_seed(args.seed)
    wandb.init()
    wandb.config.update(args)
    args = wandb.config
    set_logger(args.data_type, wandb.run.id)
    logging.info(args)
    trainer = setup(args)
    trainer.train()


if __name__ == '__main__':
    main()