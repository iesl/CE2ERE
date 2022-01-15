import os

import wandb

from torch.nn import CrossEntropyLoss
from data_loader import hieve_data_loader, matres_data_loader, esl_data_loader, get_dataloaders, get_tag2index, add_pos_tag_embedding
from loss import TransitivityLoss, CrossCategoryLoss
from model import RoBERTa_MLP, BiLSTM_MLP, Box_BiLSTM_MLP, Vector_BiLSTM_MLP, Box_RoBERTa_MLP
from parser import *
from train import Trainer, ThresholdEvaluator, VectorBiLSTMEvaluator
from utils import *
from pathlib import Path
logger = logging.getLogger()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(args):
    data_type = args.data_type
    log_batch_size = int(args.log_batch_size)
    data_dir = Path(args.data_dir).expanduser()

    if data_type == "hieve":
        num_classes = 4
        hieve_train_set, hieve_valid_set, hieve_test_set, hieve_valid_cv_set, hieve_test_cv_set = hieve_data_loader(args, data_dir)

        tag2index = get_tag2index(hieve_train_set)
        hieve_train_set, hieve_valid_set, hieve_test_set = add_pos_tag_embedding(hieve_train_set, hieve_valid_set, hieve_test_set, tag2index)
        _, hieve_valid_cv_set, hieve_test_cv_set = add_pos_tag_embedding(None, hieve_valid_cv_set, hieve_test_cv_set, tag2index)

        if args.use_pos_tag:
            n_tags = len(tag2index)
        else:
            n_tags = 0

        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = hieve_valid_set
        test_set_dict["hieve"] = hieve_test_set

        valid_cv_set_dict, test_cv_set_dict = {}, {}
        valid_cv_set_dict["hieve"] = hieve_valid_cv_set
        test_cv_set_dict["hieve"] = hieve_test_cv_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict \
            = get_dataloaders(log_batch_size, hieve_train_set, valid_set_dict, test_set_dict, valid_cv_set_dict, test_cv_set_dict)
    elif data_type == "matres":
        num_classes = 4
        matres_train_set, matres_valid_set, matres_test_set, matres_valid_cv_set, matres_test_cv_set = matres_data_loader(args, data_dir)

        tag2index = get_tag2index(matres_train_set)
        matres_train_set, matres_valid_set, matres_test_set = add_pos_tag_embedding(matres_train_set, matres_valid_set, matres_test_set, tag2index)
        _, matres_valid_cv_set, matres_test_cv_set = add_pos_tag_embedding(None, matres_valid_cv_set, matres_test_cv_set, tag2index)

        if args.use_pos_tag:
            n_tags = len(tag2index)
        else:
            n_tags = 0

        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["matres"] = matres_valid_set
        test_set_dict["matres"] = matres_test_set

        valid_cv_set_dict, test_cv_set_dict = {}, {}
        valid_cv_set_dict["matres"] = matres_valid_cv_set
        test_cv_set_dict["matres"] = matres_test_cv_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict \
            = get_dataloaders(log_batch_size, matres_train_set, valid_set_dict, test_set_dict, valid_cv_set_dict, test_cv_set_dict)
    elif data_type == "esl":
        num_classes = 4
        esl_train_set, esl_valid_set, esl_test_set, esl_valid_cv_set, esl_test_cv_set = esl_data_loader(args, data_dir)

        tag2index = get_tag2index(esl_train_set)
        esl_train_set, esl_valid_set, esl_test_set = add_pos_tag_embedding(esl_train_set, esl_valid_set, esl_test_set, tag2index)
        _, esl_valid_cv_set, esl_test_cv_set = add_pos_tag_embedding(None, esl_valid_cv_set, esl_test_cv_set, tag2index)

        if args.use_pos_tag:
            n_tags = len(tag2index)
        else:
            n_tags = 0

        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = esl_valid_set
        test_set_dict["hieve"] = esl_test_set

        valid_cv_set_dict, test_cv_set_dict = {}, {}
        valid_cv_set_dict["hieve"] = esl_valid_cv_set
        test_cv_set_dict["hieve"] = esl_test_cv_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict \
            = get_dataloaders(log_batch_size, esl_train_set, valid_set_dict, test_set_dict, valid_cv_set_dict, test_cv_set_dict)
    elif data_type == "joint":
        num_classes = 8
        hieve_train_set, hieve_valid_set, hieve_test_set, hieve_valid_cv_set, hieve_test_cv_set = hieve_data_loader(args, data_dir)
        matres_train_set, matres_valid_set, matres_test_set, matres_valid_cv_set, matres_test_cv_set = matres_data_loader(args, data_dir)

        tag2index = get_tag2index(hieve_train_set + matres_train_set)
        hieve_train_set, hieve_valid_set, hieve_test_set = add_pos_tag_embedding(hieve_train_set, hieve_valid_set, hieve_test_set, tag2index)
        _, hieve_valid_cv_set, hieve_test_cv_set = add_pos_tag_embedding(None, hieve_valid_cv_set, hieve_test_cv_set, tag2index)
        matres_train_set, matres_valid_set, matres_test_set = add_pos_tag_embedding(matres_train_set, matres_valid_set, matres_test_set, tag2index)
        _, matres_valid_cv_set, matres_test_cv_set = add_pos_tag_embedding(None, matres_valid_cv_set, matres_test_cv_set, tag2index)

        if args.use_pos_tag:
            n_tags = len(tag2index)
        else:
            n_tags = 0
        joint_train_set = hieve_train_set + matres_train_set
        valid_set_dict, test_set_dict = {}, {}
        valid_set_dict["hieve"] = hieve_valid_set
        valid_set_dict["matres"] = matres_valid_set
        test_set_dict["hieve"] = hieve_test_set
        test_set_dict["matres"] = matres_test_set

        valid_cv_set_dict, test_cv_set_dict = {}, {}
        valid_cv_set_dict["hieve"] = hieve_valid_cv_set
        test_cv_set_dict["hieve"] = hieve_test_cv_set
        valid_cv_set_dict["matres"] = matres_valid_cv_set
        test_cv_set_dict["matres"] = matres_test_cv_set
        train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict \
            = get_dataloaders(log_batch_size, joint_train_set, valid_set_dict, test_set_dict, valid_cv_set_dict, test_cv_set_dict)
    print("# of tags:", n_tags)
    return train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict, num_classes, n_tags

def create_model(args, num_classes, n_tags):
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
            roberta_size_type=args.roberta_type,
        )
    elif args.model == "vector":
        model = Vector_BiLSTM_MLP(
            num_classes=num_classes,
            data_type=args.data_type,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.num_layers,
            mlp_size=args.mlp_size,
            lstm_input_size=args.lstm_input_size,
            mlp_output_dim=args.mlp_output_dim,
            hieve_mlp_size=args.hieve_mlp_size,
            matres_mlp_size=args.matres_mlp_size,
            roberta_size_type=args.roberta_type,
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
            mlp_output_dim=args.mlp_output_dim,
            proj_output_dim=args.proj_output_dim,
            loss_type=args.loss_type,
            roberta_size_type=args.roberta_type,
            n_tags=n_tags,
        )
    elif args.model == "box-finetune":
        model = Box_RoBERTa_MLP(
            num_classes=num_classes,
            data_type=args.data_type,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.num_layers,
            mlp_size=args.mlp_size,
            lstm_input_size=args.lstm_input_size,
            volume_temp=args.volume_temp,
            intersection_temp=args.intersection_temp,
            mlp_output_dim=args.mlp_output_dim,
            proj_output_dim=args.proj_output_dim,
            loss_type=args.loss_type,
            roberta_size_type=args.roberta_type,
            n_tags=n_tags,
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


def get_init_box_weights(device: torch.device):
    HierPC = 1802.0
    HierCP = 1846.0
    HierCo = 758.0
    HierNo = 63755.0 * 0.015
    HierTo = HierPC + HierCP + HierCo + HierNo  # total number of event pairs
    TempTo = 818.0
    Total = float(HierTo + TempTo)
    hier_weights = [0.25 * HierTo / HierPC, 0.25 * HierTo / HierCP, 0.25 * HierTo / HierCo, 0.25 * HierTo / HierNo, 0.5 * Total/HierTo]
    temp_weights = [0.25 * 818.0 / 412.0, 0.25 * 818.0 / 263.0, 0.25 * 818.0 / 30.0, 0.25 * 818.0 / 113.0, 0.5 * Total/TempTo]
    return torch.tensor(hier_weights, dtype=torch.float, requires_grad=True).to(device), torch.tensor(temp_weights, dtype=torch.float, requires_grad=True).to(device)
    # return torch.nn.Parameter(torch.tensor(Total/HierTo, dtype=torch.float).to(device)), \
    #        torch.nn.Parameter(torch.tensor(Total/TempTo, dtype=torch.float).to(device))

def setup(args, model_state_dict=None):
    device = cuda_if_available(args.no_cuda)
    args.data_type = args.data_type.lower()
    train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict, num_classes, n_tags = create_dataloader(args)

    if model_state_dict:
        model = create_model(args, num_classes, n_tags)
        model.load_state_dict(model_state_dict, strict=True)
        model = model.to(device)
    else:
        model = create_model(args, num_classes, n_tags)
        model = model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay) # AMSGrad

    if not args.model.startswith("box") and args.model != "vector":
        print("Using VectorBiLSTMEvaluator..!")
        evaluator = VectorBiLSTMEvaluator(
            train_type=args.data_type,
            model_type=args.model,
            model=model,
            device=device,
            valid_dataloader_dict=valid_dataloader_dict,
            test_dataloader_dict=test_dataloader_dict,
            valid_cv_dataloader_dict=valid_cv_dataloader_dict,
            test_cv_dataloader_dict=test_cv_dataloader_dict,
        )

        hier_weights, temp_weights = get_init_weights(device)
    else:
        print("Using ThresholdEvaluator..!")
        if args.eval_type == "one":
            evaluator = ThresholdEvaluator(
                train_type=args.data_type,
                model_type=args.model,
                model=model,
                device=device,
                valid_dataloader_dict=valid_dataloader_dict,
                test_dataloader_dict=test_dataloader_dict,
                valid_cv_dataloader_dict=valid_cv_dataloader_dict,
                test_cv_dataloader_dict=test_cv_dataloader_dict,
                threshold1=args.threshold1,
                eval_type=args.eval_type,
                save_plot=args.save_plot,
                wandb_id=wandb.run.id,
            )
        elif args.eval_type == "two":
            evaluator = ThresholdEvaluator(
                train_type=args.data_type,
                model_type=args.model,
                model=model,
                device=device,
                valid_dataloader_dict=valid_dataloader_dict,
                test_dataloader_dict=test_dataloader_dict,
                valid_cv_dataloader_dict=valid_cv_dataloader_dict,
                test_cv_dataloader_dict=test_cv_dataloader_dict,
                threshold1=args.threshold1,
                threshold2=args.threshold2,
                eval_type=args.eval_type,
                save_plot=args.save_plot,
                wandb_id=wandb.run.id,
            )

        hier_weights, temp_weights = get_init_box_weights(device)

    loss_anno_dict = {}
    loss_anno_dict["hieve"] = CrossEntropyLoss(weight=hier_weights)
    loss_anno_dict["matres"] = CrossEntropyLoss(weight=temp_weights)
    loss_transitivity_h = TransitivityLoss()
    loss_transitivity_t = TransitivityLoss()
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
        loss_transitivity_h=loss_transitivity_h,
        loss_transitivity_t=loss_transitivity_t,
        loss_cross_category=loss_cross_category,
        lambda_dict=lambdas_to_dict(args),
        no_valid=args.no_valid,
        wandb_id=wandb.run.id,
        eval_step=args.eval_step,
        debug=args.debug,
        patience=args.patience,
        cv_valid=args.cv_valid,
        model_save=args.model_save,
        max_grad_norm=args.max_grad_norm,
        const_eval=1 if "const_eval" not in args.keys() else args.const_eval,
        hier_weights=hier_weights,
        temp_weights=temp_weights,
        weighted=args.weighted,
    )

    return trainer, evaluator


def main():
    args = build_parser()

    if args.load_model:
        assert args.saved_model != ""
        assert args.wandb_id != ""
        model_state_dict_path = Path(args.saved_model).expanduser()
        print("Loading model state dict...", end="", flush=True)
        model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
        print("done!")

        wandb.init(reinit=True)

        api = wandb.Api()
        run = api.run(args.wandb_id)
        wandb.config.update(run.config, allow_val_change=True)
        wandb.config.update({"save_plot": 1}, allow_val_change=True)
        wandb.config.update({"symm_eval": args.symm_eval}, allow_val_change=True)
        wandb.config.update({"symm_train": args.symm_train}, allow_val_change=True)
        wandb.config.update({"model_save": 0}, allow_val_change=True)

        for key, value in sorted(vars(args).items()):
            if key not in run.config.keys():
                wandb.config.update({key: value}, allow_val_change=True)

        if args.threshold_test != 0:
            wandb.config.update({"eval_type": args.eval_type}, allow_val_change=True)
            wandb.config.update({"threshold1": args.threshold1}, allow_val_change=True)
            wandb.config.update({"threshold2": args.threshold2}, allow_val_change=True)

        args = wandb.config
        set_seed(args.seed)
        set_logger(args.data_type, args.wandb_id.replace("/", "_"))
        logger.info(args)
        trainer, evaluator = setup(args, model_state_dict)
        trainer.evaluation(-1)

    else:
        wandb.init()
        wandb.config.update(args, allow_val_change=True)
        args = wandb.config
        set_seed(args.seed)
        set_logger(args.data_type, wandb.run.id)
        logging.info(args)
        trainer, evaluator = setup(args)
        trainer.train()


if __name__ == '__main__':
    main()