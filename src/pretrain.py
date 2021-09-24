import argparse
import os
import random

import pandas as pd
import spacy
import torch
import tqdm
import math
import numpy as np

from os import listdir
from os.path import isfile, join
from pathlib import Path

from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 50        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-5    # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42               # random seed (default: 42)
MAX_LEN = 128
SUMMARY_LEN = 7

nlp = spacy.load("en_core_web_sm")


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

        self.encoded_line = []
        for line in data:
            x = self.tokenizer.encode(line)
            self.encoded_line += [x]

    def __len__(self):
        return len(self.encoded_line)

    def __getitem__(self, item):
        return torch.tensor(self.encoded_line[item])


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic=True


def create_txt_file(data_dir, txt_path):
    hieve_dir = data_dir / "hievents_v2/processed/"
    hieve_files = natsorted([f for f in listdir(hieve_dir) if isfile(join(hieve_dir, f)) and f[-4:] == "tsvx"])
    # write lines to .txt file
    print(f"Writing file contents into {os.path.abspath(txt_path)}...")
    with open(txt_path, "w") as output:
        for file in tqdm.tqdm(hieve_files):
            # print(file)
            file_path = hieve_dir / file
            for line in open(file_path, mode="r"):
                line = line.split("\t")
                type = line[0].lower()
                if type == "text":
                    text = line[1]
                    tokens = nlp(text)
                    for sent in tokens.sents:
                        output.write(f"{sent.string.strip()}\n")
    print("done!")


def get_num_lines(txt_path):
    num_lines = len(list(open(txt_path)))
    print("# of lines:", num_lines)
    return num_lines


def get_train_test_split(data_dir, txt_path):
    with open(txt_path, "r") as input:
        lines = input.readlines()
        lines = [line.rstrip() for line in lines]

    train, test = train_test_split(lines, test_size=0.2)
    train_file = data_dir / "article_train_split.txt"
    test_file = data_dir / "article_test_split.txt"
    with open(train_file, "w") as f:
        for line in train:
            f.write(f"{line.strip()}\n")

    with open(test_file, "w") as f:
        for line in test:
            f.write(f"{line.strip()}\n")

    return {
        "train": os.path.abspath(train_file),
        "test": os.path.abspath(test_file),
    }


def setup(roberta_model_save_dir, file_path_dict):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_dataset = CustomDataset(file_path_dict["train"], tokenizer)
    test_dataset = CustomDataset(file_path_dict["test"], tokenizer)

    data_collector = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=roberta_model_save_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        save_steps=8192,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collector,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    return trainer


def verify_model(roberta_path):
    fill_mask = pipeline(
        "fill-mask",
        model=roberta_path,
        tokenizer="roberta-base",
    )
    print(fill_mask("Firefighters and <mask> crews were called to the scene."))
    print(fill_mask("Instead, he was able to trigger the <mask>."))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("pretrain arg parser")
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--create_file", default=False, action="store_true")
    parser.add_argument("--save_dir", default="./roberta_retrained_tmp/", help="roberta model save directory", type=str)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser()

    set_seed()

    roberta_model_save_dir = args.save_dir
    Path(roberta_model_save_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = roberta_model_save_dir

    txt_path = data_dir / "all_article_text.txt"

    if args.create_file:
        create_txt_file(data_dir, txt_path)
    get_num_lines(txt_path)
    file_path_dict = get_train_test_split(data_dir, txt_path)

    trainer = setup(roberta_model_save_dir, file_path_dict)

    trainer.train()
    trainer.save_model(roberta_model_save_dir)

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    verify_model(roberta_model_save_dir)
