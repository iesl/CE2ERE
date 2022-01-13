# Event-Event Relation Extraction using Probabilistic Box Embedding

This is the repository for the Paper "Event-Event Relation Extraction using Probabilistic Box Embedding". This repository contains the source code and datasets used in our paper.

## Abstract

To understand a story with multiple events, it is important to capture the proper relations across these events. However, existing event relation extraction (ERE) framework regards it as a multi-class classification task and do not guarantee any coherence between different relation types, such as anti-symmetry. If a phone line "died" after "storm", then it is obvious that the "storm" happened before the "died". Current framework of event relation extraction do not guarantee this coherence and thus enforces it via constraint loss function (Wang et al., 2020). In this work, we propose to modify the underlying ERE model to guarantee coherence by representing each event as a box representation (BERE) without applying explicit constraints. From our experiments, BERE also shows stronger conjunctive constraint satisfaction while performing on par or better in F1 compared to previous models with constraint injection.

## How to run the code
### Environment Setup et al.
```
git clone https://github.com/iesl/CE2ERE.git
conda env create -n bere -f environment.yml
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
Tested with Python 3.8 and PyTorch 1.7.

### Example Command
#### Command for BERE-p over MATRES
main.py --data_dir=data --model box --data_dir data --data_type matres --downsample 0.02 --epochs 50 --lambda_anno 1 --lambda_trans 0 --learning_rate 1e-05 --log_batch_size 6 --lstm_hidden_size 256 --lstm_input_size 768 --mlp_size 512 --roberta_hidden_size 768 --num_layers 1 --symm_eval 0 --loss_type 4

(BERE - loss_type=0, BERE-p - loss_type=4, BERE-c - loss_type=3

## Reference
Bibtex:
```
@article{hwang2022bere,
  title={Event-Event Relation Extraction using Probabilistic Box Embedding},
  author={EunJeong Hwang, Jay-Yoon Lee, Tianyi Yang, Dhruvesh Patel, Dongxu Zhang, Andrew McCallum},
  journal={}, TBD
  year={2022}
}
```
