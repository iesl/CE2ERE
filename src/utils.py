import contextlib
import datetime
import logging
import time

from os import listdir
from os.path import isfile, join

import torch
import random
from pathlib import Path
from typing import *


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_hieve_files(data_dir: Union[Path, str]):
    dir_path = data_dir / "hievents_v2/processed/"
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return dir_path, files


def get_matres_files(data_dir: Union[Path, str]):
    all_tml_dir_path_dict = {}
    all_tml_file_dict = {}
    all_txt_file_path = []

    TB_tml_dir_path = data_dir / "MATRES/TBAQ-cleaned/TimeBank/"
    AQ_tml_dir_path = data_dir / "MATRES/TBAQ-cleaned/AQUAINT/"
    PL_tml_dir_path = data_dir / "MATRES/te3-platinum/"

    all_tml_dir_path_dict["tb"] = TB_tml_dir_path
    all_tml_dir_path_dict["aq"] = AQ_tml_dir_path
    all_tml_dir_path_dict["pl"] = PL_tml_dir_path

    TB_tml_files = [f for f in listdir(TB_tml_dir_path) if isfile(join(TB_tml_dir_path, f))]
    AQ_tml_files = [f for f in listdir(AQ_tml_dir_path) if isfile(join(AQ_tml_dir_path, f))]
    PL_tml_files = [f for f in listdir(PL_tml_dir_path) if isfile(join(PL_tml_dir_path, f))]

    all_tml_file_dict["tb"] = TB_tml_files
    all_tml_file_dict["aq"] = AQ_tml_files
    all_tml_file_dict["pl"] = PL_tml_files

    TB_txt_file_path = data_dir / "MATRES/timebank.txt"
    AQ_txt_file_path = data_dir / "MATRES/aquaint.txt"
    PL_txt_file_path = data_dir / "MATRES/platinum.txt"

    all_txt_file_path.append(TB_txt_file_path)
    all_txt_file_path.append(AQ_txt_file_path)
    all_txt_file_path.append(PL_txt_file_path)

    return all_tml_dir_path_dict, all_tml_file_dict, all_txt_file_path


def get_tml_dir_path(tml_file_name: str, all_tml_dir_path_dict: Dict, all_tml_file_dict: Dict) -> str:
    if tml_file_name in all_tml_file_dict["tb"]:
        dir_path = all_tml_dir_path_dict["tb"]
    elif tml_file_name in all_tml_file_dict["aq"]:
        dir_path = all_tml_dir_path_dict["aq"]
    elif tml_file_name in all_tml_file_dict["pl"]:
        dir_path = all_tml_dir_path_dict["pl"]
    else:
        raise ValueError(f"tml file={tml_file_name} does not exist!")
    return dir_path


def lambdas_to_dict(args: Dict[str, Any]) -> Dict[str, float]:
    lambda_dict = {}
    lambda_dict["lambda_anno"] = args.lambda_anno
    lambda_dict["lambda_trans"] = args.lambda_trans
    lambda_dict["lambda_cross"] = args.lambda_cross
    lambda_dict["lambda_pair"] = args.lambda_pair
    return lambda_dict


def cuda_if_available(no_cuda: bool) -> torch.device:
    logger = logging.getLogger("CE2ERE")
    cuda = not no_cuda and torch.cuda.is_available()
    if not no_cuda and not torch.cuda.is_available():
        logger.info("Requested CUDA but it is not available, running on CPU")
    return torch.device("cuda" if cuda else "cpu")


def set_logger(data_type: str, wandb_id: str):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')
    log_dir = "./log/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging_path = log_dir + f"{data_type}_{timestamp}_{wandb_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


_LOG1MEXP_SPLIT_POINT = torch.tensor(0.5).log()
def log1mexp(x: torch.Tensor, split_point=_LOG1MEXP_SPLIT_POINT, exp_zero_eps=1e-7) -> torch.Tensor:
    """
    Computes log(1 - exp(x)).
    NOTE: this is *not* the same as (eg.) log1mexp from R, which computes log(1-exp(-x)).

    Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

    = log1p(-exp(x)) when x <= log(1/2)
    or
    = log(-expm1(x)) when log(1/2) < x <= 0

    For details, see

    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
    """
    logexpm1_switch = x > split_point
    Z = torch.zeros_like(x)
    # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
    # ie. p=1
    logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-30))
    # hack the backward pass
    # if expm1(x) gets very close to zero, then the grad log() will produce inf
    # and inf*0 = nan. Hence clip the grad so that it does not produce inf
    logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
    Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
    Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))
    return Z