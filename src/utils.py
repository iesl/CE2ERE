import datetime

from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Union


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_files(data_dir: Union[Path, str], type: str):
    if type == "hieve":
        dir_path = data_dir / "hievents_v2/processed/"
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    elif type == "matres":
        print("matres!!")
        # dir_path = data_dir / "hievents_v2/processed/"
        # files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    else:
        raise NotImplementedError(f"dataset={type} unsupported at this time")

    return dir_path, files