import random
import re
from typing import List

import pandas as pd
from loguru import logger
from texy.pipelines import strict_clean
from tqdm.notebook import tqdm

SAMPLE_SIZE = 5000


def sample_from_df(df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> List[str]:
    df = df.sample(sample_size)
    return strict_clean(df["article"])  # pyright: ignore


def split_sentence(text: str) -> List[str]:
    pattern = r"(?<=[।\?!])\s*"
    res = re.split(pattern, text)
    return [r.strip() + "\n" for r in res if len(r.strip()) > 0]


def extract_lines(texts: List[str]) -> List[str]:
    res = []
    for text in texts:
        lines = split_sentence(text)
        res.extend(lines)
    return res


def build_dataset(dataset_path: str) -> List[str]:
    df = pd.read_csv(dataset_path)
    samples = sample_from_df(df)
    return extract_lines(samples)


def write_ground_truth(paths: List[str], output_file_path: str):
    text_stream = []
    with open(f"{output_file_path}", "w") as file:
        for path in tqdm(paths):
            tmp = list(set(build_dataset(dataset_path=path)))
            text_stream.extend(tmp)
        file.writelines(text_stream)
    random.shuffle(text_stream)
    logger.info(f"Total lines: {len(text_stream)}")


def write_incorrects(lines: List[str], output_file_path: str):
    with open(f"{output_file_path}", "w") as file:
        lines = [line + "\n" for line in lines]
        file.writelines(lines)
    logger.info(f"Total lines: {len(lines)}")
