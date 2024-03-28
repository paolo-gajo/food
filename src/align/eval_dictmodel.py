import warnings
import os
import torch
from tqdm.auto import tqdm
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import data_loader, SquadEvaluator, TASTEset
import pandas as pd
import json

def main():
    data_repo = 'pgajo/mdeberta_GZ-GOLD-NER-ALIGN_105_U1_S0_DROP0'
    data = TASTEset.from_datasetdict(data_repo)

    ew_taste_path = '/home/pgajo/working/food/data/TASTEset/data/EW-TASTE/EW-TASTE_en-it_DEEPL_LOC.json'

    with open(ew_taste_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)

    




if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()