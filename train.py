import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(dataset, lang):
    """
    Generator function to extract sentences from a dataset for a specified language.

    Parameters:
        - dataset (List[Dict]): List of dictionary entries representing the dataset.
        - lang (str): Language key indicating the required language.

    Yields:
        - sentence (str): Extracted sentence for the specified language.
    """
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    """
    Retrieves an existing tokenizer or builds a new one for a specified language.

    Parameters:
        - config (Dict): Configuration dictionary containing tokenizer_file path.
        - dataset (List[Dict]): List of dictionary entries representing the dataset.
        - lang (str): Language key indicating the target language.

    Returns:
        - tokenizer (Tokenizer): Tokenizer object for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'])
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    dataset_name = "opus_books"
    dataset = load_dataset(dataset_name, f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    src_tokenizer = get_or_build_tokenizer(config, dataset, config["src_lang"])
    tgt_tokenizer = get_or_build_tokenizer(config, dataset, config["tgt_lang"])

    training_data, validation_data = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])