import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset
from model import build_transformer

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

    training_data_raw, validation_data_raw = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])

    training_data = BilingualDataset(training_data_raw, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["max_seq_len"])
    validation_data = BilingualDataset(validation_data_raw, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["max_seq_len"])

    max_src_len = 0
    max_tgt_len = 0

    for item in dataset:
        src_ids = src_tokenizer.encode(item["translation"][config["src_lang"]]).ids
        max_src_len = max(max_src_len, len(src_ids))
        tgt_ids = tgt_tokenizer.encode(item["translation"][config["tgt_lang"]]).ids
        max_tgt_len = max(max_tgt_len, len(tgt_ids))

    print(f'Max length of source sentence: {max_src_len}')
    print(f'Max length of target sentence: {max_tgt_len}')

    train_dataloader = DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, src_vocab_len, tgt_vocab_len):
    model = build_transformer(src_vocab_len, tgt_vocab_len, config["max_seq_len"], config["max_seq_len"], config["embedding_dim"])
    return model