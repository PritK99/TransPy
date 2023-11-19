import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any

class BilingualDataset(Dataset):
    """
    Dataset class for handling bilingual data in transformer models. This involves preprocessing the dataset and converting it into tensor. This also involves adding special tokens such as PAD, SOS, EOS to data.

    Parameters:
        - dataset (List[Dict]): List of dictionary entries representing the dataset.
        - src_tokenizer (Tokenizer): Tokenizer for the source language.
        - tgt_tokenizer (Tokenizer): Tokenizer for the target language.
        - src_lang (str): Language key indicating the source language.
        - tgt_lang (str): Language key indicating the target language.
        - max_seq_len (int): Maximum sequence length for padding.

    Attributes:
        - dataset (List[Dict]): List of dictionary entries representing the dataset.
        - src_tokenizer (Tokenizer): Tokenizer for the source language.
        - tgt_tokenizer (Tokenizer): Tokenizer for the target language.
        - src_lang (str): Language key indicating the source language.
        - tgt_lang (str): Language key indicating the target language.
        - max_seq_len (int): Maximum sequence length for padding.
        - sos_token (torch.Tensor): Tensor representing the [SOS] token for both source and target languages.
        - eos_token (torch.Tensor): Tensor representing the [EOS] token for both source and target languages.
        - pad_token (torch.Tensor): Tensor representing the [PAD] token for both source and target languages.

    Methods:
        - __len__(self): Returns the length of the dataset.
        - __getitem__(self, index: Any): Retrieves a specific item from the dataset.

    Returns:
        - dict: A dictionary containing various elements required for training or evaluation.
    """
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, max_seq_len):
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len

        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: Any):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_token = self.src_tokenizer(src_text).ids
        dec_input_token = self.tgt_tokenizer(tgt_text).ids

        enc_num_pad_token = self.max_seq_len - len(enc_input_token) - 2
        dec_num_pad_token = self.max_seq_len - len(dec_input_token) - 1

        if (enc_num_pad_token < 0 or dec_num_pad_token < 0):
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(enc_input_token, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token]*enc_num_pad_token, dtype=torch.int64)
            ]
        )

        # Adding only SOS token for the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_token, dtype=torch.int64),
                torch.Tensor([self.pad_token]*dec_num_pad_token, dtype=torch.int64)
            ]
        )

        # Adding only EOS token for the decoder input
        label = torch.cat(
            [
                torch.Tensor(dec_input_token, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token]*dec_num_pad_token, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert label.size(0) == self.max_seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    """
    Generates a causal (autoregressive) mask for self-attention mechanisms in transformer models.

    Parameters:
        - size (int): Size of the sequence, determining the dimensions of the mask.

    Returns:
        - torch.Tensor: Binary mask tensor where positions above the main diagonal are set to 1, and positions below are set to 0.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0