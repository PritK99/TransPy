import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, target_tokenizer, src_lang, target_lang, max_seq_len) -> None:
        super.__init__()

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.max_seq_len = max_seq_len
        
        self.sos_token = torch.Torch([src_tokenizer.token_to_id(['SOS'])], dtype = torch.int64)
        self.eos_token = torch.Torch([src_tokenizer.token_to_id(['EOS'])], dtype = torch.int64)
        self.pad_token = torch.Torch([src_tokenizer.token_to_id(['PAD'])], dtype = torch.int64)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]

        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        # Including SOS and EOS tokens
        enc_num_padding_tokens = self.max_seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype = torch.int64)
            ]
        )        
