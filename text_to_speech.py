import pandas as pd
from torchtext.data.utils import get_tokenizer
from typing import Iterable
from torchtext.vocab import build_vocab_from_iterator
import re,string
import torch
from torch import nn
from seqtoseq import *
from translate_fnc import *

data_dir = "englishvietnamese"
en_sents = open(data_dir + 'en_sents', "r").read().splitlines()
vi_sents = open(data_dir + 'vi_sents', "r").read().splitlines()
raw_data = {
        "en": [line for line in en_sents[:170000]], # Only take first 170000 lines
        "vi": [line for line in vi_sents[:170000]],
    }
df = pd.DataFrame(raw_data, columns=["en", "vi"])

def word_tokenize(text):
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def preprocessing(df): 
    df["en"] = df["en"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) # Remove punctuation
    df["vi"] = df["vi"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))  
    df["en"] = df["en"].apply(lambda ele: ele.lower()) # convert text to lowercase
    df["vi"] = df["vi"].apply(lambda ele: ele.lower())
    df["en"] = df["en"].apply(lambda ele: ele.strip()) 
    df["vi"] = df["vi"].apply(lambda ele: ele.strip()) 
    df["en"] = df["en"].apply(lambda ele: re.sub("\s+", " ", ele)) 
    df["vi"] = df["vi"].apply(lambda ele: re.sub("\s+", " ", ele))
        
    return df

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'

# Place-holders
token_transform = {}
vocab_transform = {}

# Tokenize for vietnames by underthesea
def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence)
    return tokens

token_transform[SRC_LANGUAGE] = get_tokenizer('basic_english')
token_transform[TGT_LANGUAGE] = get_tokenizer(vi_tokenizer)

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str):    
    for index,data_sample in data_iter:
        yield token_transform[language](data_sample[language])

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = df.iterrows()
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(0)
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8 # embed_dim must be divisible by num_heads
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROP_OUT = 0.1

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Split data to tran test set
split_ratio = 0.9
split = round(df.shape[0]* split_ratio)
train = df.iloc[:split]
train_ds = list(zip(train['en'],train['vi']))
valid = df.iloc[split:]
val_ds = list(zip(valid['en'],valid['vi']))