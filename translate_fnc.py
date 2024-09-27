import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import re,string
from typing import Iterable

# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# SRC_LANGUAGE = 'en'
# TGT_LANGUAGE = 'vi'
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# text_transform = {}

def word_tokenize(text):
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return tokens

def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence)
    return tokens

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str, token_transform):    
    for index,data_sample in data_iter:
        yield token_transform[language](data_sample[language])

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

def generate_square_subsequent_mask(sz,DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, DEVICE, PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids, BOS_IDX, EOS_IDX):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def collate_fn(batch, text_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX):
    src_batch, tgt_batch = [], []
    
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer, loss_fn, accumulation_steps,train_ds, BATCH_SIZE, DEVICE):
    model.train()
    losses = 0
    val_los = 0
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    optimizer.zero_grad() 
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)   
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = loss / accumulation_steps 
        loss.backward()
        
        if (i+1) % accumulation_steps == 0:        
            optimizer.step() 
            optimizer.zero_grad() # Reset gradients tensor  

        losses += loss.item()

    return losses / len(train_dataloader)

def evaluate(model, loss_fn, accumulation_steps, val_ds, BATCH_SIZE, DEVICE):
    model.eval()
    losses = 0

    #val_iter = valid.iterrows()
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = loss / accumulation_steps 
        losses += loss.item()

    return losses / len(val_dataloader)