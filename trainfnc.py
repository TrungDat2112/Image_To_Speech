import re
import unicodedata

def normalizeString(s): 
    sres=""
    for ch in unicodedata.normalize('NFD', s): 
        if unicodedata.category(ch) != 'Mn':
            sres+=ch
    sres = re.sub(r"[^a-zA-Z!?,]+", r" ", sres) 
    return sres.strip()

def ids2Sentence(ids,vocab):
    sentence=""
    for id in ids.squeeze():
        if id==0:
            continue
        word=vocab.index2word[id.item()]
        sentence+=word + " "
        if id==1:
            break
    return sentence