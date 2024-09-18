import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

class Vocab:
    def __init__(self):
        self.word2index={'SOS':0, 'EOS':1}
        self.index2word={0:'SOS', 1:'EOS'}
        self.word2count={}
        self.nwords=2
    
    def buildVocab(self,s):
        for word in s.split(" "):
            if word not in self.word2index:
                self.word2index[word]=self.nwords
                self.index2word[self.nwords]=word
                self.word2count[word]=1
                self.nwords+=1
            else:
                self.word2count[word]+=1

class CustomDataset(Dataset):
    def __init__(self,images_dir_path, img_filenames_list, captions_list, vocab, max_cap_length):
        super().__init__()
        self.images_dir_path=images_dir_path
        self.img_filenames_list=img_filenames_list
        self.captions_list=captions_list
        self.length=len(self.captions_list)
        self.transform=Compose([Resize((224,224), antialias=True), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.vocab=vocab
        self.max_cap_length=max_cap_length
    
    def __len__(self):
        return self.length
    
    def get_input_ids(self, sentence,vocab):
        input_ids=[0]*(self.max_cap_length+1)
        i=0
        for word in sentence.split(" "):
            input_ids[i]=vocab.word2index[word]
            i=i+1

        input_ids.insert(0,vocab.word2index['SOS'])
        i=i+1
        input_ids[i]=vocab.word2index['EOS']
    
        return torch.tensor(input_ids)
    
    def __getitem__(self,idx):
        imgfname,caption=self.img_filenames_list[idx],self.captions_list[idx]
        
        imgfname=self.images_dir_path+imgfname
        img=Image.open(imgfname)
        img=self.transform(img)
        
        caption=self.get_input_ids(caption,self.vocab)       
        
        return img,caption

class Encoder(nn.Module):
    def __init__(self, pretrained_feature_extractor):
        super().__init__()
        self.pretrained_feature_extractor=pretrained_feature_extractor
        
    def forward(self,x):
        features=self.pretrained_feature_extractor(x)
        return features

class Decoder(nn.Module):
    def __init__(self,output_size,embed_size,hidden_size):
        super().__init__()
        self.e=nn.Embedding(output_size,embed_size)
        self.relu=nn.ReLU()
        self.gru=nn.GRU(embed_size, hidden_size, batch_first=True)
        self.lin=nn.Linear(hidden_size,output_size)
        self.lsoftmax=nn.LogSoftmax(dim=-1)
    
    def forward(self,x,prev_hidden):
        x=self.e(x)
        x=self.relu(x)
        output,hidden=self.gru(x,prev_hidden)
        y=self.lin(output)
        y=self.lsoftmax(y)
        return y, hidden
