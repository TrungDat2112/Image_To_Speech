import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from matplotlib import pyplot as plt

from torchvision.models import resnet50,ResNet50_Weights 

from io import open
from utils import *
from trainfnc import *

if torch.cuda.is_available():
    device=torch.device(type='cuda', index=0)
else:
    device=torch.device(type='cpu', index=0)

capt_file_path="captions.txt"
images_dir_path="train\\"

#read data
data=open(capt_file_path).read().strip().split('\n')
data=data[1:]

img_filenames_list=[]
captions_list=[]

for s in data:
    templist=s.lower().split(",")
    img_path=templist[0]
    caption=",".join(s for s in templist[1:])
    caption=normalizeString(caption)
    img_filenames_list.append(img_path)
    captions_list.append(caption)

max_cap_length=73

vocab=Vocab()

for caption in captions_list:
    vocab.buildVocab(caption)

print("Vocab Length:",vocab.nwords)

dataset=CustomDataset(images_dir_path, img_filenames_list, captions_list, vocab, max_cap_length)
train_dataset,test_dataset=random_split(dataset,[0.999,0.001])

batch_size=64
train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,batch_size=1, shuffle=False)

pretrained_feature_extractor=resnet50(weights=ResNet50_Weights.DEFAULT)
pretrained_feature_extractor.fc=nn.Linear(2048,1024)

encoder=Encoder(pretrained_feature_extractor).to(device)

embed_size=300
hidden_size=1024

decoder=Decoder(vocab.nwords,embed_size,hidden_size).to(device)

def train_one_epoch():
    encoder.train()
    decoder.train()
    track_loss=0
    
    for i, (imgs,t_ids) in enumerate(train_dataloader):
        imgs=imgs.to(device)
        t_ids=t_ids.to(device)
        extracted_features=encoder(imgs)
        #extracted_features=extracted_features.detach()
        decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1))
        yhats, decoder_hidden = decoder(t_ids[:,0:-1],decoder_hidden)
                    
        gt=t_ids[:,1:]
        
        yhats_reshaped=yhats.view(-1,yhats.shape[-1])
        gt=gt.reshape(-1)
        
        
        loss=loss_fn(yhats_reshaped,gt)
        track_loss+=loss.item()
        
        opte.zero_grad()
        optd.zero_grad()
        
        loss.backward()
        
        opte.step()
        optd.step()
        
        if i%50==0:
            print("Mini Batch=", i+1," Running Loss=",track_loss/(i+1), sep="")
        
    return track_loss/len(train_dataloader)    

def eval_one_epoch():
    encoder.eval()
    decoder.eval()
    track_loss=0
    
    with torch.no_grad():
        
        for i, (imgs,t_ids) in enumerate(test_dataloader):
            
            imgs=imgs.to(device)
            t_ids=t_ids.to(device)
            
            extracted_features=encoder(imgs)
            
            decoder_hidden=torch.reshape(extracted_features,(1,extracted_features.shape[0],-1)) #n_dim=3
            
            input_ids=t_ids[:,0]
            yhats=[]
            pred_sentence=""
            
            for j in range(1,max_cap_length+2): #j starts from 1
                probs, decoder_hidden = decoder(input_ids.unsqueeze(1),decoder_hidden)
                yhats.append(probs)
                _,input_ids=torch.topk(probs,1,dim=-1)
                input_ids=input_ids.squeeze(1,2) #still a tensor
                word=vocab.index2word[input_ids.item()] #batch_size=1
                pred_sentence+=word + " "
                if input_ids.item() == 1: #batch_size=1
                    break
                                
            
            gt_sentence=ids2Sentence(t_ids,vocab)
            
            print("Input Image:")
            img=imgs[0]
            img[0]=(img[0]*0.229)+0.485
            img[1]=(img[1]*0.224)+0.456
            img[2]=(img[2]*0.225)+0.406
            plt.imshow(torch.permute(imgs[0],(1,2,0)).detach().cpu())
            plt.show()
            
            print("GT Sentence:",gt_sentence)
            print("Predicted Sentence:",pred_sentence)
            
            yhats_cat=torch.cat(yhats,dim=1)
            yhats_reshaped=yhats_cat.view(-1,yhats_cat.shape[-1])
            gt=t_ids[:,1:j+1]
            gt=gt.view(-1)
            

            loss=loss_fn(yhats_reshaped,gt)
            track_loss+=loss.item()
            
            
        print("-----------------------------------")
        return track_loss/len(test_dataloader)    
    
loss_fn=nn.NLLLoss(ignore_index=0).to(device)
lr=0.001

optd=optim.Adam(params=decoder.parameters(), lr=lr)
opte=optim.Adam(params=encoder.parameters(), lr=lr)

n_epochs=5

for e in range(n_epochs):
    print("Epoch=",e+1, " Loss=", round(train_one_epoch(),4), sep="")

for e in range(1):
    print("Epoch=",e+1, " Loss=", round(eval_one_epoch(),4), sep="")