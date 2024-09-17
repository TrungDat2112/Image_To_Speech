import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from textwrap import wrap
import re
import torch.optim as optim
from collections import Counter
from utils import *
from tqdm import tqdm
caption_file_path = "captions.txt"  
image_folder_path = "train"         

data = pd.read_csv(caption_file_path)

def readImage(image_name, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_path = os.path.join(image_folder_path, image_name)  
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img


def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(15):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = readImage(temp_df.image[i])  
        plt.imshow(image.permute(1, 2, 0).numpy())  
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))  
        plt.axis("off")
    plt.show()


def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    
    data['caption'] = data['caption'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
    
    data['caption'] = data['caption'].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    
    data['caption'] = data['caption'].apply(lambda x: "startseq " + x + " endseq")
    return data


class Tokenizer:
    def __init__(self, num_words=None, oov_token="<OOV>"):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self.num_words = num_words  
        self.oov_token = oov_token  
        self.word_index[self.oov_token] = 1  
        self.index_word[1] = self.oov_token 
    
    def fit_on_texts(self, texts):
        counter = Counter()
        
        for text in texts:
            counter.update(text.split())
        
        most_common_words = counter.most_common(self.num_words) if self.num_words else counter.items()
        
        index = 2
        for word, _ in most_common_words:
            self.word_index[word] = index
            self.index_word[index] = word
            index += 1
        
        self.vocab_size = len(self.word_index) + 1
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequences.append([self.word_index.get(word, 1) for word in text.split()])
        return sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")


data = text_preprocessing(data)
captions = data['caption'].tolist()



tokenizer.fit_on_texts(captions)


vocab_size = tokenizer.vocab_size
print(vocab_size)
max_length = max(len(caption.split()) for caption in captions)


images = data['image'].unique().tolist()
nimages = len(images)
split_index = round(0.85 * nimages)

train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)]
test = data[data['image'].isin(val_images)]

train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

tokenized_caption = tokenizer.texts_to_sequences([captions[100]])[0]
print(tokenized_caption)
print(captions[100])
display_images(train.sample(15))
model = models.densenet201(pretrained=True)
fe = nn.Sequential(*list(model.children())[:-1])  

img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fe = fe.to(device)
fe.eval()  

features = {}
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for image in tqdm(data['image'].unique().tolist()):
    img_path = os.path.join(image_folder_path, image)
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = fe(img).cpu().numpy()
    
    features[image] = feature

vocab_size = 10000  
max_length = 20     
caption_model = CaptioningModel(vocab_size, max_length)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(caption_model.parameters(), lr=0.001)

caption_model.to(device)