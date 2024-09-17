import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class CustomDataset(Dataset):
    
    def __init__(self, df, X_col, y_col, tokenizer, vocab_size, max_length, features):
        self.df = df
        self.X_col = X_col
        self.y_col = y_col
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        batch = self.df.iloc[index]
        X1, X2, y = self.__get_data(batch)
        return X1, X2, y
    
    def __get_data(self, batch):
        image = batch[self.X_col]
        feature = torch.tensor(self.features[image][0], dtype=torch.float32)
        captions = batch[self.y_col]
        X1, X2, y = [], [], []
        
        for caption in captions:
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = F.pad(torch.tensor(in_seq, dtype=torch.long), (0, self.max_length - len(in_seq)), value=0)
                out_seq = F.one_hot(torch.tensor(out_seq), num_classes=self.vocab_size).float()
                
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        
        return torch.stack(X1), torch.stack(X2), torch.stack(y)

class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(CaptioningModel, self).__init__()
        self.dense_img = nn.Linear(1920, 256)  
        self.reshape = nn.Unsqueeze(1)  
        self.embedding = nn.Embedding(vocab_size, 256)  
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, img_features, captions):
        img_features = self.dense_img(img_features)
        img_features = self.reshape(img_features)
        
        caption_embeds = self.embedding(captions)
        
        combined = torch.cat((img_features, caption_embeds), 1)
        lstm_out, _ = self.lstm(combined)
        
        lstm_out = self.dropout(lstm_out[:, -1, :])  
        x = self.fc1(lstm_out)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output


