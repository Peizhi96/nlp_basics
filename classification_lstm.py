import numpy as np
import torch
import torch.nn as nn
import torch.util.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len    

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = [self.word_to_idx.get(word, 0) for word in text.split()]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.squeeze(0) #[batch_size, hidden_dim]
        
        out = self.fc1(hidden)
        out = self.dropout(out)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.softmax(out)
        #out = self.sigmoid(out)
        return out
    
def train_model(train_data, train_labels, test_data, test_labels):
    word_counts = Counter()
    for text in train_data:
        word_counts.update(text.split())
        
    # create a word to index mapping
    word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.items())}
    word_to_idx['<PAD>'] = 0
    
    # get the maximum length of the sequences
    max_len = max(len(text.split()) for text in train_data)
    
    train_dataset = TextDataset(train_data, train_labels, word_to_idx, max_len)
    test_dataset = TextDataset(test_data, test_labels, word_to_idx, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(len(word_to_idx)+1, embedding_dim=100, hidden_dim=128, output_dim=1, input_length=max_len).to(device)

    
    #criterion = nn.BCELoss() if binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')