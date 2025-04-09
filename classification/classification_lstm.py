import numpy as np
import torch
import torch.nn as nn
import torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics import accuracy_score


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_texts, train_labels = zip(*train_data)
test_texts, test_labels = zip(*test_data)
train_texts = [text.split() for text in train_texts]
test_texts = [text.split() for text in test_texts]
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


def extract_tfidf_features(train_texts, test_texts):
    # Convert text data to feature vectors
    train_texts = [" ".join(text) for text in train_texts]
    test_texts = [" ".join(text) for text in test_texts]
    train_texts = np.array(train_texts)
    test_texts = np.array(test_texts)
    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_texts).toarray()
    text_vectors = vectorizer.transform(test_texts).toarray()

def convert_to_tensor(train_vectors, test_vectors):
    train_vectors = torch.tensor(train_vectors, dtype=torch.float32)
    test_vectors = torch.tensor(test_vectors, dtype=torch.float32)
    # Convert the labels to PyTorch tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    # Create a DataLoader for the training and test sets
    train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

# use word2vec to get the word vectors
from gensim.models import Word2Vec
def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=2):
    sentences = [sentence.split() for sentence in sentences]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model
# create a function to get the document vector
def get_doc_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        vector_size = model.vector_size if hasattr(model, 'vector_size') else model.wv.vector_size
        return np.zeros(vector_size)
    
def implement_word2vec():
    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=train_texts, vector_size=100, window=5, min_count=1, workers=2)
    word2vec_model.build_vocab(train_texts)
    word2vec_model.train(train_texts, total_examples=len(train_texts), epochs=10)
    # get the document vectors
    train_vectors = np.array([get_doc_vector(word2vec_model, tokens) for tokens in train_texts])
    test_vectors = np.array([get_doc_vector(word2vec_model, tokens) for tokens in test_texts])
    return train_vectors, test_vectors
# Convert the document vectors to PyTorch tensors
def convert_to_tensors(train_vectors, test_vectors):
    train_vectors = torch.tensor(train_vectors, dtype=torch.float32)
    test_vectors = torch.tensor(test_vectors, dtype=torch.float32)
    # Convert the labels to PyTorch tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    # Create a DataLoader for the training and test sets
    train_dataset = torch.utils.data.TensorDataset(train_vectors, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


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
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        # if this is a binary classification problem, set output_dim to 1
        # if this is a multi-class classification problem, set output_dim to the number of classes
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
        #out = self.sigmoid(out)
        return out[:, -1]  # return the output of the last time step of LSTM
    
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
    model = LSTMClassifier(len(word_to_idx), embedding_dim=100, hidden_dim=128, output_dim=1).to(device)

    
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
        

def main():
    # Load your data here
    data = [
        ("This is a positive example", 1),
        ("This is a negative example", 0),
        # Add more examples
    ]
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_texts, train_labels = zip(*train_data)
    test_texts, test_labels = zip(*test_data)
    
    train_model(train_texts, train_labels, test_texts, test_labels)
    
if __name__ == "__main__":
    main()