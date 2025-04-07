import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

"""
The principle of text generation using LSTM models is learning sequential patterns 
in text to predict the most likely next character or word given a context, 
then recursively using these predictions as new context to generate text continuously.
"""

class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        """_summary_

        Args:
            text (_type_): _text_
            seq_len (_type_): _sequence length_
        """
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(list(set(text))) # Get unique characters
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)} # Map characters to indices
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)} # Map indices to characters
        self.n_chars = len(self.chars) # Number of unique characters
        
    def __len__(self):
        return len(self.text) - self.seq_len # Number of sequences
    
    def __getitem__(self, idx):
        x_seq = self.text[idx:idx+self.seq_len] # Input sequence
        y_seq = self.text[idx+1:idx+self.seq_len+1] # Target sequence
        
        # Convert characters to indices
        x = torch.tensor([self.char_to_idx[ch] for ch in x_seq], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[ch] for ch in y_seq], dtype=torch.long)
        return x, y 
    
class LSTMTextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTextGenerator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0) # the shape of x is (batch_size, seq_len)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.fc(lstm_out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state, cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
        return (h0, c0)
    
def train(model, dataloader, criterion, optimizer, epochs, device='cpu'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, model.fc.out_features), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        return total_loss / len(dataloader)

def generate_text(model, dataset, seed_text, length=100, temperature=0.6, device='cpu'):
    model.eval()
    
    # convert seed text to indices
    seed_indices = [dataset.char_to_idx[ch] for ch in seed_text] # (seq_len,)
    seed_tensor = torch.tensor(seed_indices, dtype=torch.long).unsqueeze(0).to(device) # (1, seq_len)
    
    # Initialize hidden state
    hidden = None
    generated_text = seed_text
    
    # Generate characters one by one
    with torch.no_grad():
        for _ in range(length):
            # Forward pass
            output, hidden = model(seed_tensor, hidden) # (1, seq_len, output_size)
            # get the ooutput for the last time step
            output = output[:, -1, :] # (batch_size, output_size)
            # Apply temperature scaling
            output = output / temperature
            # Get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1).squeeze() # (output_size,)
            
            # Sample from the distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            
            # Append the predicted character to the generated text
            generated_text += dataset.idx_to_char[next_char_idx]
            # Update the seed tensor, the shape of seed_tensor is (1, seq_len), the shape of next_char_idx is (1,)
            seed_tensor = torch.cat([seed_tensor[:, 1:], torch.tensor([[next_char_idx]]).to(device)], dim=1)
            
    return generated_text
        
    

    
def main():
    # 加载文本数据
    with open('corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 设置超参数
    seq_length = 100  # 序列长度
    batch_size = 64
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.001
    epochs = 20
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建数据集和数据加载器
    dataset = TextDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = LSTMTextGenerator(
        input_size=dataset.n_chars,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=dataset.n_chars
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练模型...")
    train(model, dataloader, criterion, optimizer, device, epochs)
    
    # 保存模型
    torch.save(model.state_dict(), 'lstm_text_generator.pth')
    print("模型已保存到 'lstm_text_generator.pth'")
    
    # 生成文本示例
    seed_text = text[:seq_length]  # 使用语料库中的前seq_length个字符作为种子文本
    generated_text = generate_text(model, dataset, seed_text, length=500, temperature=0.8, device=device)
    print("\n生成的文本:")
    print(generated_text)

if __name__ == "__main__":
    main()