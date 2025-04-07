import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import math

class TransDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        """_summary_

         Args:
            src_sentences: 源语言句子列表
            tgt_sentences: 目标语言句子列表
            src_vocab: 源语言词汇表（字典）
            tgt_vocab: 目标语言词汇表（字典）
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # transform sentences to indices, and add <sos> and <eos> tokens
        src_sentence = ['<sos>'] + self.src_sentences[idx].split() + ['<eos>']
        tgt_sentence = ['<sos>'] + self.tgt_sentences[idx].split() + ['<eos>']
        
        # transform words to indices
        src_indices = [self.src_vocab.get(word, self.src_vocab['<unk>']) for word in src_sentence]
        tgt_indices = [self.tgt_vocab.get(word, self.src_vocab['<unk>']) for word in tgt_sentence]
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

def collate_fn(batch):
    """_summary_

    deal with batch data, and fill the sentences to the same length
    """
    src_batch, tgt_batch = zip(*batch)
    
    # the size of src_padded and tgt_padded is 2D: (batch_size, max_len)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True)
    
    return src_padded, tgt_padded

class PositionalEncoding(nn.Module):
    """position encoding layer for transformer
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        # create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # create a position tensor of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # create a div_term tensor of shape (1, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # fill the even indices with sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # fill the odd indices with cos
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # add positional encoding to the input tensor
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class Transformer_translation(nn.Module):
    """_summary_

    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 嵌入维度
        nhead: 多头注意力机制的头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Transformer_translation, self).__init__()
        
        # create embedding layer
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # save d_model's parameter
        self.d_model = d_model
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        # generate mask for padding
        src_key_padding_mask = self.create_pad_mask(src, 0) # (batch_size, src_seq_len)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)# (tgt_seq_len, tgt_seq_len)
        tgt_key_padding_mask = self.create_pad_mask(tgt, 0) # (batch_size, tgt_seq_len)
        
        # get the source and target embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # add positional encoding (batch_size, seq_len, d_model)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        
         # Transformer要求输入形状为(seq_len, batch_size, d_model)
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        
        # pass through transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        output = output.permute(1, 0, 2) # (batch_size, tgt_seq_len, d_model)
        
        
        # pass through final linear layer
        output = self.fc_out(output) # (batch_size, tgt_seq_len, tgt_vocab_size)
        
        return output
    
    def create_pad_mask(self, matrix, pad_idx):
        """_summary_

        Args:
            matrix: (batch_size, seq_len)
            pad_idx: padding index

        Returns:
            mask: (batch_size, seq_len) 
        """
        return (matrix == pad_idx) # return a boolean mask, where True means padding

def train(model, dataloader, optimizer, criterion, epochs, device='cpu'):
    """_summary_

    Args:
        model: transformer model
        dataloader: dataloader for training data
        optimizer: optimizer
        criterion: loss function
        epochs: number of epochs
    """
    model.train()
    for epoch in range(epochs):
        
        total_loss = 0
        
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # prepare input and target
            # src: (batch_size, src_seq_len)
            # tgt: (batch_size, tgt_seq_len)    
            tgt_input = tgt[:, :-1] # decoder input, exclude last token: <eos>
            tgt_output = tgt[:, 1:] # encoder output, exclude first token: <sos>
            
            # forward pass
            output = model(src, tgt_input)
            
            # reshape to (batch_size * tgt_seq_len, d_model)
            output_dim = output.shape[-1]
            output_flat = output.contiguous().view(-1, output_dim) # (batch_size * tgt_seq_len, d_model)
            tgt_output_flat = tgt_output.contiguous().view(-1) # (batch_size * tgt_seq_len)
            
            # calculate loss
            loss = criterion(output_flat, tgt_output_flat)
            
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

def translate(model, src_sentence, src_vocab, tgt_vocab, max_len=50, device='cpu'):
    """_summary_

    Args:
        model: transformer model
        src_sentence: source sentence
        src_vocab: source vocabulary
        tgt_vocab: target vocabulary
        max_len: maximum length of target sentence
        device: device to use (cpu or gpu)

    Returns:
        translated sentence
    """
    model.eval()
    
    # transform sentence to indices
    src_tokens = ['<sos>'] + src_sentence.split() + ['<eos>']
    src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in src_tokens]
    src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device) # transfrom (src_seq_len) to (1, src_seq_len)
    
    
    # generate target sentence
    tgt_indices = [tgt_vocab['<sos>']] # start with <sos>
    tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(device) # (1, 1), there's only <sos> in the first time
    
    for _ in range(max_len): # max length of target sentence
        
        with torch.no_grad():
            output = model(src, tgt_tensor)
        # output: (1, tgt_seq_len, tgt_vocab_size)
        
        # output[:, -1, :] is the last time step's output, [1, tgt_seq_len, tgt_vocab_size]
        # argmax to get the index of the word with the highest probability
        next_token = output[:, -1, :].argmax(dim=-1).item() # get the last token
        tgt_indices.append(next_token)
        
        if next_token == tgt_vocab['<eos>']:
            break
        
        tgt = torch.tensor([tgt_indices], dtype=torch.long).to(device) # (1, tgt_seq_len)
        
    # convert indices to words
    id_to_word = {idx: word for word, idx in tgt_vocab.items()}
    tgt_tokens = [id_to_word[idx] for idx in tgt_indices]  
    
    # 移除<sos>和<eos>
    if tgt_tokens[-1] == '<eos>':
        tgt_tokens = tgt_tokens[1:-1]
    else:
        tgt_tokens = tgt_tokens[1:]
    
    return ' '.join(tgt_tokens)

# Example usage 

if __name__ == "__main__":
    # 假设我们有一个小型平行语料库
    src_sentences = ["I love you", "Hello world"]
    tgt_sentences = ["Je t'aime", "Bonjour le monde"]
    
    # 构建词汇表
    src_vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3, 'I':4, 'love':5, 'you':6, 'Hello':7, 'world':8}
    tgt_vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3, 'Je':4, "t'aime":5, 'Bonjour':6, 'le':7, 'monde':8}
    
    # 准备数据
    dataset = TransDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer_translation(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充位置的损失
    
    # 训练模型（示例训练，实际需要更多数据和epoch）
    train(model, dataloader, optimizer, criterion, epochs=10)
    
    # 进行翻译
    test_sentence = "Hello world"
    translation = translate(model, test_sentence, src_vocab, tgt_vocab, device)
    print(f"Translation: {translation}")