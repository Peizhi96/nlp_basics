import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        
        assert self.head_dim * num_head == self.embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # linear layers for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # linear transformation
        q = self.q_linear(q) # (batch_size, seq_len, embed_dim)
        k = self.k_linear(k) # (batch_size, seq_len, embed_dim)
        v = self.v_linear(v) # (batch_size, seq_len, embed_dim)
        
        # reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        
        # calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = scores / math.sqrt(self.head_dim) # scale scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # apply attention weights to V, get weighted sum of value
        # (batch_size, num_heads, seq_len_q, seq_len_v) * (batch_size, num_heads, seq_len_v, head_dim)
        output = torch.matmul(attention_weights, v) 
        
        # output shape: (batch_size, num_heads, seq_len_q, head_dim)
        # out.transpose(1, 2) to (batch_size, seq_len_q, num_heads, head_dim)
        # batch_size * seq_len_q * num_heads * head_dim 
        # reshape output to (batch_size, seq_len_q, embed_dim)
        outcat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) 
        out = self.out_linear(outcat)
        return out, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """_summary_

        Args:
            d_model (_type_): _model dimension
            d_ff (_type_): _feed forward dimension
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # size of x is (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        """_summary_

        Args:
            d_model (_type_): _model dimension
            max_seq_len (_type_, optional): _maximum sequence length_. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        # create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # calculate the positional encoding
        # position is the index of the word in the sentence
        # unsqueeze(1) to convert to (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # div_term is the denominator of the sine and cosine functions
        # exp() is the exponential function, torch.arange(0, d_model, 2) is the index of the dimension
        # 2 * i is the index of the dimension
        # 10000 is the base of the exponential function
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # apply sine and cosine functions to the position
        # 0::2 means every second element, 1::2 means the rest
        pe[:, 0::2] = torch.sin(position * div_term) # sine
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension
        pe = pe.unsqueeze(0) 
        # register the buffer so that it is not a parameter
        # but still part of the model
        # it will be saved and loaded with the model
        # and will be moved to the GPU if the model is moved to the GPU
        # pe is a tensor of shape (1, max_seq_len, d_model)
        # pe is the positional encoding
        self.register_buffer('pe', pe)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        """_summary_

        Args:
            d_model (_type_): _model dimension
            num_head (_type_): _number of heads
            d_ff (_type_): _feed forward dimension
            dropout (float, optional): _dropout rate_. Defaults to 0.1.
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_head)
        self.ff = FeedForward(d_model, d_ff)
        # two layer normalization layers
        # one for the multi-head attention layer
        # one for the feed forward layer
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x is the input to the encoder layer
        # mask is the mask for the multi-head attention layer
        attn_output, _= self.mha(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        # apply the feed forward layer
        # x is the output of the multi-head attention layer
        # ff_output is the output of the feed forward layer
        ff_output = self.ff(x)
        x = self.layernorm2(x + ff_output)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        """_summary_

        Args:
            d_model (_type_): _model dimension
            num_head (_type_): _number of heads
            d_ff (_type_): _feed forward dimension
            dropout (float, optional): _dropout rate_. Defaults to 0.1.
        """
        super(DecoderLayer, self).__init__()
        # masked self attention, to prevent attending to future tokens
        self.mha = MultiHeadAttention(d_model, num_head)
        # encoder-decoder attention, to attend to the encoder output
        self.enc_dec_mha = MultiHeadAttention(d_model, num_head)
        self.ff = FeedForward(d_model, d_ff)
        # three layer normalization layers
        # one for the masked self attention layer
        # one for the encoder-decoder attention layer
        # one for the feed forward layer
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) 
     
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # x is the input to the decoder layer
        # enc_output is the output of the encoder layer
        # src_mask is the mask for the encoder-decoder attention layer
        # tgt_mask is the mask for the masked self attention layer
        attn_output, _ = self.mha(x, x, x, tgt_mask) 
        x = self.layernorm1(x + self.dropout1(attn_output))
        
        # apply the encoder-decoder attention layer
        enc_dec_attn_output, _ = self.enc_dec_mha(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + enc_dec_attn_output)
        
        # apply the feed forward layer
        ff_output = self.ff(x)
        x = self.layernorm3(x + ff_output)
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,  d_model=512, num_head=8, d_ff=2048, num_encoder_layers=6, num_decoder_layers=6, max_seq_len=5000, dropout=0.1):    
        """_summary_

        Args:
            d_model (_type_): _model dimension
            num_head (_type_): _number of heads
            d_ff (_type_): _feed forward dimension
            num_encoder_layers (_type_): _number of encoder layers
            num_decoder_layers (_type_): _number of decoder layers
            src_vocab_size (_type_): _input vocabulary size
            tgt_vocab_size (_type_): _target vocabulary size
            max_seq_len (int, optional): _maximum sequence length_. Defaults to 5000.
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # embedding layer for the input and target sequences
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # positional encoding layer for the input and target sequences
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # encoder layers, ModuleList is a list of nn.Module
        # it is used to store the encoder layers
        # each encoder layer is an instance of the EncoderLayer class
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_head, d_ff, dropout) for _ in range(num_encoder_layers)])
        
        # decoder layers
        # ModuleList is a list of nn.Module
        # it is used to store the decoder layers
        # each decoder layer is an instance of the DecoderLayer class
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout) for _ in range(num_decoder_layers)])
        
        # final linear layer to project the output to the target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_src_mask(self, src):
        # create a mask for the source sequence
        # src is the input sequence
        # src_mask is a tensor of shape (batch_size, 1, 1, seq_len)
        # it is used to mask the padding tokens in the input sequence
        # the padding tokens are the tokens with index 0
        # unsqueeze(1) adds a dimension to the tensor
        # unsqueeze(2) adds another dimension to the tensor
        # the final shape of src_mask is (batch_size, 1, 1, seq_len)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
        
        return src_mask
    
    def generate_tgt_mask(self, tgr):
        # create a mask for the target sequence
        batch_size, seq_len = tgr.size()
        
        # fill out the mask, (batch_size, 1, 1, seq_len)
        tgt_pad_mask = (tgr != 0).unsqueeze(1).unsqueeze(2)
        # create a subsequent mask for the target sequence
        # the subsequent mask is used to prevent attending to future tokens
        # torch.triu creates an upper triangular matrix
        # the diagonal is set to 1, the rest is set to 0
        tgt_sub_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgr.device), diagonal=1).bool()
        # tgt_sub_mask is a tensor of shape (seq_len, seq_len)
        # it is used to mask the subsequent tokens in the target sequence
        # tgt_sub_mask is used to prevent attending to future tokens
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1) # (1, 1, seq_len, seq_len)
        # tgt_mask is the final mask for the target sequence
        # it is the element-wise product of tgt_pad_mask and tgt_sub_mask
        tgt_mask = tgt_pad_mask & tgt_sub_mask # (batch_size, 1, seq_len, seq_len)
        return tgt_mask
    
    def encode(self, src, src_mask=None):
        # src_emb is the embedding of the input sequence
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        # positional encoding is added to the embedding
        src_emb = self.positional_encoding(src_emb) # (batch_size, seq_len, d_model)
        # apply dropout to the embedding
        enc_output = self.dropout(src_emb)
        for encoder_layer in self.encoder_layers:
            # apply the encoder layer to the input sequence
            src_emb = encoder_layer(enc_output, src_mask)
        
        return enc_output
    
    def decode(self, tgr, enc_output, src_mask=None, tgt_mask=None):
        # tgr_emb is the embedding of the target sequence
        tgr_emb = self.decoder_embedding(tgr) * math.sqrt(self.d_model)
        # positional encoding is added to the embedding
        tgr_emb = self.positional_encoding(tgr_emb)
        # apply dropout to the embedding    
        dec_output = self.dropout(tgr_emb)
        for decoder_layer in self.decoder_layers:
            # apply the decoder layer to the input sequence
            dec_output = decoder_layer(dec_output, enc_output, src_mask, tgt_mask)
        return dec_output
    
    def forward(self, src, tgr):
        # generate the source and target masks
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgr)
        
        # encode the input sequence
        enc_output = self.encode(src, src_mask)
        # decode the target sequence
        dec_output = self.decode(tgr, enc_output, src_mask, tgt_mask)
        # apply the final linear layer to the decoder output
        output = self.fc_out(dec_output)
        return output
        
# Example usage 
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_head = 8
    d_ff = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    max_seq_len = 5000
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_head, d_ff, num_encoder_layers, num_decoder_layers, max_seq_len)
    
    src = torch.randint(0, src_vocab_size, (32, 10)) # batch size of 32 and sequence length of 10
    tgr = torch.randint(0, tgt_vocab_size, (32, 10)) # batch size of 32 and sequence length of 10
    
    output = model(src, tgr)
    print(output.shape) # should be (32, 10, tgt_vocab_size)