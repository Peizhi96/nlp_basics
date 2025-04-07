import torch
import torch.nn 
import torch.nn.functional as F 

class MultiHeadAttention(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads

		assert self.head_dim * num_heads == embed_dim

		self.q_linear = nn.Linear(embed_dim, embed_dim)
		self.k_linear = nn.Linear(embed_dim, embed_dim)
		self.v_linear = nn.Linear(embed_dim, embed_dim)
		self.out_linear = nn.Linear(embed_dim, embed_dim)

	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0) # get the size of batch_size, (batch_size, seq_len, embed_dim)

		# linear transformation
		q = self.q_linear(query) # (batch_size, seq_len, embed_dim)
		k = self.k_linear(key) # (batch_size, seq_len, embed_dim)
		v = self.v_linear(value) # (batch_size, seq_len, embed_dim)

		# reshape and transpose, (batch_size, num_heads, seq_len, head_dim)
		q = q.view(batch_size, -1, self.num_heads, self.head_dim).tranpose(1, 2) 
		k = k.view(batch_size, -1, self.num_heads, self.head_dim).tranpose(1, 2)
		v = v.view(batch_size, -1, self.num_heads, self.head_dim).tranpose(1, 2) 
  
		# calculate attention scores (batch_size, num_heads, seq_len, head_dim)
		# k.tranpose(-2, -1) (batch_size, num_heads, head_dim, seq_len)
		scores = torch.matmul(q, k.tranpose(-2, -1)) # (batch_size, num_heads, seq_len_q, seq_len_k)

		# scale scores
		scores = scores / (self.head_dim ** 0.5)

		# apply mask if provided
		if mask is not None:
			scores = scores.masked_fill(mask==0, 1e-9)

		# apply softmax to get attention weights
		attention_weight = F.softmax(scores, dim=-1)

		# get weighted sum of values
		output = torch.matmul(attention_weight, v)

		# output shape: (batch_size, num_heads, seq_len_q, head_dim)
		# output.tranpose(1, 2) to (batch_size, seq_len_q, num_heads, head_dim)
		# reshape output to (batch_size, seq_len, embed_dim)
		outcat = output.tranpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
		out = self.out_linear(outcat)
		return output, attention_weight


import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout=0.1):
		self.heads = heads
		self.d_model = d_model
		self.d_k = d_model // heads
		self.q_linear = nn.Linear(d_model, d_model) # batch_size, seq_len, d_model
		self.k_linear = nn.Linear(d_model, d_model) # batch_size, seq_len, d_model
		self.v_linear = nn.Linear(d_model, d_model) # batch_size, seq_len, d_model
		self.dropout = dropout
		self.out = nn.Linear(d_model, d_model)


	def attention(self, q, k, v, d_k, mask=None, dropout=None):
		scores = torch.matmul(q, k.tranpose(-2, -1)) / math.sqrt(d_k)
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)
		scores = F.softmax(scores, dim=-1)
		if dropout is not None:
			scores = dropout(scores)
		output = torch.matmul(scores, v)
		return output 

	def attention(self, q, k, v, d_k, mask=None, dropout=None):
		scores = torch.matmul(q, k.tranpose(-2, -1)) / math.sqrt(d_k)
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)
		scores = F.softmax(scores, dim=-1)
		if dropout is not None:
			scores = dropout(scores)
		output = torch.matmul(scores, v)
		return output

	def forward(self, q, k, v, mask=None):
		bs = q.size(0) #获取batch_size的大小

		"""
		对Q, K, V分别进行线性变换
		输入q, k, v的维度位(batch_size, seq_len, d_model)
		"""
		k = self.k_linear(k) #线性变换后的k
		q = self.q_linear(q) #线性变换后的q
		v = self.v_linear(v) #线性变换后的v

		"""
		维度变换,将d_model拆分为多个头(heads)
		将k, q, v reshape为(batch_size, seq_len, heads, d_k)
		然后再交换维度, 转换成(batch_size, heads, seq_len, d_k)
		"""
		k = k.view(bs, -1, self.heads, self.d_k).tranpose(1, 2)
		q = q.view(bs, -1, self.heads, self.d_k).tranpose(1, 2)
		v = v.view(bs, -1, self.heads, self.d_k).tranpose(1, 2)

		#计算注意力，返回的scores维度为（batch_size, heads, seq_len, d_k)
		scores = self.attantion(q, k, v, self.d_k, mask, self.dropout)

		#把多头的输出拼起来
		concat = scores.tranpose(1, 2).contiguous().view(bs, -1, self.d_model)

		#通过输出线性层，再次映射回 （batch_size, seq_len, d_model)
		output = self.out(concat)

		return output





















































