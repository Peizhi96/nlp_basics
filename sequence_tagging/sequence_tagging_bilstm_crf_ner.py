import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# pad_sequence is used to pad sequences to the same length
# pack_padded_sequence is used to pack padded sequences for LSTM
# pad_packed_sequence is used to unpack packed sequences
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        
        #embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        #bilstm layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        #linear layer, this layer is used to map the output of the LSTM to the tag space
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        # nn.Parameter to register a parameter can be optimized in a model
        # it can be updated during training
        # torhc.randn is used to generate random numbers from a normal distribution, size: tag_size x tag_size
        # build a transition matrix is to store the transition scores between tags
        self.transitions = nn.Parameter(torch.randn(tag_size, tag_size))
        
        #initialize transition parameters
        self.transitions.data[tag_size-1, :] = -10000.0 #start with random values
        self.transitions.data[:, 0] = -10000.0 #from any tag to PAD
    
    def _get_lstm_features(self, sentences, lengths):
        """
        embeds is the output of the embedding layer
        packed is the output of the packed sequence
        lstm_out is the output of the LSTM layer
        pad_packed_sequence is used to unpack the packed sequence
        lstm_feats is the output of the linear layer
        """
        embeds = self.word_embeds(sentences)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # this function is used to calculate the score of a given sequence
        # feats is the output of the linear layer, tags is the output of the CRF layer
        
        # torch.zeros(1) is used to create a tensor of size 1 with all elements set to 0
        scores = torch.zeros(1, device=feats.device)
        # the first tag is the start tag, so we need to add the transition score from start tag to the first tag
        tags = torch.cat([torch.tensor([self.tag_size-1], device=feats.device).long(), tags])
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i], tags[i+1]] + feat[tags[i+1]]
        return scores
    
    def _viterbi_decode(self, feats):
        # this function is used to decode the best path using Viterbi algorithm
        # feats is the output of the linear layer
        # backpointers is used to store the best path
        # init_vvars is used to initialize the forward variable
        backpointers = []
        init_vvars = torch.full((1, self.tag_size), -10000.0, device=feats.device)
        init_vvars[0][self.tag_size-1] = 0
        
        
        # init_vvars is used to initialize the forward variable, it contains the score of the start tag
        forward_var = init_vvars
        # feats is the output of the linear layer, it contains the score of each tag
        for feat in feats:
            # bptrs_t is used to store the best path, means the best tag for each time step
            bptrs_t = []
            # viterbivars_t is used to store the best score for each tag
            viterbivars_t = []
            
            # iterate through all the tags, calculate the score of each tag and store the best tag
            for next_tag in range(self.tag_size):
                #next_tag_var is to store the total score of the maximum score of the previous tag add the transition score
                next_tag_var = forward_var + self.transitions[:, next_tag]
                # argmax is used to get the index of the maximum value
                best_tag_id = next_tag_var.argmax().item()
                # bptrs_t is used to store the best tag for each time step
                bptrs_t.append(best_tag_id)
                # viterbivars_t is used to store the best score for each tag
                # view(1) is used to reshape the tensor to (1, tag_size)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            # forward_var is used to store the best score for each tag
            # cat is used to concatenate the best score for each tag
            # view(1, -1) is used to reshape the tensor to (1, tag_size)
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # append the best tag for each time step to the backpointers
            # backpointers is used to store the best path
            backpointers.append(bptrs_t)
        
        #terminal_var is used to store the best score for each tag
        # transitions[:, 0] is used to get the transition score from the last tag to the start tag
        terminal_var = forward_var + self.transitions[:, 0]
        # argmax is used to get the index of the maximum value
        best_tag_id = terminal_var.argmax().item()
        # terminal_var[0] means the best score for each tag
        path_score = terminal_var[0][best_tag_id]
        
        # best_path is initialized with the best tag id
        best_path = [best_tag_id]
        # reverse(backpointers) is used to reverse the backpointers
        # this is to simulate the backtracking process
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        # best_path[::-1] is used to reverse the best path
        best_path = best_path[::-1]
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags, lengths):
        # this function is used to calculate the negative log likelihood
        feats = self._get_lstm_features(sentence, lengths)
        batch_size = feats.size(0)
        loss = torch.tensor(0., device=feats.device)
        
        for i in range(batch_size):
            feat = feats[i, :lengths[i]]
            tag = tags[i, :lengths[i]]
            # 计算所有可能路径的分数
            forward_score = self._forward_alg(feat)
            # 计算正确路径的分数
            gold_score = self._score_sentence(feat, tag)
            # 负对数似然 = 所有路径分数 - 正确路径分数
            loss += forward_score - gold_score
        
        return loss / batch_size
    
    def _forward_alg(self, feats):
        # this function is used to calculate the forward algorithm
        init_alphas = torch.full((1, self.tag_size), -10000., device=feats.device)
        init_alphas[0][self.tag_size-1] = 0.
        
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_size)
                trans_score = self.transitions[:, next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        
        terminal_var = forward_var + self.transitions[:, 0]
        alpha = self._log_sum_exp(terminal_var)
        return alpha
    
    def _log_sum_exp(self, vec):
        # 数值稳定的log sum exp实现
        max_score = vec.max()
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))
    
    def forward(self, sentence, lengths):
        # 获取BiLSTM特征
        lstm_feats = self._get_lstm_features(sentence, lengths)
        # 找出最佳路径
        batch_size = lstm_feats.size(0)
        tag_seqs = []
        
        for i in range(batch_size):
            feat = lstm_feats[i, :lengths[i]]
            _, tag_seq = self._viterbi_decode(feat)
            tag_seqs.append(tag_seq)
            
        return tag_seqs

def bilstm_crf_ner(train_data, train_labels, test_data):
    # 构建词汇表和标签集
    words = set()
    tags = set()
    for sentence in train_data:
        for word in sentence:
            words.add(word)
    for label_sequence in train_labels:
        for tag in label_sequence:
            tags.add(tag)
    
    # 词到索引的映射
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    
    # 标签到索引的映射
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    tag2idx["START"] = len(tag2idx)  # 开始标签
    
    # 转换数据为索引
    X_train = [[word2idx[w] for w in s] for s in train_data]
    y_train = [[tag2idx[t] for t in s] for s in train_labels]
    
    # 计算每个序列的长度
    lengths_train = [len(s) for s in X_train]
    
    # 转换为PyTorch张量
    def to_tensor(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = pad_sequence([torch.LongTensor(seq) for seq in sequences], batch_first=True, padding_value=0)
        return padded_seqs, lengths
    
    X_tensor, X_lengths = to_tensor(X_train)
    y_tensor, y_lengths = to_tensor(y_train)
    
    # 创建模型
    model = BiLSTM_CRF(len(word2idx), len(tag2idx)+1)
    optimizer = optim.Adam(model.parameters())
    
    # 训练模型
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(X_tensor, y_tensor, X_lengths)
        loss.backward()
        optimizer.step()
    
    # 测试
    X_test = [[word2idx.get(w, 1) for w in s] for s in test_data]
    X_test_tensor, X_test_lengths = to_tensor(X_test)
    
    # 预测
    model.eval()
    with torch.no_grad():
        tag_seqs = model(X_test_tensor, X_test_lengths)
    
    # 将索引转换回标签
    idx2tag = {i: t for t, i in tag2idx.items()}
    predictions = []
    for i, seq in enumerate(tag_seqs):
        pred_tags = [idx2tag.get(idx, "O") for idx in seq[:X_test_lengths[i]]]
        predictions.append(pred_tags)
    
    return predictions

# 测试代码
train_data = [["This", "is", "a", "test"]]
train_labels = [["O", "O", "O", "O"]]
test_data = [["This", "is", "another", "test"]]
print(bilstm_crf_ner(train_data, train_labels, test_data))
