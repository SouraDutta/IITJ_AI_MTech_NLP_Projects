import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Rel_based_labeling(nn.Module):

    def __init__(self, opt):
        super(Rel_based_labeling, self).__init__()
        self.opt = opt
        self.embedding = Embedding(opt)
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, sent, sen_len, rel, mask, poses=None, chars=None):
        # print(rel)
        embedding = self.embedding(sent, poses, chars)
        sen_embedding = self.encoder(embedding, sen_len)
        return self.decoder(sen_embedding, rel, mask, sen_len)

class charEmbedding(nn.Module):
    '''
    Input: (max_len, max_word_len) max_len是指句子的最大长度，也就是char的batch size
    Output: (max_len, filter_num)
    '''
    def __init__(self, opt):
        super(charEmbedding, self).__init__()
        self.opt = opt
        self.emb = nn.Embedding(self.opt.char_vocab_size, self.opt.char_embedding_size)
        self.conv = weight_norm(nn.Conv1d(in_channels=self.opt.char_embedding_size, out_channels = self.opt.filter_number, kernel_size=self.opt.kernel_size))
        self.pool = torch.nn.MaxPool1d(self.opt.max_word_len - self.opt.kernel_size + 1, stride=1)
        self.drop = torch.nn.Dropout(opt.dropout_rate)
        self.init()
    def init(self):
        nn.init.kaiming_uniform_(self.emb.weight.data)
        nn.init.kaiming_uniform_(self.conv.weight.data)
    def forward(self, x):
        '''
            x: one char sequence. shape: (max_len, max_word_len)
        '''
        # 如果输入是一句话
        inp = self.drop(self.emb(x))  # (max_len, max_word_len) -> (max_len, max_word_len, hidden)
        inp = inp.permute(0,2,1) # (max_len, max_word_len, hidden) -> (max_len,  hidden, max_word_len)
        out = self.conv(inp) # out: (max_len, filter_num, max_word_len - kernel_size + 1)
        return self.pool(out).squeeze() # out: (max_len, filter_num)

class Embedding(nn.Module):
    def __init__(self, opt):
        super(Embedding, self).__init__()
        self.opt = opt
        self.word_embedding = nn.Embedding(self.opt.vocab_size, self.opt.word_embedding_size)
        self.relu = nn.ReLU()
        if self.opt.use_pos:
            self.pos_embedding = nn.Embedding(self.opt.pos_vocab_size, self.opt.pos_embedding_size, padding_idx=0)
        if self.opt.use_char:
            self.char_encode = charEmbedding(opt)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.word_embedding.weight.data)
        unk = torch.Tensor(1, self.opt.word_embedding_size).uniform_(-1, 1)
        pad = torch.zeros(1, self.opt.word_embedding_size)
        self.word_embedding.weight.data = torch.cat([pad, unk, self.word_embedding.weight.data])
        if self.opt.use_pos:
            nn.init.kaiming_uniform_(self.pos_embedding.weight.data)

    def forward(self, x, pos=None, char=None):
        '''
            x,pos : (batch, max_len)
            char : (batch, max_len, max_word_len)
        '''
        word_emb = self.dropout(self.word_embedding(x))
        if char is not None:
            char_embedding = []
            for i in range(char.shape[0]):
                one_word_char_emb = self.char_encode(char[i])
                char_embedding.append(one_word_char_emb)
            char_embedding = torch.stack(char_embedding)
            word_emb = torch.cat((word_emb, char_embedding), -1)
        if pos is not None:
            pos_emb = self.dropout(self.pos_embedding(pos))
            word_emb = torch.cat((word_emb, pos_emb), -1)
        return word_emb


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        if self.opt.use_pos and self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size + self.opt.filter_number
        elif self.opt.use_pos:
            self.in_width = self.opt.word_embedding_size + self.opt.pos_embedding_size
        elif self.opt.use_char:
            self.in_width = self.opt.word_embedding_size + self.opt.filter_number
        else:
            self.in_width = self.opt.word_embedding_size
        self.birnn = nn.GRU(self.in_width, self.opt.rnn_hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)

    def forward(self, x, sen_len):
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = x[perm_idx]
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.birnn(packed_input)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        output = torch.index_select(output, 0, un_idx)
        output = torch.relu(self.linear(output))

        return output


class AttentionDot(nn.Module):
    def __init__(self, opt):
        super(AttentionDot, self).__init__()
        self.opt = opt
        self.rel2att = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.rel2att.weight.data)

    def forward(self, sent, rel, mask):
        # sent: batch, max_len, hidden
        # rel: batch, rel_dim -> relation: batch, hidden
        relation = self.rel2att(rel).unsqueeze(1)
        # batch, max_len
        weight = torch.matmul(relation, sent.transpose(-1,-2)).squeeze()
        weight = weight * mask.float()
        weight = torch.softmax(weight, -1)
        att_res = torch.bmm(weight.unsqueeze(1), sent).squeeze(1) # batch_size * att_hidden_size
        return att_res, weight

class AttentionNet(nn.Module):
    def __init__(self, opt):
        super(AttentionNet, self).__init__()
        self.opt = opt
        self.Wg = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wh = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.Wr = nn.Linear(self.opt.rel_dim, self.opt.att_hidden_size)
        self.alpha_net = nn.Linear(self.opt.att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wg.weight.data)
        nn.init.xavier_uniform_(self.Wh.weight.data)
        nn.init.xavier_uniform_(self.Wr.weight.data)
        nn.init.xavier_uniform_(self.alpha_net.weight.data)

    def forward(self, sent_h, rel, pool, mask):
        relation = self.Wr(rel)
        sent = self.Wh(sent_h)
        global_sen = self.Wg(pool)

        relation = relation.unsqueeze(1).expand_as(sent)
        global_sen = global_sen.unsqueeze(1).expand_as(sent)

        mix = torch.tanh(relation + sent + global_sen)
        weight = self.alpha_net(mix).squeeze()
        weight.masked_fill_(mask==0, -1e9)
        weight_ = torch.softmax(weight, -1)

        #weight = weight * mask.float()
        att_res = torch.bmm(weight_.unsqueeze(1), sent).squeeze(1) # batch_size * att_hidden_size
        return att_res, weight_

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.relation_matrix = nn.Embedding(self.opt.rel_num, self.opt.rel_dim)

        self.attention = AttentionNet(opt)
        self.W = nn.Linear(self.opt.att_hidden_size*3, self.opt.att_hidden_size)
        self.dropout = nn.Dropout(self.opt.dropout_rate)
        self.bilstm = nn.LSTM(self.opt.att_hidden_size, self.opt.rnn_hidden_size, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.opt.rnn_hidden_size*2, self.opt.label_num)

        self.W1 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W2 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size)
        self.W3 = nn.Linear(self.opt.att_hidden_size, self.opt.att_hidden_size*2)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.xavier_uniform_(self.hidden2tag.weight.data)
        #nn.init.kaiming_uniform_(self.W.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.W.weight.data)
        nn.init.xavier_uniform_(self.W1.weight.data)
        nn.init.xavier_uniform_(self.W2.weight.data)
        nn.init.xavier_uniform_(self.W3.weight.data)

    def masked_mean(self, sent, mask):
        mask_ = mask.masked_fill(mask==0, -1e9)
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def nn_decode(self, inputs, sen_len):
        sort_len, perm_idx = sen_len.sort(0, descending=True)
        _, un_idx = torch.sort(perm_idx, dim=0)
        x_input = inputs[perm_idx]
        packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)

        packed_out, _ = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.opt.max_len)
        output = torch.index_select(output, 0, un_idx)
        return output

    def forward(self, sent, rel, mask, sen_len):
        rel_embedding = self.relation_matrix(rel)
        global_sen = self.masked_mean(sent, mask)
        sent_att, weight = self.attention(sent, rel_embedding, global_sen, mask)

        concats = torch.cat([self.W1(global_sen), self.W2(sent_att)], -1)
        alpha = torch.sigmoid(concats)
        gate = alpha * torch.tanh(self.W3(sent_att))
        decode_input = torch.cat([sent, gate.unsqueeze(1).expand(sent.shape[0], sent.shape[1], -1)], -1)
        decode_input = self.W(decode_input)
        #decode_input = sent + (alpha * (sum_sen_rel)).unsqueeze(1).expand_as(sent)
        decode_out = self.nn_decode(decode_input, sen_len)
        project = self.hidden2tag(decode_out)
        return project, weight
