import torch
import torch.nn as nn
from utils import get_num_output, get_activation, get_num_linear
import math

class TransEncoder(nn.Module):
    def __init__(self, inc=512, dropout=0.6, dim_feedforward=1024, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.d_model = inc
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayer)

        # add positional encoding
        self.max_seq_len = 1000  # 设置最大序列长度
        self.pos_encoding = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        # add positional encoding
        x = x + self.pos_encoding[:x.size(1), :].unsqueeze(0).to(x.device)

        out = self.transformer_encoder(x)
        return out

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.modal_num = len(args.feat_dim)
        self.num_features = args.feat_dim
        affine_dim = args.d_affine_dim * self.modal_num
        self.dim_feedforward = affine_dim * 2
        self.input = nn.ModuleList()
        for i in range(self.modal_num):
            self.input.append(nn.Linear(self.num_features[i], args.d_affine_dim))
        self.dropout_embed = nn.Dropout(p=args.dropout_embed)

        # affine_dim = min(args.d_in // 2, args.d_affine_dim)
        hidden_dim = affine_dim // 2
        # self.feat_fc = nn.Linear(args.d_in, affine_dim)
        self.encoder = TransEncoder(inc=affine_dim, dropout=args.dropout, dim_feedforward=self.dim_feedforward, nheads=args.t_nheads, nlayer=args.t_nlayer)
        num_output = get_num_output(args.task)
        num_linear = get_num_linear(args.task)
        self.output_list = nn.ModuleList()
        for i in range(num_linear):
            output = nn.Sequential(
                nn.Linear(affine_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, num_output))
            self.output_list.append(output)

        self.final_activation = get_activation(args.task)
    def forward(self, x):
        # feat = self.feat_fc(x)

        x = torch.split(x, self.num_features, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.input[i](x[i]))
        x = torch.cat(_x_list, dim=-1)
        x = self.dropout_embed(x)

        out = self.encoder(x)
        bs, sl = out.shape[0], out.shape[1]
        out = out.reshape(bs*sl,-1)
        outs = [y(out) for y in self.output_list]
        outs = torch.cat(outs, dim=1)
        outs = outs.reshape(bs,sl,-1)
        outs = self.final_activation(outs)
        return outs