import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TorchTransformer(nn.Module):

    def __init__(self, vocab, model_dim, ff_dim, num_heads, num_layers, num_classes, max_len=1000, dropout=0.45, ds_init=0.9, cls_token=False):
        
        super(TorchTransformer, self).__init__()

        self.vocab = vocab
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.ds_init = ds_init
        self.cls_token = cls_token

        self.input_embedding = nn.Embedding(vocab, model_dim)
        self.pos_encoder = PositionalEncoder(model_dim, max_len=max_len)

        transformer_encoder_layer = nn.TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self.decoder = nn.Linear(model_dim, num_classes)

        self.curr_mask = None

    # https://www.aclweb.org/anthology/D19-1083.pdf
    def init_weights(self):
        
        gamma_constant = (self.ds_init / math.sqrt(self.num_layers))

        #input embedding weights
        d_i = self.vocab
        d_o = self.model_dim
        gamma_range = math.sqrt( 6 / (d_i + d_o) ) * gamma_constant

        self.input_embedding.weight.data.uniform_(-gamma_range, gamma_range)

        #decoder weights
        d_i = self.model_dim
        d_o = self.num_classes
        gamma_range = math.sqrt( 6 / (d_i + d_o) ) * gamma_constant

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-gamma_range, gamma_range)

    def generateMask(self, size):
        mask = (torch.triu (torch.ones(size, size)) == 1).transpose(0, 1) 
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, inputs):
        #inputs of shape B, Seq
        inputs = self.input_embedding(inputs) * math.sqrt(self.model_dim) #B, S, Model_Dim
        inputs.permute(1, 0, 2) #S, B, Model_Dim

        if self.curr_mask is None or self.curr_mask.shape[0] != inputs.shape[0]:
            curr_device = inputs.device
            self.curr_mask = self.generateMask(len(inputs)).to(curr_device)
        
        inputs = self.pos_encoder(inputs)

        outputs = self.transformer_encoder(inputs, self.curr_mask)

        #outputs of size B, S, Hidden_Dim (#CLS IS AT SEQUENCE 0 if using CLS)
        if self.cls_token:
            outputs = outputs[:, 0, :] #(B, H)
        else:
            outputs = torch.mean(outputs, dim=1) #B, Hidden_Dim
        # 

        outputs = self.decoder(outputs)

        return outputs

class PositionalEncoder(nn.Module):

    def __init__(self, model_dim, dropout=0.1, max_len=1000):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        positional_encoder = torch.zeros(max_len, model_dim)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        denom = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))

        positional_encoder[:, 0::2] = torch.sin(positions * denom)
        positional_encoder[:, 1::2] = torch.cos(positions * denom)

        positional_encoder = positional_encoder.unsqueeze(0).transpose(0, 1)

        self.register_buffer('positional_encoder', positional_encoder)

    def forward(self, x):
        x = x + self.positional_encoder[:x.size(0), :]
        return self.dropout(x)


