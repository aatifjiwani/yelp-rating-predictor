import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn



class SkipGram(torch.nn.Module):
    def __init__(self, embedding_dim: int = 512, vocabulary_size: int = 50002):

        super(SkipGram, self).__init__()

        self.embed_dim = embedding_dim
        self.vocab_size = vocabulary_size
        self.encoder = nn.Linear(self.vocab_size, self.embed_dim, bias=False)
        self.decoder = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, word_pairs):
        # word pairs of shape (B, Vocab)
        encoder_feat = self.encoder(word_pairs) #B, Emb
        decoder_feat = self.decoder(encoder_feat) #B, Vocab

        return decoder_feat

    