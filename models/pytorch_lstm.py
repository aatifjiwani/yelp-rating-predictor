import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchBiLSTM(nn.Module):

    def __init__(self, embedding_matrix, hidden_size=128, dropout=0, batch_first=False, num_classes=5):
        super(TorchBiLSTM, self).__init__()

        self.num_classes = num_classes

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False

        self.hidden_size = hidden_size
        self.biLSTM = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=hidden_size, \
                            batch_first=batch_first, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        self.prediction = nn.Sequential(
                            nn.Linear( in_features=hidden_size*2, out_features=num_classes, bias=False),
                            nn.Softmax( dim=1 )
                        )

    def forward(self, reviews):
        assert len(reviews.shape) == 2, "Review must be of shape Batch x Seq_len"
        batch_size = reviews.shape[0]
        seq_len = reviews.shape[1]

        reviews = self.embedding(reviews)

        #reviews of shape (Batch, SeqLen, embedding dim)
        reviews = reviews.permute(1,0,2)

        output, (h_n, c_n) = self.biLSTM(reviews)

        #h_n of shape (2, Batch, Hidden_Dim)
        h_n = self.dropout(h_n.permute(1, 0, 2).reshape(batch_size, 2*self.hidden_size))

        logits = self.prediction(h_n)

        return logits

        
