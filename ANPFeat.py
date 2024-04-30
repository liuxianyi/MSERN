'''
Author: goog
Date: 2021-12-20 22:19:00
LastEditTime: 2022-02-24 15:30:50
LastEditors: goog
Description: 
FilePath: /event/zlj_complex_event/EnventDetection/MMFSTSR_v12/ANPFeat.py
Time Limit Exceeded!
'''
import torch
import numpy as np

class ANPFeatExtract(torch.nn.Module):
    def __init__(self, opt, hidden_size=256):
        super(ANPFeatExtract, self).__init__()
        self.hidden_size = hidden_size
        self.opt = opt
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        self.biLSTM = torch.nn.LSTM(input_size=400, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    '''
    description: 
    param {*} self
    param {*} input ANP (batchsize, seq_len, 2)
    return {*}
    '''
    def forward(self, input):
        embedded = self.embedding(input).view(-1, 20, 2 * self.embedding_size)  
        output, _ = self.biLSTM(embedded, None) 
        RNN_state = torch.mean(output, 1)
        return RNN_state, output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt(self.opt.TextEmbbeding, delimiter=' ', dtype='float32'))

