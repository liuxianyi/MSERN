'''
Author: goog
Date: 2021-12-20 22:19:00
LastEditTime: 2022-02-24 16:00:41
LastEditors: goog
Description: 
FilePath: /event/zlj_complex_event/EnventDetection/MMFSTSR_v12/TextFeature.py
Time Limit Exceeded!
'''
import torch
import numpy as np
import MSERN.dataset.LoadData as LoadData

class ExtractTextFeature(torch.nn.Module):
    def __init__(self, opt, dropout_rate=0.2):
        super(ExtractTextFeature, self).__init__()
        self.opt = opt
        text_length, hidden_size = opt.TEXT_LENGTH, opt.TEXT_HIDDEN
        self.hidden_size=hidden_size
        self.text_length=text_length
        embedding_weight=self.getEmbedding()
        self.embedding_size=embedding_weight.shape[1]
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
        # self.layerNorm = torch.nn.LayerNorm([60,200])
        self.biLSTM = torch.nn.LSTM(input_size=200, hidden_size=hidden_size, bidirectional=True, batch_first=True)

        # early fusion
        self.Linear_1=torch.nn.Linear(512,hidden_size)
        self.Linear_2=torch.nn.Linear(512,hidden_size)
        self.Linear_3=torch.nn.Linear(512,hidden_size)
        self.Linear_4=torch.nn.Linear(512,hidden_size)

        # dropout
        # self.dropout=torch.nn.Dropout(dropout_rate)

    '''
    description: 
    param {*} self
    param {*} input 文本固定60个词 (batchsize, seq_len)
    param {*} guidence
    return {*}
    '''
    def forward(self, input, guidence ):
        embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size) # (batchsize, seq_len, 200)
        # embedded = self.layerNorm(embedded)
        if(guidence is not None):
            # early fusion
            hidden_init=torch.stack([torch.relu(self.Linear_1(guidence)),torch.relu(self.Linear_2(guidence))],dim=0)
            cell_init=torch.stack([torch.relu(self.Linear_3(guidence)),torch.relu(self.Linear_4(guidence))],dim=0)
            output,_=self.biLSTM(embedded, (hidden_init, cell_init))
        else:
            output,_=self.biLSTM(embedded,None) # (batchsize, seq_len, hid_dim*num_directions)

        # dropout
        # output=self.dropout(output)

        RNN_state=torch.mean(output,1) # (batchsize, hid_dim*num_directions)
        return RNN_state, output

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt(self.opt.TextEmbbeding, delimiter=' ', dtype='float32'))


if __name__ == "__main__":
    test=ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    for text_index,group,id in LoadData.train_loader:
        result,seq=test(text_index,None)
        # [2, 512]
        print(result)
        print(result.shape)
        # [2, 75, 512]
        print(seq.shape)
        break
