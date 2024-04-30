'''
Author: goog
Date: 2021-12-20 22:19:00
LastEditTime: 2022-03-03 20:45:43
LastEditors: goog
Description: 
FilePath: /event/zlj_complex_event/EnventDetection/MMFSTSR_v12/Classifier.py
Time Limit Exceeded!
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ClassifierLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout_rate=0.5):
        super(ClassifierLayer, self).__init__()
        self.in_dim = in_dim  ## 1572 + 512
        self.out_dim = out_dim
        self.hidden = 512
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            # nn.BatchNorm1d(self.out_dim),
            # nn.Dropout(dropout_rate),
            nn.ReLU())

        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # stdv = 1. / math.sqrt(m.weight.size(1))
                # m.weight.data.uniform_(-stdv, stdv)
                torch.nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, vp, tp, at):
        VAT = torch.cat((torch.cat((vp, tp), dim=1), at), dim=1) # cat(s_P, tp)
        output = F.softmax(self.fc(VAT), dim=1)
        
        return output
