'''
Author: goog
Date: 2021-12-20 22:19:00
LastEditTime: 2022-03-03 18:48:52
LastEditors: goog
Description: 
FilePath: /event/zlj_complex_event/EnventDetection/MMFSTSR_v12/AnpTextRnet.py
Time Limit Exceeded!
'''
import torch
import MSERN.dataset.LoadData as LoadData
import TextFeature
import ANPFeat
import torch.nn.functional as F

class RepresentationFusion(torch.nn.Module):
    def __init__(self, text_feature_hidden_dim, anp_feature_didden_dim):
        super(RepresentationFusion, self).__init__()
        self.linear1_1 = torch.nn.Linear(text_feature_hidden_dim+text_feature_hidden_dim, int((text_feature_hidden_dim+text_feature_hidden_dim)/2))
        self.linear2_1 = torch.nn.Linear(int((text_feature_hidden_dim+text_feature_hidden_dim)/2), 1)

        self.linear1_2 = torch.nn.Linear(text_feature_hidden_dim+anp_feature_didden_dim, int((text_feature_hidden_dim+anp_feature_didden_dim)/2))
        self.linear2_2 = torch.nn.Linear(int((text_feature_hidden_dim+anp_feature_didden_dim)/2), 1)

    '''
    description: 
    param {*} self
    param {*} text
    param {*} anp
    param {*} text_seq (seq_len, batchsize, hid_dim*direct)
    return {*}
    '''
    def forward(self, text, anp, text_seq):
        text_text_list=list()
        text_anp_list=list()
        length=text_seq.size(0) # seq_len
    
        for i in range(length):
            text_text = torch.tanh(self.linear1_1(torch.cat([text_seq[i], text], dim=1))) #[(batchszie, hid_dim*direct) (batchszie, hid_dim*direct)] -> [(batchsize, (2*hid_dim*direct)/2)]
            text_anp = torch.tanh(self.linear1_2(torch.cat([text_seq[i], anp], dim=1))) # [(batchszie, hid_dim*direct) (batchszie, hid_dim*direct)]
        
            text_text_list.append(self.linear2_1(text_text))
            text_anp_list.append(self.linear2_2(text_anp))
        weight_1 = torch.nn.functional.softmax(torch.stack(text_text_list), dim=0) # [seq_len, batchsize, 1] normalization attention
        weight_2 = torch.nn.functional.softmax(torch.stack(text_anp_list), dim=0) # [seq_len, batchsize, 1] 
        output = torch.mean((weight_1+weight_2)*text_seq/2, 0) # [seq_len, batchsize, 1]*(seq_len, batchsize, hid_dim*direct)
        # There are differences with the paper
        return output

class RNet(torch.nn.Module):
    def __init__(self, opt):
        super(RNet, self).__init__()
        text_feature_size=opt.TEXT_HIDDEN*2 #text_feature.size(1)
        anp_feature_size=opt.ANPFeatHidden*2 #anp_feature.size(1)
        # print(text_feature.size(1), anp_feature.size(1))
        # self.text_layerNorm = torch.nn.LayerNorm([text_feature_size])
        # self.anp_layerNorm = torch.nn.LayerNorm([anp_feature_size])
        self.text_attention = RepresentationFusion(text_feature_size, anp_feature_size)
        self.text_linear = torch.nn.Linear(512, 512)
    def forward(self, text_feature, text_seq, anp_feature, attribute_seq):
        # text_feature = self.text_layerNorm(text_feature)
        # anp_feature = self.anp_layerNorm(anp_feature)
                                             # [32, 512]      [32, 512]       [60, 32, 512]
        text_vector     =self.text_attention(text_feature, anp_feature, text_seq) # v_t

        # final fuse
        text_vector_N = F.normalize(text_vector)
        anp_feature_N = F.normalize(anp_feature)
        score = text_vector_N.T.mm(anp_feature_N)  # inner product 论文是 余弦相似度

        output = torch.relu(self.text_linear(text_vector.mm(score))) # S_J

        #output=torch.relu(self.text_linear(torch.cat([text_vector,anp_feature],dim=1)))
        return output
if __name__ == "__main__":
    text=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    anp=ANPFeat.ANPFeat()
    fuse=RNet()
    for vision_feature,text_index,anp_index,y_true,id in LoadData.train_loader:
        text_result,text_seq=text(text_index,None)
        anp_result,anp_seq=anp(anp_index)
        result=fuse(text_result,text_seq.permute(1,0,2),anp_result,anp_seq.permute(1,0,2))
        print(result.shape)

        break