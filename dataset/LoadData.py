import torch, re
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import copy
import pickle
from nltk import word_tokenize


TEXT_LENGTH=60

class my_data_set(Dataset):
    def __init__(self, opt, train=True):
        name = "train" if train else "test"
        self.opt = opt
        self.ANP = np.load(opt.ANP.format(name), allow_pickle=True)
        self.texts_np = np.load(opt.Text.format(name), allow_pickle=True).item()
        name_list = np.load(opt.video_list.format(name), allow_pickle=True)
        self.Vision = np.load(opt.Vision.format(name))
        self.Y = np.load(opt.Y.format(name))
        self.length = self.Y.shape[0]

        video_cls = defaultdict(list)
        for idx, v_name in enumerate(name_list):
            v_name = re.findall(r'(v_[a-zA-Z]+_.*_.*)_', v_name)[0]
            video_cls[v_name].append(idx)
        self.video_list = video_cls.keys()


        self.word2index = self.load_word_index()

    def load_word_index(self):
        word2index = pickle.load(open(self.opt.WordIndex, 'rb'), encoding='latin1')
        return word2index
    def __anp_loader(self, ind):
        anp_text=[]
        anp_str = self.ANP[ind].split(":")[-1].strip()
        for words in anp_str.split():
            anp_text.append(words.split('_')[0])  # adj.
            anp_text.append(words.split('_')[1])  # n.
        anp_index = np.empty(40, dtype=np.longlong)
        for i in range(40):
            if anp_text[i] in self.word2index:
                anp_index[i] = self.word2index[anp_text[i]]
            else:
                anp_index[i] = self.word2index["<unk>"]    
        return copy.copy(anp_index.reshape([20,2]))

    def __text_loader(self, ind):
        texts_np_copy = {}
        for k, v in self.texts_np.items():
            v_name = re.findall(r'.*/(v_[a-zA-Z]+_.*_.*)_', k)[0]
            texts_np_copy[v_name] = v
        text=word_tokenize(texts_np_copy[self.video_list[ind]])
        text_index=torch.empty(TEXT_LENGTH,dtype=torch.long)
        curr_length=len(text)
        for i in range(TEXT_LENGTH):
            if i>=curr_length:
                text_index[i]=self.word2index["<pad>"]
            elif text[i] in self.word2index:
                text_index[i]=self.word2index[text[i]]
            else:
                text_index[i]=self.word2index["<unk>"]

    def __getitem__(self, index): 
        y = self.Y[index, :]
        anp = self.__anp_loader(index)
        vision = self.Vision[index, :]
        text = self.__text_loader(index)
        return torch.from_numpy(vision), torch.from_numpy(text), torch.from_numpy(anp), torch.from_numpy(y)

    def __len__(self):
        return self.length



# import torch
# from torch.utils.data import Dataset, random_split
# import numpy as np
# import os
# import pickle
# from nltk import word_tokenize


# TEXT_LENGTH=60

# """
# read text file, find corresponding image path
# """
# def load_data(opt):
#     data_set=dict()
#     file = open(opt.TextData,'r')
#     for line in file:
#         content = eval(line)
#         group = content[2]
#         image = str(group)+'_'+str(content[0])  ##视频号
#         sentence = content[1]
#         if os.path.isfile(os.path.join(opt.VisionFeature, image+'.npy')):
#             data_set[image] = {"text": sentence, "group": int(group)}
#     return data_set





# class my_data_set(Dataset):
#     def __init__(self, opt, data):
#         self.data=data
#         self.opt = opt
#         self.image_ids=list(data.keys())
#         for id in data.keys():
#             self.data[id]["image_path"] = os.path.join(opt.VisionFeature, str(id)+".npy")

#         self.word2index = self.load_word_index()
#         self.anp = self.load_anp()
        
#         # load all text
#         for id in data.keys():
#             text=word_tokenize(self.data[id]["text"])
#             text_index=torch.empty(TEXT_LENGTH,dtype=torch.long)
#             curr_length=len(text)
#             for i in range(TEXT_LENGTH):
#                 if i>=curr_length:
#                     text_index[i]=self.word2index["<pad>"]
#                 elif text[i] in self.word2index:
#                     text_index[i]=self.word2index[text[i]]
#                 else:
#                     text_index[i]=self.word2index["<unk>"]
#             self.data[id]["text_index"] = text_index

#     # load anp
#     def load_anp(self):
#         # get anp
#         anp = dict()
#         with open(self.opt.ANP, 'r') as fl:
#             for line in fl:
#                 content = eval(line)
#                 anp[content[0]] = content[1:]
#         return anp

#     # load word index
#     def load_word_index(self):
#         word2index=pickle.load(open(self.opt.WordIndex, 'rb'), encoding='latin1')
#         return word2index

#     # load image feature data - resnet 50 result
#     def __vision_feature_loader(self,id):
#         vision_feature = np.load(os.path.join(self.opt.VisionFeature, id+".npy"))
#         return torch.from_numpy(vision_feature)

#     def group_label(self, id):
#         y_true=np.zeros(20).reshape([1,20])
#         # id = 1 baseball +
#         # id = 2 basketball +
#         if id == 5: # camping +
#             id=3
#         elif id == 7: # concert +
#             id=4 
#         elif id == 8: # cooking +
#             id=5
#         elif id == 9: # cycling, biking +
#             id=6
#         elif id == 10: # dance +
#             id=7
#         elif id == 11: # diving +
#             id=8 
#         elif id == 12: # drawing +
#             id=9
#         elif id == 13: # driving +
#             id=10
#         elif id == 14: # football +
#             id=11
#         elif id == 15: # golf +
#             id=12
#         elif id == 18: # parade +
#             id=13
#         elif id == 19: # party +
#             id=14 
#         elif id == 21: # runing +
#             id=15
#         elif id == 23: # singing +
#             id=16 
#         elif id == 24: # ski +
#             id=17
#         elif id == 26: # surfing +
#             id=18
#         elif id == 27: # swim-graduation
#             id=19
#         elif id == 28: # wedding +
#             id=20

#         y_true[0,int(id)-1]=1
#         return torch.from_numpy(y_true)

#     def group_label_ap(self,id):
#         if id == 5:
#             id=3
#         elif id == 7:
#             id=4
#         elif id == 8:
#             id=5
#         elif id == 9:
#             id=6
#         elif id == 10:
#             id=7
#         elif id == 11:
#             id=8
#         elif id == 12:
#             id=9
#         elif id == 13:
#             id=10
#         elif id == 14:
#             id=11
#         elif id == 15:
#             id=12
#         elif id == 18:
#             id=13
#         elif id == 19:
#             id=14
#         elif id == 21:
#             id=15
#         elif id == 23:
#             id=16
#         elif id == 24:
#             id=17
#         elif id == 26:
#             id=18
#         elif id == 27:
#             id=19
#         elif id == 28:
#             id=20

#         y_true=int(id)-1
#         return y_true


#     # load attribute feature data - 5 words label
#     def __anp_loader(self, id):
#         anp_text=[]
#         for words in self.anp[id]:
#             anp_text.append(words.split('_')[0])  # adj.
#             anp_text.append(words.split('_')[1])  # n.
#         anp_index = torch.empty(40, dtype=torch.long)
#         for i in range(40):
#             if anp_text[i] in self.word2index:
#                 anp_index[i] = self.word2index[anp_text[i]]
#             else:
#                 anp_index[i] = self.word2index["<unk>"]

#         # if id in anp:
#         #     for words in anp[id]:
#         #         anp_text.append(words.split('_')[0])  # adj.
#         #         anp_text.append(words.split('_')[1])  # n.
#         #     anp_index = torch.empty(40, dtype=torch.long)
#         #     for i in range(40):
#         #         if anp_text[i] in word2index:
#         #             anp_index[i] = word2index[anp_text[i]]
#         #         else:
#         #             anp_index[i] = word2index["<unk>"]
#         # else:
#         #     anp_index=torch.zeros([2,20])
#         #     with open('/home/zlj/miss1.txt','a') as op:
#         #         op.writelines(id+'\n')

#         # labels=img2labels[id]
#         # label_index=list(map(lambda label:label2index[label],labels))
#         # return torch.tensor(anp_index).reshape([20,2]).clone().detach() # torch.Size([40])
#         return anp_index.reshape([20,2]).clone().detach() # torch.Size([40])


#     def __text_index_loader(self,id):
#         return self.data[id]["text_index"]

#     def text_loader(self,id):
#         return self.data[id]["text"]

#     def __getitem__(self, index):
#         id = self.image_ids[index]
#         text_index = self.__text_index_loader(id)
#         vision_feature = self.__vision_feature_loader(id).squeeze()
#         y_true = self.group_label(self.data[id]["group"]).squeeze()
#         y_true_ap = self.group_label_ap(self.data[id]["group"])
#         anp_index = self.__anp_loader(id)
#         return vision_feature, text_index, anp_index, y_true, y_true_ap

#     def __len__(self):
#         return len(self.image_ids)


# def train_val_test_split(all_Data,train_fraction):
#     # split the data
#     train_val_test_count=[int(len(all_Data)*train_fraction),len(all_Data)-int(len(all_Data)*train_fraction)]
#     return random_split(all_Data,train_val_test_count,generator=torch.Generator().manual_seed(42))


