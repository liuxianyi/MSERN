# coding=utf-8
import torch
from ANPFeat import ANPFeatExtract
from AnpTextRnet import *
from TextFeature import *
import torch.nn.functional as F
from VTRnet import *
from Classifier import *
from torch.autograd import Variable
from collections import OrderedDict


class MMFSTSR(object):
    def __init__(self, opt):
        ## Build Models
        self.opt = opt
        self.TextFea = ExtractTextFeature(opt)
        self.ANPFea = ANPFeatExtract(opt, hidden_size=opt.ANPFeatHidden)
        self.AnpText = RNet(opt)

        self.VT = VTRnet(opt.dim_vision_encoder, opt.dim_text_encoder)
        self.ClassLayer = ClassifierLayer(opt.class_in_dim, opt.class_out_dim)

        params = list(self.TextFea.parameters())
        params += list(self.ANPFea.parameters())
        params += list(self.AnpText.parameters())
        params += list(self.VT.parameters())
        params += list(self.ClassLayer.parameters())
        self.params = params

        if opt.optimizer.type == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer.type == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=opt.learning_rate,
                momentum=opt.optimizer.momentum,
                weight_decay=opt.optimizer.weight_decay,
                nesterov=opt.optimizer.nesterov)
        else:
            raise NotImplementedError('Only support Adam and SGD optimizer.')
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            opt.lr_scheduler.lr_steps,
            gamma=0.1,
            last_epoch=-1)
        if torch.cuda.is_available():
            self.TextFea.to(opt.device)
            self.ANPFea.to(opt.device)
            self.AnpText.to(opt.device)
            self.VT.to(opt.device)
            self.ClassLayer.to(opt.device)

        self.Eiters = 0

    def loss_criterion(self, vc, tc, vp, tp, output, y_true, vd, vln, td, tln):
        ## orthogonal loss
        orth_loss = torch.norm(torch.mm(vp.T, vc)) + torch.norm(
            torch.mm(tp.T, tc))

        ## class loss
        classification = torch.nn.BCELoss()
        c_loss = classification(output, y_true.float())
        ## recon_loss
        recon_loss = (F.mse_loss(vd, vln) + F.mse_loss(td, tln)) / 2
        ##all_loss=c_loss+self.opt.lam3 * recon_loss
        all_loss = self.opt.lam1 * c_loss + self.opt.lam2 * orth_loss + self.opt.lam3 * recon_loss
        return all_loss

    def state_dict(self):
        state_dict = [
            self.TextFea.state_dict(),
            self.ANPFea.state_dict(),
            self.AnpText.state_dict(),
            self.VT.state_dict(),
            self.ClassLayer.state_dict()
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.TextFea.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            new_state_dict[k] = v
        self.ANPFea.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[2].items():
            new_state_dict[k] = v
        self.AnpText.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[3].items():
            new_state_dict[k] = v
        self.VT.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[4].items():
            new_state_dict[k] = v
        self.ClassLayer.load_state_dict(new_state_dict, strict=True)

    def train_start(self):
        """
        switch to the train mode
        :return:
        """
        self.TextFea.train()
        self.ANPFea.train()
        self.AnpText.train()
        self.VT.train()
        self.ClassLayer.train()

    def eval_start(self):
        """
        switch to the eval mode
        :return:
        """
        self.TextFea.eval()
        self.ANPFea.eval()
        self.AnpText.eval()
        self.VT.eval()
        self.ClassLayer.eval()

    def train_emb(self, vision, text_index, anp_index, y_true):
        vision = Variable(vision)
        text_index = Variable(text_index)
        anp_index = Variable(anp_index)
        y_true = Variable(y_true)
        if torch.cuda.is_available():
            vision = vision.to(self.opt.device)
            text_index = text_index.to(self.opt.device)
            anp_index = anp_index.to(self.opt.device)
            y_true = y_true.to(self.opt.device)

        self.optimizer.zero_grad()
        anp, anp_seq = self.ANPFea(anp_index) #(batchsize, hid_dim*num_directions) the whole sentence, (batchsize, seq_len, hid_dim*num_directions)
        text, text_seq = self.TextFea(text_index, anp) # (batchsize, hid_dim*num_directions), (batchsize, seq_len, hid_dim*num_directions)

        AnpText = self.AnpText(text, text_seq.permute(1, 0, 2), anp,
                               anp_seq.permute(1, 0, 2)) # S_J
        vc, tc, vp, tp, vd, td = self.VT(vision, text)
        output = self.ClassLayer(vp, tp, AnpText)

        loss = self.loss_criterion(vc, tc, vp, tp, output, y_true, vd, vision,
                                   td, text)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.opt.grad_clip)
        self.optimizer.step()
        # self.scheduler.step()
        return output, loss

    def test_emb(self, vision, text_index, anp_index, y_true, y_true_ap):
        vision = Variable(vision)
        text_index = Variable(text_index)
        anp_index = Variable(anp_index)
        y_true = Variable(y_true)
        if torch.cuda.is_available():
            vision = vision.to(self.opt.device)
            text_index = text_index.to(self.opt.device)
            anp_index = anp_index.to(self.opt.device)
            y_true = y_true.to(self.opt.device)

        with torch.no_grad():
            anp, anp_seq = self.ANPFea(anp_index)
            text, text_seq = self.TextFea(text_index, anp)
            AnpText = self.AnpText(text, text_seq.permute(1, 0, 2), anp,
                                   anp_seq.permute(1, 0, 2))
            vc, tc, vp, tp, vd, td = self.VT(vision, text)
            output = self.ClassLayer(vp, tp, AnpText)  # after softmax
            loss = self.loss_criterion(vc, tc, vp, tp, output, y_true, vd,
                                       vision, td, text)

        return output, loss, torch.cat([vp, tp], dim=1), AnpText
