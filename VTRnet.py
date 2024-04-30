import torch.nn as nn
import torch


class VTRnet(nn.Module):

    def __init__(self, dim_vision_encoder, dim_text_encoder, dropout_rate=0.5):
        super(VTRnet, self).__init__()

        self.vision_dim = dim_vision_encoder  ##2048
        self.text_dim = dim_text_encoder  ## 1356
        self.hidden = 1024
        # self.vision_layerNorm = torch.nn.LayerNorm([self.vision_dim])
        # self.text_layerNorm = torch.nn.LayerNorm([self.text_dim])
        ##########################################
        # vision encoder
        ##########################################
        self.vision_encoder = nn.Sequential(
            nn.Linear(self.vision_dim, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        ##########################################
        # private vision encoder
        ##########################################

        self.vision_private_encoder = nn.Sequential(
            nn.Linear(self.hidden, 786), nn.BatchNorm1d(786),
            nn.Dropout(dropout_rate), nn.ReLU())

        ##########################################
        # text encoder
        ##########################################

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden), nn.BatchNorm1d(self.hidden),
            nn.Dropout(dropout_rate), nn.ReLU())

        ##########################################
        # private text encoder
        ##########################################

        self.text_private_encoder = nn.Sequential(nn.Linear(self.hidden, 786),
                                                  nn.BatchNorm1d(786),
                                                  nn.Dropout(dropout_rate),
                                                  nn.ReLU())

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.common_encoder = nn.Sequential(nn.Linear(self.hidden, 786),
                                            nn.BatchNorm1d(786),
                                            nn.Dropout(dropout_rate),
                                            nn.ReLU())

        ######################################
        # vision decoder (small decoder)
        ######################################

        self.vision_decoder = nn.Sequential(
            # nn.Linear(786, (786+self.vision_dim)//2),
            # nn.BatchNorm1d((786 + self.vision_dim) // 2),
            # nn.Dropout(dropout_rate),
            # nn.Tanh(),
            # nn.Linear((786+self.vision_dim)//2, self.vision_dim),
            # nn.BatchNorm1d(self.vision_dim),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),
            nn.Linear(786, self.vision_dim),
            nn.BatchNorm1d(self.vision_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        ######################################
        # text decoder (small decoder)
        ######################################

        self.text_decoder = nn.Sequential(
            # nn.Linear(786, (786+self.text_dim)//2),
            # nn.BatchNorm1d((786 + self.text_dim) // 2),
            # nn.Dropout(dropout_rate),
            # nn.Tanh(),
            #
            # nn.Linear((786+self.text_dim)//2, self.text_dim),
            # nn.BatchNorm1d(self.text_dim),
            # nn.Dropout(dropout_rate),
            # nn.ReLU(),
            nn.Linear(786, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )

        # self.VT_Linear = nn.Sequential(
        #     nn.Linear(1572, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.Dropout(dropout_rate),
        #     nn.ReLU(),
        # )

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

    def forward(self, vision, text):
        # vln = self.vision_layerNorm(vision)
        # tln = self.text_layerNorm(text)
        visionEncoder = self.vision_encoder(
            vision)  ## encoder,get hidden representation

        vision_private_encoder = self.vision_private_encoder(
            visionEncoder)  ## private encoder # h^p_v
        vision_common_encoder = self.common_encoder(
            visionEncoder)  ## common encoder

        vision_decoder = self.vision_decoder(vision_private_encoder +
                                             vision_common_encoder)  ## decoder

        textEncoder = self.text_encoder(text)

        text_private_encoder = self.text_private_encoder(textEncoder) # h^p_t
        text_common_encoder = self.common_encoder(textEncoder)

        text_decoder = self.text_decoder(text_private_encoder +
                                         text_common_encoder)

        # VT = self.VT_Linear(torch.cat((vision_private_encoder, text_private_encoder), dim=1))

        return vision_common_encoder, text_common_encoder, vision_private_encoder, text_private_encoder, vision_decoder, text_decoder
