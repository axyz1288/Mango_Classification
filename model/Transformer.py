import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.ReLU6(),
                                    nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False, groups=hidden_dim),
                                    nn.BatchNorm2d(hidden_dim),
                                    nn.ReLU6(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(hidden_dim, output_dim, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(output_dim),
                                    nn.ReLU6(),
                                    nn.MaxPool2d(2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        

class Transformer(nn.Module):
    def __init__(self, ninp, nemb, nhead, nhid, nlayers, nclass, dropout=0.2, mask=False):
        super(Transformer, self).__init__()
        self.mask = mask
        self.encoder = Encoder(ninp, nemb, nemb*4)
        encoderlayer = nn.TransformerEncoderLayer(nemb, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoderlayer, nlayers)
        self.decoder = nn.Linear(nemb, nclass)
            
    def init_weights(self, m):
        classname = m.__class__.__name__
        if (classname == 'Conv2d'):
            n = torch.tensor(m.in_channels, dtype=torch.float)
            y = 1.0 / torch.sqrt(n)
            m.weight.data.uniform_(-y, y)
        if (classname == 'Linear'):
            n = torch.tensor(m.in_features, dtype=torch.float)
            y = 1.0 / torch.sqrt(n)
            m.weight.data.uniform_(-y, y)
        
    def forward(self, x):         
        x = self.encoder(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        if(self.mask is True):
            mask = (torch.triu(torch.ones(x.shape[0], x.shape[0]), diagonal=1) == 1)
            mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
            x = self.transformer_encoder(x, self.mask)
        else:
            x = self.transformer_encoder(x)
        x = self.decoder(x[int(x.shape[0] * 0.5), :, :])
        return x