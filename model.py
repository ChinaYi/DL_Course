import torch
import math
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv2d(3,16,3, stride = 1, padding = (1,1)),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3, stride=1, padding = (1,1)),
            nn.Conv2d(32,32,3, stride=1, padding = (1,1)),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, stride = 1, padding = (1,1)),
            nn.Conv2d(64,64,3, stride = 1, padding = (1,1)),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, stride = 1, padding = (1,1)),
            nn.Conv2d(128,128,3, stride = 1, padding = (1,1)),
            nn.AvgPool2d(4)
        )
        
        self.fc = nn.Linear(128, 10)
        self._initialize_weights()
        
    def forward(self, input):
        features = self.feature(input)
        features = features.view(features.size(0), -1)
        results = self.fc(features)
        return features, results
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        
    