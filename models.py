import torch.nn as nn
import torch.nn.functional as F
import timm


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained = True)
        num_filters = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Sequential(nn.Linear(num_filters,512),
                                                     nn.ReLU(),
                                                     nn.Linear(512,10))

        self.fc1 = nn.Linear(10, 1)

    def forward(self, x1, x2):
        output1 = self.efficientnet(x1)
        output2 = self.efficientnet(x2)

        diff_tensor = torch.abs(output1 - output2)
        scores = torch.sigmoid(self.fc1(diff_tensor))
        return scores

    
class TripletClassificationNet(nn.Module):
    def __init__(self):
        super(TripletClassificationNet, self).__init__()
        self.embedding_net = timm.create_model('efficientnet_b0', pretrained = True)
        num_filters = self.embedding_net.classifier.in_features
        self.embedding_net.classifier = nn.Sequential(nn.Linear(num_filters,512),
                                                      nn.ReLU(),
                                                      nn.Linear(512,10))
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)

        return output1, output2, output3
