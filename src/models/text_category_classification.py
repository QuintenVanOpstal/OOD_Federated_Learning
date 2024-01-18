import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCategoryClassificationModel(nn.Module):

    def __init__(self, params):
        super(TextCategoryClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(params['vocabSize'], params['embeddingDim'], sparse=False)
        self.criterion = nn.CrossEntropyLoss().to(params["device"])

        # self.fc = nn.Linear(params['embeddingDim'], 256)
        # self.fc1 = nn.Linear(256, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64,params['outputDim'])

        self.fc = nn.Linear(params['embeddingDim'], params['outputDim'])

        self.init_weights()
        # self.softmax = nn.Softmax()
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(params['dropout'])


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        # self.fc1.weight.data.uniform_(-initrange, initrange)
        # self.fc1.bias.data.zero_()
        # self.fc2.weight.data.uniform_(-initrange, initrange)
        # self.fc2.bias.data.zero_()
        # self.fc3.weight.data.uniform_(-initrange, initrange)
        # self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)

        out = self.fc(embedded)
        # out = self.relu(out)
        # out = self.dropout(out)
        #
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        #
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        #
        # out = self.fc3(out)
        # out = self.softmax(out)

        return out