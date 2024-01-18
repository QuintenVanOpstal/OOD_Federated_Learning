import torch.nn
import torch.optim as optim
from torchtext.data import get_tokenizer

from torch.utils.data import DataLoader
import logging
import os
from globalUtils import *
import globalUtils
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class GenericModelTraining:

    def __init__(self, config, lr=None, isAttacker=False, loadFromCkpt=False, trainData=None,
                 testData=None, workerId=0, activeWorkersId=None):
        self.workerId = workerId

        # self.workerDataIdxMap = workerDataIdxMap
        self.trainData = trainData
        self.testData = testData
        self.activeWorkersId = activeWorkersId
        self.device = config['device']

        self.trainConfig = config['attackerTrainConfig'] if isAttacker else config['normalTrainConfig']

        self.model, self.criterion = createModel(config)
        if (lr is None):
            lr = self.trainConfig['initLr']
        self.lr = lr

        if (self.trainConfig['optimizer'] == 'adam'):
            self.optim = optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    # momentum=trainConfig['momentum'],
                                    # weight_decay=trainConfig['weightDecay']
                                    )
        else:
            self.optim = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=self.trainConfig['momentum'])

        # self.lr = self.trainConfig['initLr']
        # if(logger is None):
        #    self.logger = globalUtils.getLogger("worker_{}.log".format(workerId), stdoutFlag, logging.INFO)
        # else:
        #     self.logger = logger
        self.trainBatchSize = self.trainConfig['batchSize']
        self.testBatchSize = self.trainConfig['testBatchSize']
        # self.hidden = self.model.initHidden(trainConfig['batchSize'])

    # def createModel(self,conf):
    # model = TextBinaryClassificationModel(conf["modelParams"])
    # def setData(self,trainData,testData):

    def setFLParams(self, flParams):
        self.workerId = flParams['workerId']
        self.activeWorkersId = flParams['activeWorkersId']

    def setLogger(self, logger):
        self.logger = logger

    def createDataLoaders(self, trainData, testData, config, vocab, batchSize=32, testBatchSize=32):

        # print("length of dataset" + str(len(trainData)))
        if config["arch"] == "textCC":
            label_pipeline = lambda x: int(x) - 1

            def collate_batch(batch):
                label_list, text_list, offsets = [], [], [0]
                for _text, _label in batch:
                    label_list.append(label_pipeline(_label))
                    processed_text = torch.tensor(_text, dtype=torch.int64)
                    text_list.append(processed_text)
                    offsets.append(processed_text.size(0))
                label_list = torch.tensor(label_list, dtype=torch.int64)
                offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
                text_list = torch.cat(text_list)
                return label_list.to(config["device"]), text_list.to(config["device"]), offsets.to(config["device"])

            self.trainLoader = DataLoader(trainData, batch_size=self.trainBatchSize, shuffle=True,
                                          collate_fn=collate_batch)
            self.testLoader = DataLoader(testData, batch_size=self.testBatchSize, collate_fn=collate_batch)
        else:
            self.trainLoader = DataLoader(trainData, batch_size=self.trainBatchSize, shuffle=True)
            self.testLoader = DataLoader(testData, batch_size=self.testBatchSize)

        # return trainLoader,testLoader

    def projectToL2Ball(self, w0_vec, eps):

        w = list(self.model.parameters())
        w_vec = parameters_to_vector(w)
        nd = torch.norm(w_vec - w0_vec)
        if (nd > eps):
            # project back into norm ball
            w_proj_vec = eps * (w_vec - w0_vec) / torch.norm(
                w_vec - w0_vec) + w0_vec
            # plug w_proj back into model
            vector_to_parameters(w_proj_vec, w)

    def scaleForReplacement(self, globalModel, totalPoints):
        W0 = list(globalModel.parameters())
        gamma = totalPoints / len(self.trainLoader)
        for idx, param in enumerate(self.model.parameters()):
            param.data = (param.data - W0[idx]) * gamma + W0[idx]

    def trainOneEpoch(self, epoch, w0_vec=None):
        self.model.train()
        epochLoss = 0
        for batchIdx, (data, target) in enumerate(self.trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()  # set gradient to 0
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()  # compute gradient

            if batchIdx % 20 == 0:
                self.logger.info('Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.workerId,
                                                                                                     epoch,
                                                                                                     batchIdx * len(
                                                                                                         data),
                                                                                                     len(self.trainLoader.dataset),
                                                                                                     100. * batchIdx / len(
                                                                                                         self.trainLoader),
                                                                                                     loss.item()))
            self.optim.step()

            if (self.trainConfig['method'] == 'pgd'):
                eps = self.trainConfig['epsilon']
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batchIdx % self.trainConfig['projectFrequency'] == 0 or batchIdx == len(self.trainLoader) - 1):
                    self.logger.info('Projecting')
                    self.projectToL2Ball(w0_vec, eps)

            epochLoss += loss.item()

        # self.model.eval()
        # self.logger.info("Accuracy of model {}".format(self.workerId))
        # currTestLoss, curTestAcc = self.validateModel()
        # return currTestLoss, curTestAcc
        # lss,acc_bf_scale = self.validate_model(logger)
        return epochLoss, 0, 0

    def trainNEpochs(self, w0_vec=None, validate=False):
        lstTestLosses = []
        lstTestAcc = []
        lstTrainLosses = []
        # lstTrainAcc    = []
        n = self.trainConfig['internalEpochs']
        for i in range(n):
            a, b, c = self.trainOneEpoch(i, w0_vec)
            lstTrainLosses.append(a)
            lstTestLosses.append(b)
            lstTestAcc.append(c)
            if (validate):
                testLoss, testAcc = self.validateModel()
                self.logger.info('Epoch: {} Validation Accuracy: {}'.format(i, testAcc))
        return lstTrainLosses, lstTestLosses, lstTestAcc

    def validateModel(self, model=None, dataLoader=None):
        if (model is None):
            model = self.model
        if (dataLoader is None):
            dataLoader = self.testLoader

        model.eval()
        testLoss = 0
        correct = 0
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                testLoss += self.criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        testLoss /= len(dataLoader)
        testAcc = 100. * correct / len(dataLoader.dataset)
        # self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #                    testLoss, correct, len(dataLoader.dataset), testAcc))
        return testLoss, testAcc

        # def trainOneAdversarialEpoch(self):

