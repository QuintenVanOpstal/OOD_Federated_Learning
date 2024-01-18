from datetime import time

import torch.nn

from globalUtils import *

import torch
from .generic_model_training import GenericModelTraining

class TextCCModelTraining(GenericModelTraining):
    
    def __init__(self, config,lr=None, isAttacker=False, loadFromCkpt=False, trainData=None,
                       testData=None,workerId=0,activeWorkersId=None):
        super().__init__(config,lr,isAttacker,loadFromCkpt,trainData,testData,workerId,activeWorkersId)
            
   
    def trainOneEpoch(self,epoch,w0_vec=None):
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time()

        self.model.train()
        epochLoss = 0
        for batchIdx, (label, text, offsets) in enumerate(self.trainLoader):
            if (len(label) < self.trainConfig['batchSize']):
                # self.logger.info('ignore batch due to small size = {}'.format(len(label)))
                continue

            self.optim.zero_grad()  # set gradient to 0
            # hidden = self.repackage_hidden(hidden)

            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if batchIdx % log_interval == 0 and batchIdx > 0:
                elapsed = time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, batchIdx, len(self.trainLoader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time()
            self.optim.step()
            # self.hidden = hidden
            epochLoss += loss.item()

        # clip = 5
        # self.model.train()
        # hidden = self.model.initHidden(self.trainConfig['batchSize'])
        # criterion =  torch.nn.BCEWithLogitsLoss()
        # epochLoss = 0
        # for batchIdx, (text, label) in enumerate(self.trainLoader):
        #     if (len(text) < self.trainConfig['batchSize']):
        #         self.logger.info('ignore batch due to small size = {}'.format(len(label)))
        #         continue
        #
        #     text, label = text.to(self.device), label.to(self.device)
        #     hidden = tuple([each.data for each in hidden])
        #
        #     self.optim.zero_grad()  # set gradient to 0
        #     # hidden = self.repackage_hidden(hidden)
        #
        #     output, hidden = self.model(text, hidden)
        #     # print(output.shape,target.shape)
        #     # print(output)
        #     loss = criterion(output.squeeze(), label.float())
        #     loss.backward()  # compute gradient
        #
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        #     # print(self.trainConfig)
        #     if (self.trainConfig['method'] == 'pgd'):
        #         eps = self.trainConfig['epsilon']
        #         # make sure you project on last iteration otherwise, high LR pushes you really far
        #         if (batchIdx % self.trainConfig['projectFrequency'] == 0 or batchIdx == len(self.trainLoader) - 1):
        #             self.projectToL2Ball(w0_vec, eps)
        #
        #     if batchIdx % 20 == 0:
        #         self.logger.info('Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.workerId,
        #                                                                                              epoch,
        #                                                                                              batchIdx * len(
        #                                                                                                  text),
        #                                                                                              len(self.trainLoader.dataset),
        #                                                                                              100. * batchIdx / len(
        #                                                                                                  self.trainLoader),
        #                                                                                              loss.item()))
        #     self.optim.step()
        #     # self.hidden = hidden
        #     epochLoss += loss.item()

        #self.model.eval()
        #self.logger.info("Accuracy of model {}".format(self.workerId))
        #currTestLoss, curTestAcc = self.validateModel()
        #return currTestLoss, curTestAcc
        #lss,acc_bf_scale = self.validate_model(logger)
        return epochLoss,0,0
     
   
    def validateModel(self,model=None,dataLoader=None):
        # if(model is None):
        #     model = self.model
        # if(dataLoader is None):
        #     dataLoader = self.testLoader
        #
        # criterion = torch.nn.BCEWithLogitsLoss()
        #
        # model.eval()
        # testLoss = 0
        # correct = 0
        # hidden = self.model.initHidden(self.trainConfig['testBatchSize'])
        # with torch.no_grad():
        #     for idx, (data, label) in enumerate(dataLoader):
        #
        #         data, label = data.to(self.device), label.to(self.device)
        #         label = label.type(torch.float32)
        #
        #         hidden       = self.repackage_hidden(hidden)
        #         predicted_label,hidden       = model(data,hidden)
        #         # loss += criterion(predicted_label, label)
        #         predicted_label = predicted_label.to(self.device)
        #         testLoss += criterion(predicted_label, label)
        #         # print(testLoss)
        #
        #         t = predicted_label
        #         a = t.argmax(1).to(self.device)
        #         pred = torch.zeros(t.shape).to(self.device).scatter(1, a.unsqueeze(1), 1.0) # Replace the biggest value with 1.0
        #         # print(pred)
        #         correct     +=  (pred.add(label)==2).sum().item() # add the labels, and count how many 2's there are
        #         # print(correct)
        #         # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # testLoss /= len(dataLoader.dataset)
        # testAcc = 100. * correct / len(dataLoader.dataset)
        #
        # # return loss/len(dataLoader),  100. *  total_acc / len(dataLoader.dataset)
        # return testLoss, testAcc

        if (model is None):
            model = self.model
        if (dataLoader is None):
            dataLoader = self.testLoader

        model.eval()
        total_acc, total_count, loss = 0, 0, 0
        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataLoader):
                predicted_label = model(text, offsets)
                loss += self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)

        return 100 * loss / total_count, 100. * total_acc / total_count


    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)