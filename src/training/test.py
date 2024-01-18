import sys
sys.path.append('../')
from globalUtils import *
from dataset.datasets import loadDataset
from model_training_factory import *

if __name__ == "__main__":

    seed(42)
    workerId = 0
    stdoutFlag = True
    logger = getLogger("fl.log", stdoutFlag, logging.INFO) 
    
    configFile = sys.argv[1]
    print("loading conf from {}".format(configFile))
    config = loadConfig(configFile)
    curDataset = loadDataset(config)
    curDataset.buildDataset(conf=config)

    if(config['text']):
        config['modelParams']['vocabSize'] = curDataset.vocabSize +1 

    trainer = getModelTrainer(config)
    trainer.setLogger(logger)
    trainer.createDataLoaders(curDataset.trainData, curDataset.testData)
    trainer.trainNEpochs(config['numEpochs'],validate=True)
    trainer.validateModel()
    
     
    
    
