from dataset.ag_news_dataset import AGNewsData

def loadDataset(conf):
    datasetName = conf['dataset']
    dataPath    = conf['dataPath']
    if (datasetName == "ag_news"):
        return AGNewsData()
    else:
        print('Datset {} Not Defined'.format(datasetName))
        return None

