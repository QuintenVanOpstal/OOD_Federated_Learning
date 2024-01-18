from training.text_cc_training import TextCCModelTraining


def getModelTrainer(conf, lr=None, isAttacker=False):
    model = conf['arch']
    if (model == "textCC"):
        return TextCCModelTraining(conf, lr, isAttacker=isAttacker)

