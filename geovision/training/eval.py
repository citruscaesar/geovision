from lightning import Callback

class EvaluateClassification(Callback):
    # store model output logits in memory in a pre-reserved numpy array during eval step
    # save model outputs and compute confusion matrix and other matrices during eval epoch end 
    # log everything to CSVLogger and WandbLogger during eval end
    pass

class EvaluateSegmentation(Callback):
    pass