import pytorch_lightning as pl



class Trainer(pl.Trainer):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.logger.watch(model)