import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        print(self.hparams.student.hidden_size)
