import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
