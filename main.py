import os
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    data_module = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    # data_module.setup()
    # d = next(iter(data_module.train_dataloader()))
    # loss = model.training_step(d, 0)    
    trainer = hydra.utils.instantiate(config.trainer, model=model)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()