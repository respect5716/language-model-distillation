import os
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    model = hydra.utils.instantiate(config.model)

if __name__ == '__main__':
    main()