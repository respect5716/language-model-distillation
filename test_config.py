import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path='conf', config_name='minilmv2')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config, resolve=True))
    print('=' * 100)

if __name__ == '__main__':
    main()