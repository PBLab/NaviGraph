from omegaconf import DictConfig
import hydra
from navigraph.utils.utils import get_input_arguments
from navigraph.manager import Manager

config_path, config_name = get_input_arguments()


@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):

    manager = Manager(cfg=cfg)
    manager.run()


if __name__ == '__main__':
    main()

