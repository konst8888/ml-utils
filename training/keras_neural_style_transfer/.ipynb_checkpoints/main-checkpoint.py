import os
import warnings

from omegaconf import DictConfig
import hydra
from utilities import save_code
from train import train

warnings.filterwarnings('ignore')


def os_based_path(path):
    if path:
        return os.path.join(hydra.utils.get_original_cwd(), os.path.join(*path.split("/")))
    return ""


@hydra.main(config_path='conf', config_name='config_server')
def run(cfg: DictConfig) -> None:
    print(cfg.pretty())
    if cfg.general.save_code:
        save_code()
    # somehow squeeze?
    cfg.dataset.style_path = os_based_path(cfg.dataset.style_path)
    cfg.dataset.path = os_based_path(cfg.dataset.path)
    cfg.model.path = os_based_path(cfg.model.path)
    cfg.model.vgg_path = os_based_path(cfg.model.vgg_path)
    train(cfg)


if __name__ == '__main__':
    run()
