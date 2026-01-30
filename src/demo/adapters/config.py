import yaml
import tqdm
from importlib import import_module

from core.services.image_processing import ImageProcessingService

def parse_model_config(cfg):
    model_path, model_name = cfg["model"].rsplit(".", 1)
    kwargs = cfg["params"]
    backbone_path, backbone_name = cfg["backbone"].rsplit(".", 1)

    module = import_module(model_path)
    model_constructor = getattr(module, model_name)
    backbone_module = import_module(backbone_path)
    backbone_constructor = getattr(backbone_module, backbone_name)

    return model_constructor, backbone_constructor, kwargs

def load_processors():
    with open("infrastructure/config/models.yaml") as f:
        config = yaml.safe_load(f)
    
    for name, cfg in config["processors"].items():
        model_constructor, backbone_constructor, kwargs = parse_model_config(cfg)
        ImageProcessingService.register_processor(name, (model_constructor, backbone_constructor, kwargs))