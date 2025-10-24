from .sam2 import SAM2
from .ours import TubeletGraph

def get_tracker(config):
    module = globals()[config.module]
    kwargs = dict(config)
    if 'name' in kwargs:
        kwargs.pop('name')
    if 'module' in kwargs:
        kwargs.pop('module')
    return module(**kwargs)