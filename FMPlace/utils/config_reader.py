import yaml
from dataclasses import dataclass

@dataclass
class Config:
    mode: str
    batchSize: int
    cacheBatchSize: int
    nEpochs: int
    nGPU: int
    lr: float
    lrStep: float
    lrGamma: float
    weightDecay: float
    momentum: float
    threads: int
    seed: int
    runsPath: str
    cachePath: str
    load_from: str
    ckpt: str
    dataset_path: str
    USE_NCLT: bool
    USE_HELILPR: bool
    representation: str
    

def get_config(cfg_path='./representation_analysis/FMPlace/cfg/cfg.yaml'):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Config(**cfg_dict)

# Example usage
if __name__ == "__main__":
    config = get_config()
    print(config)
