import wandb

from master_thesis_benge.self_supervised_learning.training.train import train


from master_thesis_benge.self_supervised_learning.config.constants import (
    SENTINEL_2_INDEX_KEY,
    SENTINEL_1_INDEX_KEY,
)

def train_setup():
    
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "parameters": {
            "batch_size": {"values": [128]},
            "temperature": {"values": [0.1]},
            "modalities": {"values": [
                                        [SENTINEL_1_INDEX_KEY, SENTINEL_2_INDEX_KEY]
                                    ]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="self-supervised-test-run"
    )

    wandb.agent(sweep_id, function=train, count=20)