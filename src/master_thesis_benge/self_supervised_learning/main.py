import wandb

from master_thesis_benge.self_supervised_learning.training.train import train

from master_thesis_benge.supervised_baseline.config.constants import (
    SENTINEL_2_INDEX_KEY,
)


if __name__ == "__main__":

    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "parameters": {
            "batch_size": {"values": [256, 512]},
            "temperature": {"values": [0.2, 0.4, 0.6, 0.8, 1]},
            "modalities": {'values':    [#[SENTINEL_2_INDEX_KEY, CLIMATE_ZONE_INDEX_KEY],
                            [SENTINEL_2_INDEX_KEY]]
                },
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="self-supervised-test-run"
    )
    wandb.agent(sweep_id, function=train, count=20)
