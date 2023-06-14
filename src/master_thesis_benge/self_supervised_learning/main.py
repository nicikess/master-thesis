import wandb

from master_thesis_benge.self_supervised_learning.training.train import train


from master_thesis_benge.self_supervised_learning.config.constants import (
    SENTINEL_2_INDEX_KEY,
    SENTINEL_1_INDEX_KEY,
    MODE_TRAIN_KEY,
    MODE_SAVE_WEIGHTS_KEY,
    MODE_EVALUATE_KEY
)

from master_thesis_benge.self_supervised_learning.model import (
    SimCLR_pl
)

if __name__ == "__main__":

    mode = MODE_TRAIN_KEY

    if mode == MODE_TRAIN_KEY:

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

    if mode == MODE_EVALUATE_KEY:
        pass

    if mode == MODE_SAVE_WEIGHTS_KEY:
        pass