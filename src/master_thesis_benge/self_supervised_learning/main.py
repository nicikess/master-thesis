import wandb

from master_thesis_benge.self_supervised_learning.training.train import train

if __name__ == "__main__":

    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "parameters": {
            "batch_size": {"values": [256, 512]},
            "temperature": {"values": [0.2, 0.4, 0.6, 0.8, 1]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="self-supervised-test-run"
    )
    wandb.agent(sweep_id, function=train, count=20)
