import torchmetrics
import torch
from sklearn.metrics import roc_auc_score, roc_curve

def log_metrics(metrics: dict, 
                step: int,
                args,
                wandb = None, 
                train = True,
                ) -> None:

    if train:
        print(step, "Loss:", round(metrics['loss'].item(), 2))

    print('In Logging', wandb)
    if wandb:
        if not train:
            new_metrics = {}
            for i in metrics:
                new_metrics['Val ' + i] = metrics[i]
            print('logging')
            print(new_metrics)
            wandb.log(new_metrics, step = step)
        else:
            print('logging')
            wandb.log(metrics, step = step)

    return
