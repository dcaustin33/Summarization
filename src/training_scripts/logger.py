import torchmetrics
import torch

def log_metrics(metrics: dict, 
                step: int,
                args,
                wandb = None, 
                train = True,
                ) -> None:

    for i in metrics:
        if 'rouge' in i:
            metrics[i] = torch.mean(torch.tensor(metrics[i]))
    if train:
        print(step, "Loss:", round(metrics['loss'].item(), 2))

    print('In Logging', wandb)
    if wandb:
        if not train:
            new_metrics = {}
            for i in metrics:
                new_metrics['Val ' + i] = metrics[i]
            print('logging val', step)
            #print(new_metrics)
            wandb.log(new_metrics, step = step)
        else:
            print('logging')
            wandb.log(metrics, step = step)

    return
