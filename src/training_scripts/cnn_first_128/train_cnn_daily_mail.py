import torch
from datasets import load_dataset
from rouge import Rouge

import transformers

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import transformers
from trainer import Trainer
from torch.utils.data import DataLoader
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from logger import log_metrics



class PegasusCNNDataset(torch.utils.data.Dataset):
    def __init__(self, model_name = 'google/pegasus-large', max_length=256, split = 'Train'):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.tokenizer.max_length = max_length
        self.dataset = load_dataset('cnn_dailymail', '3.0.0', split = split)
        self.max_length = max_length
        
        #we want to tokenize both our inputs and outputs before passing to the model
        #self.inputs = self.tokenizer(self.dataset['article'], max_length=self.max_length, truncation=True, padding="longest", return_tensors="pt")
        #self.outputs = self.tokenizer(self.dataset['highlights'], max_length=self.max_length, truncation=True, padding="longest", return_tensors="pt")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['article']

        summary_text = self.dataset[idx]['highlights']
        return {'article_text':text, 'summary_text': summary_text}

#create the model
#create the model
def create_model(model_name, max_length):
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.config.max_length = max_length
    return model

def reset_metrics(metrics):
    for i in metrics:
        metrics[i] = 0
    return metrics

def training_step(data, model, metrics, step, log = False, wandb = None, args = None):
    data['article']['input_ids'] = data['article']['input_ids'].cuda()
    data['article']['attention_mask'] = data['article']['attention_mask'].cuda()
    data['summary']['input_ids'] = data['summary']['input_ids'].cuda()


    out =  model(input_ids = data['article']['input_ids'], labels = data['summary']['input_ids'], attention_mask = data['article']['attention_mask'])
    metrics['loss'] += out['loss']
    if log:
        log_metrics(metrics, step, args, wandb = wandb, train = True)
        reset_metrics(metrics)
    return out['loss']

def validation_step(data, model, metrics, steps, log = False, wandb = None, args = None):
    with torch.no_grad():
        data['article']['input_ids'] = data['article']['input_ids'].cuda()
        data['article']['attention_mask'] = data['article']['attention_mask'].cuda()
        data['summary']['input_ids'] = data['summary']['input_ids'].cuda()

        out = model(input_ids = data['article']['input_ids'].cuda(), labels = data['summary']['input_ids'].cuda(), attention_mask = data['article']['attention_mask'].cuda())
        generate_out = model.generate(input_ids = data['article']['input_ids'], attention_mask = data['article']['attention_mask'])
        model_out = dataset.tokenizer.batch_decode(generate_out)

        rouge = Rouge()
        rouge_score = rouge.get_scores(model_out, list(data['summary_text']), avg = True)

        metrics['loss'] += out['loss']
        metrics['rouge1_f'] += rouge_score['rouge-1']['f']
        metrics['rouge2_f'] += rouge_score['rouge-2']['f']
        metrics['rougeL_f'] += rouge_score['rouge-l']['f']
        metrics['rouge1_p'] += rouge_score['rouge-1']['p']
        metrics['rouge2_p'] += rouge_score['rouge-2']['p']
        metrics['rougeL_p'] += rouge_score['rouge-l']['p']
        metrics['rouge1_r'] += rouge_score['rouge-1']['r']
        metrics['rouge2_r'] += rouge_score['rouge-2']['r']
        metrics['rougeL_r'] += rouge_score['rouge-l']['r']

        print(log)
        if log:
            log_metrics(metrics, steps, args, wandb = wandb, train = False)
            reset_metrics(metrics)
        return None



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Summarization Experiments')
    parser.add_argument('--model_name', type=str, default='google/pegasus-large', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max length of the input')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to train for')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--log_n_train_steps', type=int, default=100, help='Log every n steps')
    parser.add_argument('--log_n_val_steps', type=int, default=100, help='Log every n steps')
    parser.add_argument('--val_steps', type=int, default=5, help='Log every n steps')
    parser.add_argument('-checkpoint', action='store_true', help='Load from checkpoint')
    parser.add_argument('--checkpoint_path', default = None, type = str, help = 'Path to checkpoint')
    parser.add_argument('--checkpoint_every_n_steps', default = 100, type = int, help = 'Save checkpoint every n steps')
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--warmup_steps', default = 100, type = int, help = 'Number of warmup steps')
    parser.add_argument('-log', action='store_true', help='Use wandb')

    args = parser.parse_args()

    #create the dataset
    dataset = PegasusCNNDataset(model_name = args.model_name, max_length=args.max_length, split = 'train')
    val_dataset = PegasusCNNDataset(model_name = args.model_name, max_length=args.max_length, split = 'validation')

    #create the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    #create the model
    model = create_model(args.model_name, args.max_length)
    if args.checkpoint:
        checkpoint = torch.load('{name}'.format(name = args.checkpoint_path), map_location='cpu')
        new_dict = checkpoint['model_state_dict']
        model.load_state_dict(new_dict)
        optimizer =  torch.optim.AdamW(model.parameters(), lr = 1e-5, weight_decay = .0001)
        
        schedule = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs= args.warmup_steps,
                        max_epochs= args.steps,
                        warmup_start_lr=3e-05,
                        eta_min=0)
        for i in range(checkpoint['step']):
            schedule.step()

        step = checkpoint['step']
        
        print('Restarting from step:', step, 'with learning rate', schedule.get_last_lr()[0])
    else:
        step = 0
        optimizer =  torch.optim.AdamW(model.parameters(), lr = 1e-5, weight_decay = .0001)
        schedule = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs= args.warmup_steps,
                        max_epochs= args.steps,
                        warmup_start_lr=3e-05,
                        eta_min=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    metrics = {}
    metrics['loss'] = 0

    val_metrics = {}
    val_metrics['loss'] = 0
    val_metrics['rouge1_f'] = 0
    val_metrics['rouge2_f'] = 0
    val_metrics['rougeL_f'] = 0
    val_metrics['rouge1_p'] = 0
    val_metrics['rouge2_p'] = 0
    val_metrics['rougeL_p'] = 0
    val_metrics['rouge1_r'] = 0
    val_metrics['rouge2_r'] = 0
    val_metrics['rougeL_r'] = 0

    if args.log:
        wandb = wandb.init(config = args, name = args.name, project = 'Pegasus Summarization')
    else: wandb = None

    trainer = Trainer(model = model,
                        dataloader = dataloader,
                        dataset = dataset,
                        val_dataloader = val_dataloader,
                        args = args,
                        training_step=training_step,
                        validation_step=validation_step,
                        optimizer = optimizer,
                        schedule = schedule,
                        current_step=step,
                        metrics = metrics,
                        val_metrics = val_metrics,
                        wandb = wandb)

    trainer.train()
