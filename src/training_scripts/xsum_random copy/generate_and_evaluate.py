import torch
from datasets import load_dataset
from rouge import Rouge
from torch import nn
import transformers

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import transformers
from trainer import Trainer
from torch.utils.data import DataLoader
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from logger import log_metrics
from evaluator import Evaluator
from evaluate import load
import numpy as np

import sys
import time
import os
import argparse

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True


def create_model(model_name, max_length):
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.config.max_length = max_length
    return model


class XSumDatasetRandom(torch.utils.data.Dataset):
    def __init__(self, model_name = 'google/pegasus-large', max_length=256, split = 'train'):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.tokenizer.max_length = max_length
        self.dataset = load_dataset("xsum", split = split)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['document']
        text = text.split(' ')
        max_idx = max(1, len(text) - self.max_length)
        text = text[np.random.randint(max_idx):]
        text = ' '.join(text)

        summary_text = self.dataset[idx]['summary']
        return {'article_text':text, 'summary_text': summary_text}

def validation_step(data, model, metrics, steps, log = False, wandb = None, args = None, file_name = None):
    with torch.no_grad():
        data['article']['input_ids'] = data['article']['input_ids'].cuda()
        data['article']['attention_mask'] = data['article']['attention_mask'].cuda()
        data['summary']['input_ids'] = data['summary']['input_ids'].cuda()

        out = model(input_ids = data['article']['input_ids'].cuda(), labels = data['summary']['input_ids'].cuda(), attention_mask = data['article']['attention_mask'].cuda())
        generate_out = model.generate(input_ids = data['article']['input_ids'], attention_mask = data['article']['attention_mask'])
        model_out = val_dataset.tokenizer.batch_decode(generate_out)

        if steps < args.write_steps:
            file = open(file_name, "a")

            for i in range(len(model_out)):
                file.write('Article text: ' + data['article_text'][i] + '\n')
                file.write('----------------------------------------' + '\n')
                file.write('Ground truth Summary: ' + data['summary_text'][i] + '\n')
                file.write('----------------------------------------' + '\n')
                file.write('Generated Summary: ' + model_out[i] + '\n')
                file.write('----------------------------------------' + '\n')
                file.write('----------------------------------------' + '\n')
                file.write('----------------------------------------' + '\n')
                file.write('----------------------------------------' + '\n')
            file.close()

        rouge = Rouge()
        rouge_score = rouge.get_scores(model_out, list(data['summary_text']), avg = True)
        results = bertscore.compute(predictions=model_out, references=list(data['summary_text']), lang="en")

        metrics['loss'] += out['loss']
        metrics['rouge1_f'].append(rouge_score['rouge-1']['f'])
        metrics['rouge2_f'].append(rouge_score['rouge-2']['f'])
        metrics['rougeL_f'].append(rouge_score['rouge-l']['f'])
        metrics['rouge1_p'].append(rouge_score['rouge-1']['p'])
        metrics['rouge2_p'].append(rouge_score['rouge-2']['p'])
        metrics['rougeL_p'].append(rouge_score['rouge-l']['p'])
        metrics['rouge1_r'].append(rouge_score['rouge-1']['r'])
        metrics['rouge2_r'].append(rouge_score['rouge-2']['r'])
        metrics['rougeL_r'].append(rouge_score['rouge-l']['r'])
        metrics['BERT Score'].append(np.mean(results['f1']))

        if log:
            log_metrics(metrics, steps, args, wandb = wandb, train = False)
        return None



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Summarization Experiments')
    parser.add_argument('--model_name', type=str, default='google/pegasus-large', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max length of the input')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--log_n_val_steps', type=int, default=100, help='Log every n steps')
    parser.add_argument('--val_steps', type=int, default=5, help='Log every n steps')
    parser.add_argument('--model_path', default = None, type = str, help = 'Path to checkpoint')
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--write_steps', nargs='?', default = 10,  type=int)
    parser.add_argument('--num_beams', nargs='?', default = 4,  type=int)
    parser.add_argument('-log', action='store_true', help='Use wandb')

    #create model and load weights
    args = parser.parse_args()
    model = create_model(args.model_name, args.max_length)
    checkpoint = torch.load('{name}'.format(name = args.model_path), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.config.num_beams = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    val_dataset = XSumDatasetRandom(model_name = args.model_name, max_length=args.max_length, split = 'validation')
    print('Not shuffling validation set')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_metrics = {}
    val_metrics['loss'] = 0
    val_metrics['rouge1_f'] = []
    val_metrics['rouge2_f'] = []
    val_metrics['rougeL_f'] = []
    val_metrics['rouge1_p'] = []
    val_metrics['rouge2_p'] = []
    val_metrics['rougeL_p'] = []
    val_metrics['rouge1_r'] = []
    val_metrics['rouge2_r'] = []
    val_metrics['rougeL_r'] = []
    val_metrics['BERT Score'] = []

    generation_name = 'generation_{name}.txt'.format(name = args.name)

    with open(generation_name, 'w') as file:
        file.write(f'Generations for {args.name}\n')


    if args.log:
        wandb = wandb.init(config = args, name = args.name, project = 'Pegasus Summarization')
    else: wandb = None

    evaluator = Evaluator(model, 
                            val_dataloader, 
                            args, 
                            validation_step, 
                            val_dataset, 
                            val_metrics, 
                            wandb = wandb, 
                            file_name = generation_name)
    bertscore = load("bertscore")

    evaluator.evaluate()