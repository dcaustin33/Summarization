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
#import gradient checkpointing
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class CNNBERT(torch.utils.data.Dataset):
    def __init__(self, model_name = 'google/pegasus-large', max_length=256, split = 'train'):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.tokenizer.max_length = max_length
        self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.dataset = load_dataset('cnn_dailymail', '3.0.0', split = split)
        self.max_length = max_length

    @torch.no_grad()
    def get_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        embeddings = self.model(**inputs)['pooler_output']
        #compute cosine similarity between embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['article']
        text = text.split('.')
        text = [i.strip() for i in text]
        
        embeddings = self.get_bert_embeddings(text)
        first = np.random.choice(len(text), 1, replace=False)[0]
        chosen_embeddings = torch.empty((1, 768)).cuda()
        bag_of_sentences = [first]
        chosen_embeddings[0] = embeddings[first]
        current_size = len(text[first])

        while current_size < self.max_length and len(bag_of_sentences) < len(text):
            new_cosine_sim = torch.mm(chosen_embeddings, embeddings.T)
            vals, indices = torch.topk(-torch.sum(new_cosine_sim, dim = 0), k = len(text))
            for i in indices:
                if i not in bag_of_sentences:
                    chosen_embeddings = torch.cat((chosen_embeddings, embeddings[i].unsqueeze(0)), dim = 0)
                    bag_of_sentences.append(i.item())
                    current_size += len(text[i])
                    break

        bag_of_sentences = sorted(bag_of_sentences)
        final = [text[i] for i in bag_of_sentences]
        final = '. '.join(final)

        summary_text = self.dataset[idx]['highlights']
        return {'article_text':final, 'summary_text': summary_text}

#create the model
def create_model(model_name, max_length):
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    model.config.max_length = max_length
    return model

#create wrapper for PegasusForConditionalGeneration
class PegasusWrapper(torch.nn.Module):
    def __init__(self, model_name, max_length):
        super().__init__()
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.model.config.max_length = max_length

    def forward(self, input_variables):
        input_ids = input_variables['input_ids']
        attention_mask = input_variables['attention_mask']
        labels = input_variables['labels']
        return self.model(input_ids = input_ids, labels = labels, attention_mask = attention_mask)

def reset_metrics(metrics, val = False):
    for i in metrics:
        metrics[i] = 0
    if val:
        metrics['rouge1_f'] = []
        metrics['rouge2_f'] = []
        metrics['rougeL_f'] = []
        metrics['rouge1_p'] = []
        metrics['rouge2_p'] = []
        metrics['rougeL_p'] = []
        metrics['rouge1_r'] = []
        metrics['rouge2_r'] = []
        metrics['rougeL_r'] = []
    return metrics

def training_step(data, model, metrics, step, log = False, wandb = None, args = None):
    data['article']['input_ids'] = data['article']['input_ids'].cuda()
    data['article']['attention_mask'] = data['article']['attention_mask'].cuda()
    data['summary']['input_ids'] = data['summary']['input_ids'].cuda()


    out =  model(input_ids = data['article']['input_ids'], labels = data['summary']['input_ids'], attention_mask = data['article']['attention_mask'])
    metrics['loss'] += out['loss'].detach().cpu().numpy()
    if log:
        log_metrics(metrics, step, args, wandb = wandb, train = False)
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
        metrics['rouge1_f'].append(rouge_score['rouge-1']['f'])
        metrics['rouge2_f'].append(rouge_score['rouge-2']['f'])
        metrics['rougeL_f'].append(rouge_score['rouge-l']['f'])
        metrics['rouge1_p'].append(rouge_score['rouge-1']['p'])
        metrics['rouge2_p'].append(rouge_score['rouge-2']['p'])
        metrics['rougeL_p'].append(rouge_score['rouge-l']['p'])
        metrics['rouge1_r'].append(rouge_score['rouge-1']['r'])
        metrics['rouge2_r'].append(rouge_score['rouge-2']['r'])
        metrics['rougeL_r'].append(rouge_score['rouge-l']['r'])

        if log:
            log_metrics(metrics, steps, args, wandb = wandb, train = False)
            reset_metrics(metrics, val = True)
            print(metrics)
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
    parser.add_argument('--num_beams', nargs='?', default = 5,  type=int)
    parser.add_argument('--warmup_steps', default = 100, type = int, help = 'Number of warmup steps')
    parser.add_argument('--divisor', nargs='?', default = 2.0, type = float)
    parser.add_argument('-log', action='store_true', help='Use wandb')

    args = parser.parse_args()

    #create the dataset
    dataset = CNNBERT(model_name = args.model_name, max_length=args.max_length, split = 'train')
    val_dataset = CNNBERT(model_name = args.model_name, max_length=args.max_length, split = 'validation')

    #create the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #create the model
    model = create_model(args.model_name, args.max_length)
    if args.checkpoint:
        checkpoint = torch.load('{name}'.format(name = args.checkpoint_path), map_location='cpu')
        new_dict = checkpoint['model_state_dict']
        model.load_state_dict(new_dict)
        optimizer =  transformers.Adafactor(model.parameters(), 
                                            lr = 1e-4, 
                                            relative_step = False,
                                            eps=(1e-30, 1e-3),
                                            clip_threshold=1.0,
                                            decay_rate=-0.8,
                                            beta1=None,
                                            scale_parameter=False)
        
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
        optimizer =  transformers.Adafactor(model.parameters(), 
                                            lr = 1e-4, 
                                            relative_step = False,
                                            eps=(1e-30, 1e-3),
                                            clip_threshold=1.0,
                                            decay_rate=-0.8,
                                            beta1=None,
                                            scale_parameter=False)

        schedule = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs= args.warmup_steps,
                        max_epochs= args.steps,
                        warmup_start_lr=3e-05,
                        eta_min=0)
    model.config.num_beams = args.num_beams
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    metrics = {}
    metrics['loss'] = 0

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




