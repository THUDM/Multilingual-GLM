# Importing stock libraries
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import argparse
import wandb
import time
import numpy as np
from rouge_score import rouge_scorer
from datetime import date


from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the MT5 modules from huggingface/transformers
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from torch import cuda

# Import distributed training module
from accelerate import Accelerator

config = None

class TaskDefinition:
    dataset = {
            "ncls|multitask": {"train": ["ncls:zh", "ncls:en"], "valid": ["ncls:zh"]},
            "ncls|sum2zh": {"train": ["ncls:zh"], "valid": ["ncls:zh"]},
            "ncls|sum2en": {"train": ["ncls:en"], "valid": ["ncls:en"]},
            "ncls|sci_eval": {"train": ["ncls:zh"], "valid": ["scisummnet:zh"]},
            "scisummnet|simple": {"train": ["scisummnet:zh:0.8"], "valid": ["scisummnet:zh:-0.2"]},
            "scisummnet|ncls_mixed": {"train": ["scisummnet:zh:0.8", "ncls:zh:19200"], "valid": ["scisummnet:zh:-0.2"]},
            "mtgcrosssum|S-en-T-zh-V-all": {"train": ["mtgcrosssum:en2zh"], "valid": ["mtgcrosssum:de2zh", "mtgcrosssum:fr2zh", "mtgcrosssum:es2zh", "mtgcrosssum:zh2zh"]}
    }
    def __init__(self, task, subtask):
        self.task = task + "|" + subtask
    def get_train_dataset(self):
        return self.dataset[self.task]["train"]
    def get_valid_dataset(self):
        return self.dataset[self.task]["valid"]

def read_MTGCrossSum(lang, split, size):
    source_lang, target_lang = lang.split("2")
    if target_lang != 'zh':
        raise RuntimeError("Summary prompt for target language hasn't been defined. Please define it first.")
    prompt = " 中文摘要："
    prefix = "../data/mtg_sum/train" if split == "train" else "../data/mtg_sum/dev.annotation"
    texts = []
    f = open(prefix + ".doc." + source_lang)
    for line in f:
        text = line.strip()
        texts.append(text + prompt)
    f.close()
    summs = []
    f = open(prefix + ".sum." + target_lang)
    for line in f:
        summ = line.strip()
        summs.append(summ)
    f.close()
    if len(texts) != len(summs):
        raise RuntimeError("Sizes of source texts and summarys don't match.")
    df = pd.DataFrame({"text": texts, "zh_sum": summs})
    print(df.head())
    return df

def read_NCLS(lang, split, size):
    prompt = " 中文摘要：" if lang == "zh" else " Please summarize in English:"
    df = pd.read_csv('../data/ncls/ncls_en2zh_'+split+'.csv',encoding='utf-8')
    df = df[['text', lang+'_sum']]
    for index, text in enumerate(df.text):
        try:
            text = text.strip()
            if len(text) > config.MAX_LEN - len(prompt):
                df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
        except TypeError:
            continue
        except AttributeError:
            continue
    df.text = df.text + prompt
    if lang == "en":
        df = df.rename(columns={'en_sum':'zh_sum'})
    if size != None:
        size = float(size)
        size = int(size * df.shape[0]) if abs(size) < 1 else int(size)
        df = df.head(size) if size > 0 else df.tail(-size)
    print(df.head())
    return df

def read_Scisummnet(lang, split, size):
    if lang != "zh":
        raise RuntimeError("Scisummnet Summary-to-English Dataset Is Not Included.")
    prompt = " 中文摘要："
    df = pd.read_csv('../data/scisummnet/scisummnet.csv',encoding='utf-8')
    df = df[['text', 'zh_sum']]
    for index, text in enumerate(df.text):
        try:
            text = text.strip()
            if len(text) > config.MAX_LEN - len(prompt):
                df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
        except TypeError:
            continue
        except AttributeError:
            continue
    df.text = df.text + prompt
    if size != None:
        size = float(size)
        size = int(size * df.shape[0]) if abs(size) < 1 else int(size)
        df = df.head(size) if size > 0 else df.tail(-size)
    print(df.head())
    return df

def init_data(task):
    reader = {"ncls": read_NCLS, "scisummnet": read_Scisummnet, "mtgcrosssum": read_MTGCrossSum}
    train_dfs = []
    for trainset in task.get_train_dataset():
        train_args = trainset.split(":") + [None]
        dset, lang, size = train_args[0], train_args[1], train_args[2]
        df = reader[dset](lang, "train", size)
        train_dfs.append(df)
    valid_dfs = []
    for validset in task.get_valid_dataset():
        valid_args = validset.split(":") + [None]
        dset, lang, size = valid_args[0], valid_args[1], valid_args[2]
        df = reader[dset](lang, "val", size)
        valid_dfs.append(df)
    train_df = pd.concat(train_dfs,ignore_index=True)
    valid_df = pd.concat(valid_dfs,ignore_index=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    print("========== train ==========")
    print(train_df.head())
    print("========== valid ==========")
    print(valid_df.head())
    return train_df, valid_df

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.ctext = self.data.text
        self.text = self.data.zh_sum

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

'''
def init_data(task, config, multitask=False):
    
    if task == 'ncls':
        df = pd.read_csv('./ncls_en2zh_train.csv',encoding='utf-8') #latin-1
        df = df[['text', 'zh_sum']]
        prompt = '中文摘要:'
        for index, text in enumerate(df.text):
            try:
                if len(text) > config.MAX_LEN - len(prompt):
                    df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
            except TypeError:
                continue
        df.text = df.text + prompt
        #print(df.head())
               
        if multitask:
        #Multi-Task Learning English Summary + Chinese Summary
            en_df = pd.read_csv('./ncls_en2zh_train.csv',encoding='utf-8') #latin-1
            en_df = en_df[['text', 'en_sum']]
            en_df = en_df.rename(columns={'en_sum':'zh_sum'})
            en_prompt = 'Please summarize in English:'
            for index, text in enumerate(df.text):
                try:
                    if len(text) > config.MAX_LEN - len(prompt):
                        en_df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
                except TypeError:
                    continue
            en_df.text = en_df.text + en_prompt
            #print(en_df.head())

            df = pd.concat([df, en_df], ignore_index=True)
            df = df.sample(frac=0.8).reset_index(drop=True)
    
        print(df.head())
    
        val_df = pd.read_csv('./ncls_en2zh_val.csv', encoding='utf-8')
        val_df = val_df[['text', 'zh_sum']]
        for index, text in enumerate(val_df.text):
            try:
                if len(text) > config.MAX_LEN - len(prompt):
                    df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
            except TypeError:
                continue
        val_df.text = val_df.text + prompt
        print(val_df.head())
        
        return df, val_df

    elif task == 'scisummnet':
        df = pd.read_csv('scisummnet.csv',encoding='utf-8') #latin-1
        df = df[['text', 'zh_sum']]
        prompt = '中文摘要:'
        for index, text in enumerate(df.text):
            try:
                if len(text) > config.MAX_LEN - len(prompt):
                    df.iloc[index].text = text[:config.MAX_LEN - len(prompt)]
            except TypeError:
                continue
        df.text = df.text + prompt
        print(df.head())
        
        train_size = 0.8
        train_dataset=df.sample(frac=train_size,random_state = config.SEED)
        val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        
        return train_dataset, val_dataset 
'''

def train(epoch, tokenizer, model, device, loader, optimizer, accelerator):
    model.train()
    start = time.time()
    for iteration, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        if iteration%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if iteration%50 == 0:
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        #loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    start = time.time()
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=110, 
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.0, 
                length_penalty=0.7, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            if _%100==0:
                now = time.time()
                print(f'evaluation used {now-start} seconds')
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def main(args):
   
    global config

    #Initialize distributed training 
    accelerator = Accelerator()
    device = accelerator.device
    #device = 'cuda' if cuda.is_available() else 'cpu'
    
    # WandB – Initialize a new run
    
    wandb.init(project="mt5-ncls", entity="dogtooth")
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 8 # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 8   # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 40        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 5e-4   # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 768
    config.SUMMARY_LEN = 160

    #Setting finetuned model path
    
    print(f"lr = {config.LEARNING_RATE}")

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only.
    # Adding the summarize text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task.

    #df = pd.read_csv('./news_summary.csv', encoding='latin-1')
    #df = df[['text', 'ctext']]
    #df.ctext = 'summarize: ' + df.ctext
    
    task = args[1]
    subtask = args[2]
    
    '''
    print(f"now training on {task}")
    multitask = False
    if args[2] == 'multitask':
        multitask = True
    train_dataset, val_dataset = init_data(task, config, multitask)
    if args[2] == 'sci_eval':
        _, val_dataset = init_data('scisummnet', config, multitask)
    #ncls 
    if args[2] == 'ncls_mixed':
        ncls_train, _ = init_data('ncls', config)
        ncls_train = ncls_train.head(19200)
        train_dataset = pd.concat([train_dataset, ncls_train],ignore_index=True)
        train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
    #print("FULL Dataset: {}".format(df.shape))
    '''
    
    task_def = TaskDefinition(task, subtask)
    train_dataset, val_dataset = init_data(task_def)
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", from_tf=True)
    
    if config.TRAIN_EPOCHS == 0:
        print("train epoch set to zero, evaluating only")
        model = MT5ForConditionalGeneration()
        model.load_state_dict(torch.load(PATH))
    
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE) 
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Pass every important object (model, optimizer, dataloader) to *accelerator.prepare*
    model, optimizer, training_loader = accelerator.prepare(model, optimizer, training_loader)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    prev_best = 0.0

    today = date.today()
    
    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer, accelerator)
        #evaluating rouge score at end of every epoch
        #scheduler.step()
        for val_epoch in range(config.VAL_EPOCHS):
            predictions, actuals = validate(val_epoch, tokenizer, model, device, val_loader)
                   
            preds = [' '.join(str(encoding) for encoding in tokenizer.encode(pred)) for pred in predictions]
            acts = [' '.join(str(encoding) for encoding in tokenizer.encode(act)) for act in actuals]
            #preds = [' '.join(pred) for pred in predictions]
            #acts = [' '.join(act) for act in actuals]
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
            r1 = []
            r2 = []
            rl = []
            for pred, target in zip(preds, acts):
                scores = scorer.score(pred, target)
                r1.append(scores['rouge1'].fmeasure)
                r2.append(scores['rouge2'].fmeasure)
                rl.append(scores['rougeLsum'].fmeasure)
            #results = rouge.compute(predictions=preds, references=acts, rouge_types=['rouge1', 'rouge2', 'rougeLsum'])
            #print(r1)
            #print(r2)
            #print(rl)
            if np.mean(r1) > prev_best:
                print(f"found best validation r-1 score {np.mean(r1)} at epoch {epoch}")

                prev_best = np.mean(r1)
                torch.save(model.state_dict(), f"saved/{task}-{subtask}-{today}.pth")

            print(f"epoch= {epoch}|r1 = {np.mean(r1)} r2 = {np.mean(r2)} rl = {np.mean(rl)}")
            
            final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
            print(final_df.head(5))
            #final_df.to_csv(f'./predictions_{epoch}.csv')
            #print('Output Files generated for review')

    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        results = rouge.compute(predictions=predictions, references=actuals)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('./predictions.csv')
        print('Output Files generated for review')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    main(sys.argv)
