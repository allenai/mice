from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Config, T5TokenizerFast 
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import more_itertools as mit
import math
import numpy as np
import pandas as pd
import os
from tqdm import tqdm 
from types import SimpleNamespace
import logging
import json
import sys

# Local imports
from src.masker import Masker, RandomMasker, GradientMasker
from src.dataset import StageOneDataset, RaceStageOneDataset
from src.utils import *

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

def train_epoch(epoch, editor_tokenizer, editor_model, device, loader, optim):
    """ Runs training for epoch """

    editor_model.train()
    total_loss = 0
    logger.info(f"Training epoch: {epoch}")

    for _, data in tqdm(enumerate(loader, 0), total = len(loader)):
        lm_labels = data['target_ids'].to(device, dtype = torch.long)
        lm_labels[lm_labels[:, :] == editor_tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        outputs = editor_model(input_ids = ids, labels=lm_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        del lm_labels
        del ids
        torch.cuda.empty_cache()

    logger.info(f'Epoch: {epoch}, Avg Batch Loss:  {total_loss/len(loader)}')
    return total_loss

def validate_epoch(epoch, editor_tokenizer, editor_model, device, loader):
    """ Runs validation for epoch """

    editor_model.eval()
    total_loss = 0
    logger.info(f"Validating epoch: {epoch}")

    for _, data in tqdm(enumerate(loader, 0), total = len(loader)):
        lm_labels = data['target_ids'].to(device, dtype = torch.long)        
        lm_labels[lm_labels[:, :] == editor_tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)

        outputs = editor_model(input_ids = ids, labels=lm_labels)
        loss = outputs[0]
        total_loss += loss.item()

        del lm_labels
        del ids
        torch.cuda.empty_cache()

    logger.info(f'Epoch: {epoch}, Avg Batch Loss:  {total_loss/len(loader)}')
    return total_loss    

def get_datasets(predictor, dr, masker, data_dir, train_inputs, val_inputs, 
        train_labels, val_labels, editor_tokenizer, args):
    """ Writes data for Editor fine-tuning """

    train_data_path = os.path.join(data_dir, "train_data.csv")
    val_data_path = os.path.join(data_dir, "val_data.csv")

    # If data already exists for experiment, read data
    if os.path.exists(train_data_path) and os.path.exists(val_data_path):
        logger.info("Data for Editor fine-tuning already exist.")
        logger.info(f"Loading train data from: {train_data_path}")
        logger.info(f"Loading val data from: {val_data_path}")

        train_csv = pd.read_csv(train_data_path, sep="\t")
        val_csv = pd.read_csv(val_data_path, sep="\t")

        train_dataset = StageOneDataset(editor_tokenizer, 
                max_length=args.model.model_max_length, 
                masked_strings=train_csv['inputs'], 
                targets=train_csv['targets'])
        val_dataset = StageOneDataset(editor_tokenizer, 
                max_length=args.model.model_max_length, 
                masked_strings=val_csv['inputs'], 
                targets=val_csv['targets'])

    # Else, create data by calling create_inputs() function in dataset.py
    else:
        logger.info("Creating masked data for Editor fine-tuning...")
        logger.info(f"Target label (options are 'pred' or 'gold'): " + \
                f"{args.misc.target_label}")
        # For RACE, pass dr to create_inputs() to correctly truncate
        if args.meta.task == "race":
            train_dataset = RaceStageOneDataset(editor_tokenizer, 
                    max_length=args.model.model_max_length)
            train_dataset.create_inputs(dr, train_inputs, train_labels, 
                    predictor, masker, target_label=args.misc.target_label)
            val_dataset = RaceStageOneDataset(editor_tokenizer, 
                    max_length=args.model.model_max_length)
            val_dataset.create_inputs(dr, val_inputs, val_labels, predictor, 
                    masker, target_label=args.misc.target_label)
        else:
            train_dataset = StageOneDataset(editor_tokenizer, 
                    max_length=args.model.model_max_length)
            val_dataset = StageOneDataset(editor_tokenizer, 
                    max_length=args.model.model_max_length)
            train_dataset.create_inputs(train_inputs, train_labels, predictor, 
                    masker, target_label=args.misc.target_label)
            val_dataset.create_inputs(val_inputs, val_labels, predictor, 
                    masker, target_label=args.misc.target_label)
        logger.info("Done creating data.")

        # Write data
        logger.info(f"Writing train data to: {train_data_path}")
        train_masked_df = pd.DataFrame({
            'inputs':train_dataset.masked_strings, 
            'targets':train_dataset.targets})
        train_masked_df.to_csv(train_data_path, sep="\t")
        logger.info(f"Writing val data to: {val_data_path}")
        val_masked_df = pd.DataFrame({
            'inputs':val_dataset.masked_strings, 
            'targets':val_dataset.targets})
        val_masked_df.to_csv(val_data_path, sep="\t")

    return train_dataset, val_dataset

def get_stage_one_masker(args, predictor):
    """ Helper function for loading appropriate masker, random or grad """

    logger.info(f"Creating masker of type: {args.mask.mask_type}")
    editor_tokenizer_wrapper = PretrainedTransformerTokenizer(
            "t5-base", max_length=args.model.model_max_length)
    if args.mask.mask_type == "random":
        logger.info("Loading Random masker...")
        masker = RandomMasker(None, editor_tokenizer_wrapper, 
                args.model.model_max_length)
    elif args.mask.mask_type == "grad":
        logger.info("Loading Gradient Masker...")
        # In stage 1, if signed gradients, mask tokens pushing *towards* target
        sign_direction = 1 if "signed" in args.mask.grad_type else None 
        masker = GradientMasker(None, editor_tokenizer_wrapper, predictor, 
                args.model.model_max_length, grad_type=args.mask.grad_type,
                sign_direction=sign_direction)
    logger.info("Done.")
    return masker

def get_task_data(args, dr):
    """ Helper function for loading original data of task. 
    Calls get_inputs() function of dataset reader dr """

    if args.meta.task == 'race':
        strings = dr.get_inputs('train')
        labels = [int(s['answer_idx']) for s in strings]
    elif args.meta.task == "newsgroups" or args.meta.task == "imdb":
        strings, labels = dr.get_inputs('train', return_labels=True)
    
    string_indices = np.array(range(len(strings)))
    np.random.shuffle(string_indices)
    num_train = math.ceil(args.train.data_split_ratio * len(strings))
    train_string_indices, val_string_indices = \
            string_indices[:num_train], string_indices[num_train:]
    train_inputs = [strings[idx] for idx in train_string_indices]
    train_labels = [labels[idx] for idx in train_string_indices]
    val_inputs = [strings[idx] for idx in val_string_indices]
    val_labels = [labels[idx] for idx in val_string_indices]
    
    logger.info(f"Num train for Editor fine-tuning: {len(train_inputs)}")
    logger.info(f"Num val for Editor fine-tuning: {len(val_inputs)}")

    return train_inputs, val_inputs, train_labels, val_labels

def run_train_editor(predictor, dr, args):
    """ Runs Editor training """

    # Set random seeds for reproducibility
    torch.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True

    editor_tokenizer, editor_model = load_base_t5(
            max_length=args.model.model_max_length)
    device = get_device()
    editor_model = editor_model.to(device)

    task_dir = os.path.join(args.meta.results_dir, args.meta.task)
    stage_one_dir = os.path.join(task_dir, f"editors/{args.meta.stage1_exp}")
    data_dir = os.path.join(stage_one_dir, 'editor_train_data')
    checkpoint_dir = os.path.join(stage_one_dir, 'checkpoints')
    
    logger.info(f"Task dir: {task_dir}")
    logger.info(f"Stage one dir: {stage_one_dir}")
    logger.info(f"Stage one training data dir: {data_dir}")
    logger.info(f"Checkpoints dir: {checkpoint_dir}")

    for dir in [task_dir, data_dir, stage_one_dir, checkpoint_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # Save args
    args_path = os.path.join(stage_one_dir, "stage_one_args.json")
    write_args(args_path, args)

    masker = get_stage_one_masker(args, predictor)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': args.train.train_batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': args.train.val_batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    optim = torch.optim.Adam(params=editor_model.parameters(), \
            lr=args.train.lr)

    # Load original task data
    train_inputs, val_inputs, train_labels, val_labels = \
            get_task_data(args, dr)

    # Get datasets for Editor training
    train_dataset, val_dataset = get_datasets(predictor, dr, masker, 
            data_dir, train_inputs, val_inputs, train_labels, val_labels, 
            editor_tokenizer, args)
    train_data_loader = DataLoader(train_dataset, **train_params)
    val_data_loader = DataLoader(val_dataset, **val_params)

    # Training loop
    logger.info('Initiating Editor Fine-Tuning.')

    best_path = os.path.join(checkpoint_dir, 'best.pth')
    best_val_loss = 1000000
    for epoch in range(args.train.num_epochs):
        path = os.path.join(checkpoint_dir, f"{epoch}.pth")
        if os.path.exists(path):
            logger.info(f"Found checkpoint for epoch. Loading from: {path}")
            editor_model.load_state_dict(torch.load(path))
        else:
            train_loss = train_epoch(epoch, editor_tokenizer, editor_model, 
                    device, train_data_loader, optim)
            logger.info("Saving Editor checkpoint to: " + path)
            torch.save(editor_model.state_dict(), path)  
           
            val_loss = validate_epoch(epoch, editor_tokenizer, editor_model, 
                    device, val_data_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"Lowest loss. Saving weights to: {best_path}")
                torch.save(editor_model.state_dict(), best_path)
