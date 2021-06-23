import torch
from torch.utils.data import Dataset
from tqdm import tqdm 
import numpy as np
import random
import logging

# Local imports
from src.masker import MaskError
from src.utils import * 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StageOneDataset(Dataset):
    """ Dataset for training Editor models in Stage One. Creates masked inputs 
    from task training inputs. Inherits from torch.utils.data.Dataset. """

    def __init__(
            self, 
            tokenizer, 
            max_length=700, 
            masked_strings=None, 
            targets=None
        ):
        self.tokenizer = tokenizer
        self.masked_strings = masked_strings
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.masked_strings)

    def __getitem__(self, index):
        input_text = self.masked_strings[index]
        label_text = self.targets[index]

        source = self.tokenizer.batch_encode_plus([input_text], 
                truncation=True,  max_length=self.max_length, 
                pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([label_text], 
                truncation=True, max_length=self.max_length, 
                pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        eos_id = torch.LongTensor([self.tokenizer.encode(label_text)[-1]])

        return {
            'eos_id': eos_id,
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

    def create_inputs(
            self, orig_inputs, 
            orig_labels, predictor, 
            masker, target_label = "pred", 
            mask_fracs=np.arange(0.2, 0.6, 0.05), 
            mask_frac_probs=[0.125] * 8
        ):
        target_label_options = ["pred", "gold"]
        if target_label not in target_label_options:
            error_msg = f"target_label must be in {target_label_options} "
            error_msg += f"but got '{target_label}'"
            raise ValueError(error_msg)
        
        masked_strings, targets = [], []
        labels_to_ints = get_labels_to_ints(predictor) 

        num_errors = 0
        iterator = enumerate(zip(orig_inputs, orig_labels))
        for i, (orig_inp, orig_label) in tqdm(iterator, total=len(orig_inputs)):
            masker.mask_frac = np.random.choice(mask_fracs, 1, 
                    p=mask_frac_probs)[0] 

            pred = predictor.predict(orig_inp)
            pred_label = pred['label']

            label_to_use = pred_label if target_label == "pred" else orig_label
            label_idx = labels_to_ints[label_to_use]

            predictor_tokenized = get_predictor_tokenized(predictor, orig_inp)
          
            try:
                _, word_indices_to_mask, masked_input, target = \
                        masker.get_masked_string(orig_inp, label_idx, 
                                predictor_tok_end_idx=len(predictor_tokenized))
                masked_string = format_classif_input(masked_input, label_to_use) 
                masked_strings.append(masked_string)
                targets.append(target)
                
            except MaskError:
                num_errors += 1

            verbose = True if i % 500 == 0 else False

            if verbose:
                rounded_mask_frac = round(masker.mask_frac, 3)
                logger.info(wrap_text(f"Original input ({i}): " + orig_inp))
                logger.info(wrap_text(f"Mask frac: {rounded_mask_frac}"))
                logger.info(wrap_text(f"Editor input: {masked_string}"))
                logger.info(wrap_text("Editor target: " + target))

        self.masked_strings = masked_strings
        self.targets = targets

class RaceStageOneDataset(StageOneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_inputs(
            self, dr, orig_inputs, 
            orig_labels, predictor, masker, 
            mask_fracs=np.arange(0.2, 0.6, 0.05), 
            mask_frac_probs=[0.125] * 8, 
            editable_key = "article", 
            target_label = "pred"
        ):
            
        editable_keys = ["article", "question"]
        if editable_key not in editable_keys:
            raise ValueError(f"Editable key must be in {editable_keys} \
                    but got value {editable_key}")
        
        labels_to_ints = get_labels_to_ints(predictor) 
        
        num_errors = 0
        masked_strings, targets = [], []
        
        iterator = enumerate(zip(orig_inputs, orig_labels))
        for i, (orig_inp, gold_label) in tqdm(iterator, total=len(orig_inputs)):
            masker.mask_frac = np.random.choice(mask_fracs, 1, 
                    p=mask_frac_probs)[0] 

            instance, length_lst, _ = dr.text_to_instance(
                    orig_inp["id"], orig_inp["article"], 
                    orig_inp["question"], orig_inp["options"]
                    )
            options = orig_inp["options"]
            pred = predictor.predict_instance(instance)
            pred_label = int(pred['best_alternative'])

            # For RACE, label is already int, not string
            label_idx = pred_label if target_label == "pred" else gold_label 

            try:
                # Mask the article
                if editable_key == "article":
                    article_tok = get_predictor_tokenized(predictor, 
                            orig_inp["article"])
                    predictor_tok_end_idx = min(len(article_tok), 
                            length_lst[label_idx])
                    _, word_indices_to_mask, masked_article, target = \
                            masker.get_masked_string(
                                orig_inp["article"], label_idx, 
                                labeled_instance=instance, 
                                predictor_tok_end_idx=predictor_tok_end_idx
                                )
                    question = orig_inp["question"]
                    article = masked_article
               
                # Mask the question
                # TODO: Does this work? Have only tested article
                elif editable_key == "question":
                    question_tok = get_predictor_tokenized(predictor, 
                            orig_inp["question"]) 
                    predictor_tok_end_idx = length_lst[label_idx] + \
                            len(question_tok)
                    _, word_indices_to_mask, masked_question, target = \
                            masker.get_masked_string(
                                orig_inp["question"], label_idx, 
                                labeled_instance=instance, 
                                predictor_tok_start_idx=length_lst[label_idx], 
                                predictor_tok_end_idx=predictor_tok_end_idx
                            )
                    question = masked_question
                    article = orig_inp["article"]

                masked_string = format_multiple_choice_input(
                        article, question, options, label_idx)
                masked_strings.append(masked_string)
                targets.append(target)
                
            except MaskError:
                num_errors += 1

        self.masked_strings = masked_strings
        self.targets = targets
