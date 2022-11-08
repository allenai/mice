from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, \
        T5Config, T5TokenizerFast 
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp_models.classification \
        import StanfordSentimentTreeBankDatasetReader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util
from torch import backends
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
import more_itertools as mit
import numpy as np
import torch
import textwrap
import time
import os
import sys
import logging
import argparse
import json
import difflib
from munch import Munch

# local imports
from src.predictors.imdb.imdb_dataset_reader import ImdbDatasetReader
from src.predictors.newsgroups.newsgroups_dataset_reader \
        import NewsgroupsDatasetReader
from src.predictors.race.race_dataset_reader import RaceDatasetReader
from src.masker import Masker, RandomMasker, GradientMasker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

####################################################################
######################## Arg Parsing Utils #########################
####################################################################

def get_shared_parsers():
    """ Helper function to get parsers.
    Gets parsers that are shared across stage one and stage two. """

    meta_parser = argparse.ArgumentParser()
    meta_parser.add_argument("-task", required=True, 
            help='Name of task. Currently, only RACE, IMDB, \
                    and Newsgroups are supported.', 
            choices=['race', 'imdb', 'newsgroups'])
    meta_parser.add_argument("-results_dir", default="results", 
            help='Results dir. Where to store results.')

    mask_parser = argparse.ArgumentParser()
    mask_parser.add_argument("-mask_type", default="grad", 
            choices=["grad", "random"])
    mask_parser.add_argument("-grad_type", default="normal_l1", 
            choices=["integrated_l1", "integrated_signed", "normal_l1", \
                    "normal_signed", "normal_l2", "integrated_l2"],
            help="Which gradient method to use for grad-based masking. \
                    l1/signed/l2 determine how to aggregate over the emb dim.")

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("-model_max_length", default=700, 
            help="Maximum number of tokens that Editor model can take")
    return {"meta": meta_parser, "mask": mask_parser, "model": model_parser}

def get_stage_one_parsers():
    """ Helper function to get parsers for Stage 1. """

    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("-train_batch_size", default=4, type=int)
    train_parser.add_argument("-val_batch_size", default=1, type=int)
    train_parser.add_argument("-num_epochs", default=10, type=int)
    train_parser.add_argument("-lr", default=5e-5, type=float)
    train_parser.add_argument("-seed", default=42, type=int)
    train_parser.add_argument("-data_split_ratio", default=0.75, type=float)

    misc_parser = argparse.ArgumentParser()
    misc_parser.add_argument("-target_label", default="gold", 
            choices=["gold", "pred"], 
            help="Which label to use as the target during Editor training")
    return {"train": train_parser, "misc": misc_parser} 

def get_stage_two_parsers():
    """ Helper function to get parsers for Stage 2. """

    generation_parser = argparse.ArgumentParser()
    generation_parser.add_argument("-generate_type", default="sample", 
            choices=['beam', 'sample'])
    generation_parser.add_argument("-top_k", default=30)
    generation_parser.add_argument("-top_p", default=0.95)
    generation_parser.add_argument("-length_penalty", default=1.0)
    generation_parser.add_argument("-generation_num_beams", default=15)
    generation_parser.add_argument("-num_generations", default=15)
    generation_parser.add_argument("-no_repeat_ngram_size", default=2)
    
    search_parser = argparse.ArgumentParser()
    search_parser.add_argument("-max_mask_frac", default=0.55, type=float,
            help="Maximum mask fraction")
    search_parser.add_argument("-max_edit_rounds", default=3, type=int,
            help="Maximum number of edit rounds")
    search_parser.add_argument("-max_search_levels", default=4, type=int,
            help="Maximum number of search levels")
    search_parser.add_argument("-beam_width", default=3, type=int,
            help="Beam width for beam search over edits.")
    search_parser.add_argument("-search_method", default="binary", 
            choices=["binary", "linear"], 
            help="Which kind of search method to use: binary or linear.")

    misc_parser = argparse.ArgumentParser()
    misc_parser.add_argument("-grad_pred", default="original", 
            choices=["original", "contrast"], help="Whether to take gradient \
                    with respect to the contrast or original prediction")

    return {"generation": generation_parser, 
            "search": search_parser, 
            "misc": misc_parser}

def get_parsers_by_stage(stage="stage1"):
    """ Gets parsers by stage. """

    if stage not in ["stage1", "stage2"]:
        raise ValueError(f"stage must be 'stage1' or 'stage2' but got {stage}")
    parsers = get_shared_parsers()
    if stage == "stage1":
        parsers.update(get_stage_one_parsers())
        parsers["meta"].add_argument("-stage1_exp", required=True, 
                help='Stage 1 exp name. Used to create subdir in results dir \
                        for trained Editor.')
    else:
        parsers.update(get_stage_two_parsers())
        parsers["meta"].add_argument("-editor_path", required=True, 
                help="Path to trained Editor checkpoint. Can be a directory \
                        containing 'best.pth' file OR a direct path to file \
                        containing weights (if training ended prematurely).") 
        parsers["meta"].add_argument("-stage2_exp", required=True, 
                help='Stage 2 experiment name. Used to create subdir within \
                        stage 1 directory for editing results.')
    return parsers

def get_args(stage):
    """ Gets args by stage. """

    if stage not in ["stage1", "stage2"]:
        raise ValueError(f"stage must be one of ['stage1', 'stage2'] " + \
                "but got value {stage}")
    parsers = get_parsers_by_stage(stage)
    args = {}
    extra_args = sys.argv[1:]
    for arg_subset, parser in parsers.items():
        temp_args, extra_args = parser.parse_known_args(extra_args)
        args[arg_subset] = Munch(vars(temp_args))
    assert extra_args == [], f"Unrecognized arguments supplied: {extra_args}"
    return Munch(args)

def write_args(args_path, args):
    """ Helper function to write args
    Args:
        args: list[Dict]
        args_path: str
    """
    logger.info("Writing args to: " + args_path)
    for name, sub_args in args.items():
        logger.info(f"{name} args: {sub_args}")
    f = open(args_path, "w")
    f.write(json.dumps(args, indent=4))
    f.close()

####################################################################
####################### Task Specific Utils ########################
####################################################################

def get_dataset_reader(task, predictor):
    task_options = ["imdb", "race", "newsgroups"]
    if task not in task_options:
        raise NotImplementedError(f"Task {task} not implemented; \
                must be one of {task_options}")
    if task == "imdb":
        return ImdbDatasetReader(
                token_indexers=predictor._dataset_reader._token_indexers, 
                tokenizer=predictor._dataset_reader._tokenizer)
    elif task == "race":
        return RaceDatasetReader()
    elif task == "newsgroups":
        return NewsgroupsDatasetReader(
                token_indexers=predictor._dataset_reader._token_indexers, 
                tokenizer=predictor._dataset_reader._tokenizer)

def format_classif_input(inp, label):
    return "label: " + label + ". input: " + inp 

def format_multiple_choice_input(context, question, options, answer_idx):
    formatted_str = f"question: {question} answer: choice {answer_idx}:" + \
            f"{options[answer_idx]} context: {context}"
    for option_idx, option in enumerate(options):
        formatted_str += " choice" + str(option_idx) + ": " + option
    return formatted_str

def load_predictor(task, predictor_folder="trained_predictors/"):
    task_options = ["imdb", "race", "newsgroups"]
    if task not in task_options:
        raise NotImplementedError(f"Task {task} not implemented; \
                must be one of {task_options}")
    predictor_path = os.path.join(predictor_folder, task, "model/model.tar.gz")
    if not os.path.exists(predictor_path):
        raise ValueError(f"Cannot find predictor path {predictor_path}")
    logger.info(f"Loading Predictor from: {predictor_path}")

    dr_map = {
        "imdb": ImdbDatasetReader,
        "newsgroups": NewsgroupsDatasetReader,
        "race": RaceDatasetReader,
    }

    cuda_device = 0 if torch.cuda.is_available() else -1
    predictor = Predictor.from_path(predictor_path, 
            dataset_reader_to_load=dr_map[task], 
            cuda_device=cuda_device, frozen=True)
    logger.info("Done loading predictor.")
    return predictor

####################################################################
########################### Model Utils ############################
####################################################################

def load_base_t5(max_length=700):
    t5_config = T5Config.from_pretrained("t5-base", n_positions=max_length)
    model = T5ForConditionalGeneration.from_pretrained("t5-base", 
            config=t5_config)
    tokenizer = T5TokenizerFast.from_pretrained("t5-base", truncation=True)
    return tokenizer, model

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_prob_pred(pred, label_idx):
    """ Given a prediction, gets predicted probability of label_idx. """

    for idx, prob in enumerate(pred['probs']):
        if idx == label_idx:
            return prob

def get_ints_to_labels(predictor):
    vocab = predictor._model.vocab
    ints_to_labels = vocab.get_index_to_token_vocabulary('labels')
    return ints_to_labels 

def get_labels_to_ints(predictor):
    vocab = predictor._model.vocab
    labels_to_ints = vocab.get_token_to_index_vocabulary('labels')
    return labels_to_ints

def get_predictor_tokenized(predictor, string):
    return predictor._dataset_reader._tokenizer.tokenize(string)

def add_probs(pred):
    """ Computes predicted probs from logits. """

    if 'probs' not in pred:
        if isinstance(pred['logits'], torch.Tensor):
            pred['probs'] = torch.nn.functional.softmax(pred['logits'])
        else:
            pred['probs'] = np.exp(pred['logits'])/sum(np.exp(pred['logits'])) 
    return pred

####################################################################
########################### Other Utils ############################
####################################################################

def wrap_text(text, num_indents=6, width=100):
    """ Util for pretty printing. """

    indent = "".join(['\t' for _ in range(num_indents)])
    return textwrap.fill(text, subsequent_indent = indent, width=width)

def html_highlight_diffs(orig, edited, tokenizer_wrapper=SpacyTokenizer()):
    """ Given an orig and edited inputs, mark up differences in HTML. """
    
    orig = orig.replace("<br ", "<-br ").replace(" .", ".")
    edited = edited.replace("<br ", "<-br ").replace(" .", ".")

    orig_tok = tokenizer_wrapper.tokenize(orig)
    edited_tok = tokenizer_wrapper.tokenize(edited)

    orig_text_tok = [t.text for t in orig_tok]
    edited_text_tok = [t.text for t in edited_tok]

    edited_mark_indices, num_add, num_del = get_marked_indices(orig_text_tok, 
                                                        edited_text_tok, "+")
    orig_mark_indices, num_add_2, num_del_2 = get_marked_indices(orig_text_tok, 
                                                        edited_text_tok, "-")

    marked_original = orig 
    for idx in reversed(orig_mark_indices):
        token = orig_tok[idx]
        start, end = token.idx, token.idx_end
        if start == None or end == None:
            logger.info(token, start, end)
        marked_original = marked_original[:start] + "<b>" + \
                marked_original[start:end] + "</b>" + marked_original[end:]
    
    marked_edited = edited.replace("<br />", "<-br />") 
    for idx in reversed(edited_mark_indices):
        token = edited_tok[idx]
        start, end = token.idx, token.idx_end
        if start == None or end == None:
            logger.info(token, start, end)
        marked_edited = marked_edited[:start] + "<b>" + \
                marked_edited[start:end] + "</b>" + marked_edited[end:]
    return marked_original, marked_edited

def get_marked_indices(orig_tokinal, tokenized_contrast, symbol):
    """ Helper function for html_highlight_diffs. 
    Will only return indices of words deleted or replaced (not inserted). """

    index_offset = 0
    d = difflib.Differ()
    diff = d.compare(orig_tokinal, tokenized_contrast)
    list_diff = list(diff)
    tokens, modified_tokens, indices = [], [], []
    counter = 0
    additions, deletions = 0, 0

    for token_idx, token in enumerate(list_diff):
        marker = token[0]
        word = token[2:]
        if marker == symbol:        
            tokens.append(word)
            indices.append(counter)
            counter += 1
        elif marker == " ":
            modified_tokens.append(word)
            counter += 1

        if marker == "+":
            additions += 1
        if marker == "-":
            deletions += 1
            
    return indices, additions, deletions

