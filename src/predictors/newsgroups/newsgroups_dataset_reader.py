from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
from tqdm import tqdm as tqdm
import numpy as np
import math
from sklearn.datasets import fetch_20newsgroups

from src.predictors.predictor_utils import clean_text 

logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9

@DatasetReader.register("newsgroups")
class NewsgroupsDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0
        np.random.seed(self.random_seed)

    def get_data_indices(self, subset):
        np.random.seed(self.random_seed)
        if subset in ['train', 'test']:
            newsgroups_data = fetch_20newsgroups(
                    subset=subset, remove=("headers", "footers", "quotes"))
            data_indices = np.array(range(len(newsgroups_data.data))) 
        elif subset in ['train_split', 'dev_split']:
            newsgroups_data = fetch_20newsgroups(
                    subset='train', remove=("headers", "footers", "quotes"))
            data_indices = np.array(range(len(newsgroups_data.data))) 
            np.random.shuffle(data_indices)
            num_train = math.ceil(TRAIN_VAL_SPLIT_RATIO * len(data_indices))
            train_indices = data_indices[:num_train]
            val_indices = data_indices[num_train:]
            data_indices = train_indices if subset == 'train' else val_indices 
        else:
            raise ValueError("Invalid value for subset")
        return data_indices, newsgroups_data

    def get_inputs(self, subset, return_labels = False):
        np.random.seed(self.random_seed)
        data_indices, newsgroups_data = self.get_data_indices(subset)
        strings = [None] * len(data_indices)
        labels = [None] * len(data_indices)
        for i, idx in enumerate(data_indices):
            txt = newsgroups_data.data[idx]
            topic = newsgroups_data.target[idx]
            label = newsgroups_data['target_names'][topic].split(".")[0]
            txt = clean_text(txt, special_chars=["\n", "\t"])
            if len(txt) == 0 or len(label) == 0:
                strings[i] = None
                labels[i] = None
            else:
                strings[i] = txt
                labels[i] = label

        strings = [x for x in strings if x is not None]
        labels = [x for x in labels if x is not None]
        assert len(strings) == len(labels)

        if return_labels:
            return strings, labels
        return strings

    @overrides
    def _read(self, subset):
        np.random.seed(self.random_seed)
        data_indices = self.get_data_indices(subset)
        for idx in data_indices:
            txt = newsgroups_data.data[idx]
            topic = newsgroups_data.target[idx]
            label = newsgroups_data['target_names'][topic].split(".")[0]
            txt = clean_text(txt, special_chars=["\n", "\t"])
            if len(txt) == 0 or len(label) == 0:
                continue
            yield self.text_to_instance(txt, label)

    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

