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
import numpy as np
import math

from src.predictors.predictor_utils import clean_text 
logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9
        
def get_label(p):
    assert "pos" in p or "neg" in p
    return "1" if "pos" in p else "0"

@DatasetReader.register("imdb")
class ImdbDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' 
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0 # numpy random seed

    def get_path(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path in ['train_split', 'dev_split']:
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
            path_lst = list(path)
            np.random.shuffle(path_lst)
            num_train_strings = math.ceil(
                    TRAIN_VAL_SPLIT_RATIO * len(path_lst))
            train_path, path_lst[:num_train_strings]
            val_path = path_lst[num_train_strings:]
            path = train_path if file_path == "train" else val_path
        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR, 'neg')
            path = chain(
                    Path(cache_dir.joinpath(pos_dir)).glob('*.txt'), 
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        elif file_path == "unlabeled":
            unsup_dir = osp.join(self.TRAIN_DIR, 'unsup')
            path = chain(Path(cache_dir.joinpath(unsup_dir)).glob('*.txt'))
        else:
            raise ValueError(f"Invalid option for file_path.")
        return path
    
    def get_inputs(self, file_path, return_labels = False):
        np.random.seed(self.random_seed)
        
        path_lst = list(self.get_path(file_path))
        strings = [None] * len(path_lst)
        labels = [None] * len(path_lst)
        for i, p in enumerate(path_lst):
            labels[i] = get_label(str(p)) 
            strings[i] = clean_text(p.read_text(), 
                                    special_chars=["<br />", "\t"])
        if return_labels:
            return strings, labels
        return strings 

    @overrides
    def _read(self, file_path):
        np.random.seed(self.random_seed)
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and \
                not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)
        path = self.get_path(file_path)
        for p in path:
            label = get_label(str(p))
            yield self.text_to_instance(
                    clean_text(p.read_text(), special_chars=["<br />", "\t"]), 
                    label)

    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)
