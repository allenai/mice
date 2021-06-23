import logging

from allennlp.data import DatasetReader
from typing import List, Optional
from allennlp.data import DatasetReader, Instance

from overrides import overrides

from allennlp_models.mc.dataset_readers.transformer_mc import TransformerMCReader

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
from tqdm import tqdm as tqdm
import json

from src.predictors.predictor_utils import clean_text 
import os

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

@DatasetReader.register("race")
class RaceDatasetReader(DatasetReader):

    def __init__(
        self, 
        transformer_model_name: str = "roberta-large", 
        qa_length_limit: int = 128, 
        total_length_limit: int = 512,
        data_dir = "data/RACE", 
        answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3},
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        from allennlp.data.tokenizers import PretrainedTransformerTokenizer

        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False, 
            max_length=total_length_limit
        )
        from allennlp.data.token_indexers import PretrainedTransformerIndexer

        self._token_indexers = {"tokens": PretrainedTransformerIndexer(
                                                transformer_model_name)}
        self.qa_length_limit = qa_length_limit
        self.total_length_limit = total_length_limit
        self.data_dir = data_dir

        self.train_dir = os.path.join(self.data_dir, "train")
        self.dev_dir = os.path.join(self.data_dir, "dev")
        self.test_dir = os.path.join(self.data_dir, "test")

        self.answer_mapping = answer_mapping

    def get_path(self, file_path):
        if file_path == "train":
            high_dir = osp.join(self.train_dir, "high")
            middle_dir = osp.join(self.train_dir, "middle")
        elif file_path == "test":
            high_dir = osp.join(self.test_dir, "high")
            middle_dir = osp.join(self.test_dir, "middle")
        elif file_path == "dev":
            high_dir = osp.join(self.dev_dir, "high")
            middle_dir = osp.join(self.dev_dir, "middle")
        else:
            raise ValueError("Invalid value for file_path")

        path = chain(
                Path(high_dir).glob('*.txt'), 
                Path(middle_dir).glob('*.txt'))
        return path

    @overrides
    def _read(self, file_path: str):
        path = self.get_path(file_path)        

        for p in path:
            data = json.loads(p.read_text())
            iterator = enumerate(zip(data["answers"], 
                                     data["options"], 
                                     data["questions"]))
            for idx, (answer, options, question) in iterator: 
                qid = str(data["id"][:-4] + "_" + str(idx))
                yield self.text_to_instance(qid, clean_text(data["article"]), 
                        question, options, self.answer_mapping[answer])[0]

    def get_inputs(self, file_path: str):
        logger.info(f"Getting RACE task inputs from file path: {file_path}")
        path = self.get_path(file_path)
        inputs = []

        for p in path:
            data = json.loads(p.read_text())
            iterator = enumerate(zip(data["answers"], 
                                     data["options"], 
                                     data["questions"]))
            for idx, (answer, options, question) in iterator: 
                qid = str(data["id"][:-4] + "_" + str(idx))
                inp = {"id": qid, 
                        "article": clean_text(data["article"]),
                        "question": question, 
                        "options": options, 
                        "answer_idx": self.answer_mapping[answer]}
                inputs.append(inp)
        return inputs 

    @overrides
    def text_to_instance(
        self,  # type: ignore
        qid: str,
        article: str,
        question: str,
        alternatives: List[str],
        label: Optional[int] = None,
    ) -> Instance:
        # tokenize
        article = self._tokenizer.tokenize(article)
        question = self._tokenizer.tokenize(question)

        # FORMAT: article, special tokens, question, special tokens, option
        sequences = []
        article_lengths = []
        max_article_lengths = []

        for alternative in alternatives:
            alternative = self._tokenizer.tokenize(alternative)
            qa_pair = self._tokenizer.add_special_tokens(
                    question, alternative)[:self.qa_length_limit]
            length_for_article = self.total_length_limit - len(qa_pair) - \
                    self._tokenizer.num_special_tokens_for_pair()
            sequence = self._tokenizer.add_special_tokens(
                    article[:length_for_article], qa_pair)
            if len(sequence) > self.total_length_limit:
                print(len(sequence))
            assert len(sequence) <= self.total_length_limit
            sequences.append(sequence)
            article_lengths.append(len(article[:length_for_article]))
            max_article_lengths.append(length_for_article)

        # make fields
        from allennlp.data.fields import TextField

        sequences = [TextField(seq, self._token_indexers) for seq in sequences]
        from allennlp.data.fields import ListField

        sequences = ListField(sequences)

        from allennlp.data.fields import MetadataField

        fields = {
            "alternatives": sequences,
            "qid": MetadataField(qid),
        }

        if label is not None:
            if label < 0 or label >= len(sequences):
                raise ValueError("Alternative %d does not exist", label)
            from allennlp.data.fields import IndexField

            fields["correct_alternative"] = IndexField(label, sequences)

        return Instance(fields), article_lengths, max_article_lengths 
