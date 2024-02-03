# Forked from huggingface
# https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/t5_tokenizer_model.py

import json
from typing import Iterator, List, Union

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# pyright: reportGeneralTypeIssues=false
class SentencePieceUnigramTokenizer(BaseTokenizer):
    def __init__(
        self,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        unk_token: Union[str, AddedToken] = "<unk>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
    ):
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
        }

        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(
                    replacement=replacement, add_prefix_space=add_prefix_space
                ),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )

        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[
                (self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"])
            ],
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given files"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

        self.add_unk_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


class TokenizerPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int,
        max_target_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        # self.tokenizer.special_tokens_map["additional_special_tokens"] = []

    def __call__(self, examples: dict[str, list]):
        inputs = examples["from"]
        targets = examples["to"]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_input_length, truncation=True
        )
        labels = self.tokenizer(
            text_target=targets, max_length=self.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
