import copy

import editdistance
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer


def postprocess_text(preds: list[str], labels: list) -> tuple[list, list]:
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def recursive_flatten(nested_list: list) -> list:
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(recursive_flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / max(len(ref_words), 1)
    return wer


def calculate_cer(reference: str, hypothesis: str) -> float:
    distance = editdistance.eval(reference, hypothesis)
    cer = distance / max(len(reference), 1)
    return cer


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["compute_metrics"] = self.compute_metrics
        self.show_extra_metrics = kwargs.pop("show_extra_metrics", False)
        super().__init__(*args, **kwargs)
        metric = evaluate.load("sacrebleu")
        self.metric = metric

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # set_trace()
        decoded_preds = self.tokenizer.batch_decode(  # pyright: ignore
            preds, skip_special_tokens=True
        )

        labels = np.where(
            labels != -100, labels, self.tokenizer.pad_token_id  # pyright: ignore
        )
        decoded_labels = self.tokenizer.batch_decode(  # pyright: ignore
            labels, skip_special_tokens=True
        )

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        if not self.show_extra_metrics:
            result = {"bleu": result["score"]}  # pyright: ignore
        else:
            flattened_decoded_preds = recursive_flatten(decoded_preds)
            flattened_decoded_labels = recursive_flatten(decoded_labels)

            wer_score = [
                calculate_wer(ref, hyp)
                for ref, hyp in zip(flattened_decoded_labels, flattened_decoded_preds)
            ]
            cer_score = [
                calculate_cer(ref, hyp)
                for ref, hyp in zip(flattened_decoded_labels, flattened_decoded_preds)
            ]
            result = {
                "bleu": result["score"],  # pyright: ignore
                "wer": np.mean(wer_score),  # pyright: ignore
                "cer": np.mean(cer_score),  # pyright: ignore
            }
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)  # pyright: ignore
            for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)  # pyright: ignore
        result = {k: round(v, 4) for k, v in result.items()}
        return result


def compute_objective(metrics: dict[str, float]) -> float:
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss")
    _ = metrics.pop("epoch", None)
    _ = metrics.pop("eval_gen_len", None)
    _ = metrics.pop("eval_bleu", None)
    # Remove speed metrics
    speed_metrics = [
        m
        for m in metrics.keys()
        if m.endswith("_runtime")
        or m.endswith("_per_second")
        or m.endswith("_compilation_time")
    ]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())  # pyright: ignore
