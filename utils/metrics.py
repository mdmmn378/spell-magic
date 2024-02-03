import copy

import evaluate
import numpy as np
from transformers import Seq2SeqTrainer


def postprocess_text(preds: list[str], labels: list) -> tuple[list, list]:
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["compute_metrics"] = self.compute_metrics
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
        result = {"bleu": result["score"]}  # pyright: ignore
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
