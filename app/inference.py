import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_CHECKPOINT = "./model_artifacts/exported_model_pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT).to(DEVICE)


def correct_text(text):
    inputs = tokenize_text(text, return_tensors="pt")
    with torch.inference_mode():
        input_ids = inputs["input_ids"].to(DEVICE)  # pyright: ignore
        outputs = MODEL.generate(input_ids=input_ids, max_length=128)

    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)


def tokenize_text(text, return_tensors):
    inputs = TOKENIZER(text, return_tensors=return_tensors)
    return inputs
