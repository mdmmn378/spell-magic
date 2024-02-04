import random
from typing import Callable, List

from loguru import logger
from tqdm.notebook import tqdm

no_change_count = 0


def swap_adjacent_characters(sentence: str, probability: float) -> str:
    words = sentence.split()
    new_words = []
    for word in words:
        if len(word) > 1 and random.random() < probability:
            idx = random.randint(0, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]
        new_words.append(word)
    return " ".join(new_words)


def replace_characters(sentence: str, probability: float) -> str:
    typo_map = {
        "ক": "খ",
        "গ": "ঘ",
        "চ": "ছ",
        "জ": "ঝ",
        "ট": "ঠ",
        "ড": "ঢ",
        "ত": "থ",
        "দ": "ধ",
        "প": "ফ",
        "ব": "ভ",
        "স": "শ",
    }
    new_sentence = ""
    for char in sentence:
        if char in typo_map and random.random() < probability:
            new_sentence += typo_map[char]
        else:
            new_sentence += char
    return new_sentence


def insert_random_characters(sentence: str, probability: float) -> str:
    chars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহয়ড়ঢ়য়"
    new_sentence = ""
    for char in sentence:
        new_sentence += char
        if random.random() < probability:
            new_sentence += random.choice(chars)
    return new_sentence


def delete_random_characters(sentence: str, probability: float) -> str:
    new_sentence = ""
    for char in sentence:
        if random.random() < probability:
            continue
        new_sentence += char
    return new_sentence


def substitute_phonetically_similar_characters(
    sentence: str, probability: float
) -> str:
    phonetic_map = {
        "ক": "খ",
        "গ": "ঘ",
        "চ": "ছ",
        "জ": "ঝ",
        "ট": "ঠ",
        "ড": "ঢ",
        "ত": "থ",
        "দ": "ধ",
        "প": "ফ",
        "ব": "ভ",
        "স": "শ",
    }
    new_sentence = ""
    for char in sentence:
        if char in phonetic_map and random.random() < probability:
            new_sentence += phonetic_map[char]
        else:
            new_sentence += char
    return new_sentence


def repeat_characters(sentence: str, probability: float) -> str:
    new_sentence = ""
    for char in sentence:
        new_sentence += char
        if random.random() < probability:
            new_sentence += char
    return new_sentence


def remove_random_punctuation(sentence: str, probability: float) -> str:
    punctuations = "।,;:!?"
    new_sentence = ""
    for char in sentence:
        if char in punctuations and random.random() < probability:
            continue
        new_sentence += char
    return new_sentence


def delete_random_words(sentence: str, probability: float) -> str:
    words = sentence.split()
    new_words = [word for word in words if (random.random() > probability)]
    if len(new_words) < 3 or len(new_words) > 3:
        return sentence
    return " ".join(new_words)


def sentence_changed(sentence: str, modified_sentence: str) -> bool:
    return sentence != modified_sentence


def insert_random_spaces(sentence: str, probability: float) -> str:
    new_sentence = ""
    for char in sentence:
        new_sentence += char
        if random.random() < probability:
            new_sentence += " "
    return new_sentence


def randomly_choose_probabilities(num_probabilities: int) -> List[float]:
    probabilities = []
    for _ in range(num_probabilities):
        probabilities.append(min(random.random(), 0.1))
    return probabilities


def randomly_choose_functions() -> List[Callable]:
    error_functions = [
        swap_adjacent_characters,
        # replace_characters,
        insert_random_characters,
        delete_random_characters,
        substitute_phonetically_similar_characters,
        repeat_characters,
        remove_random_punctuation,
        delete_random_words,
        insert_random_spaces,
    ]
    how_many = random.randint(1, 3)
    return random.sample(error_functions, how_many)


def apply_random_errors(
    sentence: str,
    error_functions: list[Callable] | None = None,
    probabilities: list[float] | None = None,
) -> str:
    global no_change_count
    if error_functions is None:
        error_functions = randomly_choose_functions()
        probabilities = randomly_choose_probabilities(len(error_functions))
    modified_sentence = sentence
    for func, prob in zip(error_functions, probabilities):  # type: ignore
        modified_sentence = func(modified_sentence, prob)

    if not sentence_changed(sentence, modified_sentence):
        modified_sentence = error_functions[0](modified_sentence, 1)
        no_change_count += 1
    return modified_sentence


def apply_errors_on_many_sentences(
    sentences: List[str],
    error_functions: list[Callable] | None = None,
    probabilities: list[float] | None = None,
) -> List[str]:
    res = [
        apply_random_errors(sentence, error_functions, probabilities)
        for sentence in tqdm(sentences)
    ]
    logger.info(f"No change count: {no_change_count}")
    return res
