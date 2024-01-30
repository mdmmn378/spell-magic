import random


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


def apply_random_errors(
    sentence: str, error_functions: list, probabilities: list
) -> str:
    num_errors = random.randint(1, len(error_functions))
    errors_to_apply = random.sample(error_functions, num_errors)
    for error_func, prob in zip(errors_to_apply, probabilities):
        sentence = error_func(sentence, prob)
    return sentence
