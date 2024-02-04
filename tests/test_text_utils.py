import editdistance

from utils.text import (
    apply_random_errors,
    delete_random_characters,
    delete_random_words,
    insert_random_characters,
    remove_random_punctuation,
    repeat_characters,
    replace_characters,
    substitute_phonetically_similar_characters,
    swap_adjacent_characters,
)


def edit_distance_check(original, modified, max_distance):
    return editdistance.eval(original, modified) <= max_distance


def test_swap_adjacent_characters(sentence, probability):

    swap_adjacent_characters(sentence, probability)


def test_replace_characters(sentence, probability):
    replace_characters(sentence, probability)
    assert len(sentence) > 0


def test_insert_random_characters(sentence, probability):
    modified_sentence = insert_random_characters(sentence, probability)
    assert len(modified_sentence) > 0


def test_delete_random_characters(sentence, probability):
    modified_sentence = delete_random_characters(sentence, probability)
    assert len(modified_sentence) > 0


def test_substitute_phonetically_similar_characters(sentence, probability):
    modified_sentence = substitute_phonetically_similar_characters(
        sentence, probability
    )
    assert len(modified_sentence) > 0


def test_repeat_characters(sentence, probability):
    modified_sentence = repeat_characters(sentence, probability)
    assert len(modified_sentence) > 0


def test_remove_random_punctuation(sentence, probability):
    modified_sentence = remove_random_punctuation(sentence, probability)
    assert len(modified_sentence) > 0


def test_delete_random_words(sentence, probability):
    modified_sentence = delete_random_words(sentence, probability)
    assert len(modified_sentence) > 0


def test_apply_random_errors(sentence, probabilities):
    modified_sentence = apply_random_errors(
        sentence,
        [
            insert_random_characters,
            delete_random_characters,
            substitute_phonetically_similar_characters,
            repeat_characters,
            remove_random_punctuation,
            delete_random_words,
        ],
        probabilities,
    )

    assert len(modified_sentence) > 0
