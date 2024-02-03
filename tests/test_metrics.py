from utils.metrics import calculate_cer, calculate_wer, recursive_flatten


def test_recursive_flatten():
    array = [1, 2, [3, 4, [5, 6], 7], 8]
    assert recursive_flatten(array) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_calculate_cer():
    assert calculate_cer("kitten", "sitting") == 0.5
    assert calculate_cer("kitten", "kitten") == 0.0


def test_calculate_wer():
    assert calculate_wer("kitten", "sitting") == 1
