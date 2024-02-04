from pytest import fixture


@fixture
def sentence():
    return "এটি একটি পরীক্ষা বাক্য"


@fixture
def probability():
    return 0.5


@fixture
def probabilities():
    return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
