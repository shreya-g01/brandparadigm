from brandparadigm.preprocessing.tokenization import truncate_to_word_limit, word_count


def test_word_count():
    assert word_count("this has four words") == 4


def test_truncate_to_word_limit():
    assert truncate_to_word_limit("one two three four", 2) == "one two"


def test_truncate_beyond_length_is_noop():
    assert truncate_to_word_limit("one two", 10) == "one two"
