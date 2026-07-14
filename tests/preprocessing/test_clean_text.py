from brandparadigm.preprocessing.clean_text import clean_text


def test_strips_html_tags():
    assert clean_text("<b>Great</b> product!") == "Great product!"


def test_removes_urls():
    assert clean_text("Check https://example.com for details") == "Check for details"


def test_collapses_whitespace():
    assert clean_text("too   much   \n\n whitespace") == "too much whitespace"


def test_normalizes_unicode_ligatures_and_fullwidth_chars():
    assert clean_text("ﬁle name") == "file name"
    assert clean_text("ＡＢＣ") == "ABC"


def test_non_string_input_returns_empty_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
