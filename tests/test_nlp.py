import pytest
from nlp_portfolio.language_models import UnigramModel, NGramModel
from nlp_portfolio.hmm_viterbi import viterbi_decode
from nlp_portfolio.cky_parser import cky_parse
from nlp_portfolio.sentiment import make_classifier


def test_sentiment_classifier_learns_tiny_example():
    texts = ["wonderful movie", "great acting", "awful film", "terrible acting"]
    labels = ["pos", "pos", "neg", "neg"]
    model = make_classifier("logistic").fit(texts, labels)
    assert model.predict(["wonderful acting"])[0] == "pos"
    assert model.predict(["terrible movie"])[0] == "neg"


def test_unigram_probabilities_are_positive():
    model = UnigramModel(alpha=1).fit(["red blue", "red green"])
    assert model.probability("red") > model.probability("purple") > 0


def test_ngram_probability_positive_for_unseen_token():
    model = NGramModel(n=2, alpha=0.5).fit(["cats sleep", "dogs sleep"])
    assert model.probability(["cats"], "run") > 0


def test_viterbi_returns_most_likely_tags():
    observations = ["dogs", "run"]
    states = ["N", "V"]
    start = {"N": 0.9, "V": 0.1}
    trans = {("N", "V"): 0.9, ("N", "N"): 0.1, ("V", "N"): 0.2, ("V", "V"): 0.8}
    emit = {("N", "dogs"): 0.9, ("V", "dogs"): 0.01, ("V", "run"): 0.9, ("N", "run"): 0.01}
    path, score = viterbi_decode(observations, states, start, trans, emit)
    assert path == ["N", "V"]
    assert score < 0


def test_cky_parse_binary_grammar():
    rules = [("S", "NP", "VP")]
    lexical = [("NP", "cats"), ("VP", "sleep")]
    tree = cky_parse(["cats", "sleep"], rules, lexical)
    assert tree is not None
    assert tree.as_tuple() == ("S", ("NP", "cats"), ("VP", "sleep"))
    assert cky_parse(["cats", "sleep", "quickly"], rules, lexical) is None
