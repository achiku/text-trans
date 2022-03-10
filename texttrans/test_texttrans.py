import pickle
import math
import tempfile
import pytest

from collections import defaultdict

@pytest.mark.parametrize("word, prob", [
    ('pen', 0.11640052876679541),
    ('apple', 0.07209677370095303),
])
def test_simple_char_transition_prob(word, prob):
    from .texttrans import TextTrans
    tp = TextTrans()
    assert tp.prob(word) == prob


@pytest.mark.parametrize("word, tpl", [
    ('apple', [('a', 'p'), ('p', 'p'), ('p', 'l'), ('l', 'e'), ('e', '\x00')]),
    ('half', [('h', 'a'), ('a', 'l'), ('l', 'f'), ('f', '\x00')]),
])
def test_sublines_for_ngram(word, tpl):
    from .texttrans import TextTrans
    tp = TextTrans()
    t = [x for x in tp.sublines_for_ngram(word)]
    assert t == tbl

def test_load_model():
    """""
    必要なデータ構造は以下
    mat: dict(key=遷移元alphabet, val=dict(key=遷移先alphabet, val=遷移確率))
    non_pattern_prob: 元データから遷移確率が計算できなかった場合に利用する遷移確率
    ngram: 文字通り渡されたwordを何文字の組に分割するか
    """""

    d = pickle.load(open('../data/en.pki', 'rb'))
    print(d['mat']['a'])
    assert False

def test_train():
    """
    wordが1行に1つ入っているファイルを読み込み、遷移確率行列を作る
    """
    from .texttrans import TextTrans
    tp = TextTrans()
    with tempfile.NamedTemporaryFile() as f:
        tp.train('../examples/en_words.txt', f.name)
        tran_prob_matrix = pickle.load(f)
    assert tran_prob_matrix['mat']['a']['h'] == 0
    assert tran_prob_matrix['mat']['a']['d'] == -3.1416861861770706

@pytest.mark.parametrize("word, tpl", [
    ('apple', [('a', 'p'), ('p', 'p'), ('p', 'l'), ('l', 'e'), ('e', '\x00')]),
    ('half', [('h', 'a'), ('a', 'l'), ('l', 'f'), ('f', '\x00')]),
])
def test_bigram(word, tpl):
    """
    bigramを行う関数をよりシンプルに書けるか試す
    (ngramだと変数等考えなければいけないがbigramならサクッと行けるのでは)
    * https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/
    """
    from .texttrans import bigram
    r = [x for x in bigram(word)]
    assert r == tpl


def test_create_tran_prob_matrix():
    """
    遷移確率行列生成ロジックをよりシンプルに書けるか試す
    ここのロジックをしっかり理解する
    """
    from .texttrans import create_matrix, TextTrans

    prob_matrix_for_a = defaultdict()
    with tempfile.NamedTemporaryFile() as f:
        trainer = TextTrans()
        trainer.train('../examples/en_words.txt', f.name)
        tran_prob_matrix = pickle.load(f)

    filepath = '../examples/en_words.txt'
    res = create_matrix(filepath)
    assert res['matrix']['a'] == tran_prob_matrix['mat']['a']
