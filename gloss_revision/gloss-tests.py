# test_gloss_rules.py
import re
import pytest
from gloss_grammar import Rule, number_regexp, alpha_regexp, lookahead_regexp

@pytest.fixture
def base_rules():
    num   = Rule("number",    number_regexp)
    alpha = Rule("alpha",     alpha_regexp)
    look  = Rule("lookahead", lookahead_regexp)
    return num, alpha, look

def test_or_combiner_pattern(base_rules):
    num, alpha, _ = base_rules
    union = num | alpha
    # you know union.pattern should be "(?:<number_re>|<alpha_re>)"
    expected = rf"(?:{number_regexp}|{alpha_regexp})"
    assert union.pattern == expected

def test_then_combiner_pattern(base_rules):
    num, alpha, look = base_rules
    word = (num | alpha).then(look)
    # and then adds the lookahead suffix
    expected = rf"(?:{number_regexp}|{alpha_regexp}){lookahead_regexp}"
    assert word.pattern == expected

@pytest.mark.parametrize("tok", [
    "1990s",    # valid number
    "IX-1p",    # valid alpha
    "MOTHERwg", # valid alpha + wg exception
])
def test_base_rules_match(tok, base_rules):
    num, alpha, look = base_rules
    # if it matches number or alpha at all, fullmatch should not error
    assert num.compile().fullmatch(tok) or alpha.compile().fullmatch(tok)

@pytest.mark.parametrize("tok", [
    "FOObar",  # lowercase letters
    "-123",    # leading dash
    "123.45.6" # malformed decimal
])
def test_reject_invalid_base(tok, base_rules):
    num, alpha, _ = base_rules
    assert not num.compile().fullmatch(tok)
    assert not alpha.compile().fullmatch(tok)

def test_word_rule_accepts_and_rejects(base_rules):
    num, alpha, look = base_rules
    word = (num | alpha).then(look).compile()
    # should accept these “whole-word” cases
    for good in ["1990s", "IX-1p", "MOTHERwg", "ETC.:2"]:
        assert word.fullmatch(good), f"Should match {good}"
    # should reject these
    for bad in ["hello", "123.45.6", "IX-1px", "MOTHERw"]:
        assert not word.fullmatch(bad), f"Should not match {bad}"
