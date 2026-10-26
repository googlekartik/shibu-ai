import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.rule_engine import match_rule


def test_greetings():
    for word in ["hi", "hello", "hey", "hello there", "hey bot"]:
        assert match_rule(word)["intent"] == "GREETING", word


def test_goodbye_thanks_help():
    assert match_rule("goodbye")["intent"] == "GOODBYE"
    assert match_rule("thanks")["intent"] == "THANKS"
    assert match_rule("help")["intent"] == "HELP"


def test_synonyms_same_intent():
    assert match_rule("hiya")["intent"] == match_rule("hello")["intent"]
