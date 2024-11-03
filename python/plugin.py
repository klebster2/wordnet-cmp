# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  2nd November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
import re
import subprocess
import sys
import typing as t
from functools import lru_cache


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if not any(re.findall(r"pytest|py.test", sys.argv[0])):
    try:
        import vim  # pylint: disable=import-error
    except Exception as e:
        print("No vim module available outside vim")
        raise e
else:
    vim = None  # type: ignore

# This is the quick and dirty way to get packages installed if they are not already installed
try:
    import wn
except ImportError:
    install("wn")
    import wn

if vim is None:
    # pytest is running
    target_language = "en"
else:
    target_language = vim.eval("g:wn_cmp_language")  # type: ignore

ARTEFACT_NAME: t.Optional[str] = None
for dataset_name, item in wn.config.index.items():
    if item.get("language") == target_language:
        ARTEFACT_NAME = dataset_name + ":" + list(item["versions"].keys())[0]
        print(ARTEFACT_NAME)
        break

assert (
    ARTEFACT_NAME is not None
), f"Failed to find a Wordnet dataset for language {target_language}"


class WordNetCompleter:
    def __init__(self, artefact_name: str):
        # Initialize WordNet with the default English database
        self.wn = wn.Wordnet(artefact_name)
        self.lemmatizer = None

    def _normalize_word(self, word: str) -> str:
        """Normalize the word using lemmatizer if available."""
        word_lower = word.lower()
        # Replace any non-ASCII characters with their ASCII equivalents
        word_lower_rep1 = re.sub(r"[^\x00-\x7F]+", "", word_lower)
        # remove any non-alpha characters
        word_lower_rep2 = re.sub(r"\W+", "", word_lower_rep1)
        return word_lower_rep2

    def format_completion_item(
        self, word: str, kind: str, menu: str
    ) -> t.Dict[str, t.Any]:
        """Format a completion item for nvim-cmp."""
        return {
            "word": word,
            "kind": kind,
            "menu": menu,
            "dup": 0,
            "empty": 1,
        }

    @lru_cache(maxsize=128)
    def get_synsets(self, word: str, pos: t.Optional[str] = None) -> t.List[wn.Synset]:
        """Get all synsets for a word."""
        normalized = self._normalize_word(word)
        return self.wn.synsets(normalized, pos=pos)

    @lru_cache(maxsize=64)
    def get_synonyms(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get synonyms for a word formatted for nvim-cmp."""
        results = []
        for synset in self.get_synsets(word):  # pylint: disable=no-value-for-parameter
            if isinstance(synset, wn.Synset):
                for lemma in synset.lemmas():
                    if lemma.lower() != word.lower():
                        synset_definition: str | None = (
                            synset.definition()
                        )  # pylint: disable=unsupported-assignment-operation
                        if synset_definition is None:
                            synset_definition = ""
                        _item = self.format_completion_item(
                            lemma, "Synonym", f"[syn] {synset_definition[:50]}..."
                        )
                        if _item not in results:
                            results.append(_item)
        return results

    @lru_cache(maxsize=64)
    def get_hyponyms(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get hyponyms (more specific terms) for a word formatted for nvim-cmp."""
        results = []
        for synset in self.get_synsets(word):  # pylint: disable=no-value-for-parameter
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    hyponym_definition: str | None = (
                        hyponym.definition()
                    )  # pylint: disable=unsupported-assignment-operation
                    if hyponym_definition is None:
                        hyponym_definition = ""

                    _item = self.format_completion_item(
                        lemma, "Hyponym", f"[hyp] {hyponym_definition[:50]}..."
                    )
                    if _item not in results:
                        results.append(_item)
        return results

    @lru_cache(maxsize=64)
    def get_meronyms(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get meronyms (part-of relationships) for a word formatted for nvim-cmp."""
        results = []
        for synset in self.get_synsets(word):  # pylint: disable=no-value-for-parameter
            # Get both part and substance meronyms
            for meronym in synset.meronyms():
                for lemma in meronym.lemmas():
                    meronym_definition: str | None = (
                        meronym.definition()
                    )  # pylint: disable=unsupported-assignment-operation
                    if meronym_definition is None:
                        meronym_definition = ""
                    _item = self.format_completion_item(
                        lemma, "Meronym", f"[mer] {meronym_definition[:50]}..."
                    )
                    if _item not in results:
                        results.append(_item)
        return results

    def get_all_completions(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get all possible completions for a word."""
        if not word or len(word) < 2:
            return []

        completions = []
        completions.extend(self.get_synonyms(word))
        completions.extend(self.get_hyponyms(word))
        completions.extend(self.get_meronyms(word))
        return completions


try:
    # Global instance of the completer
    completer = WordNetCompleter(ARTEFACT_NAME)

except Exception as e:  # pylint: disable=broad-except
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "wn",
            "download",
            ARTEFACT_NAME,
        ]
    )
    # Global instance of the completer
    completer = WordNetCompleter(ARTEFACT_NAME)


def get_formatted_synset(word: str) -> t.List[t.Tuple[str, str, str]]:
    _output = []
    for w in completer.wn.words(word):
        for w0 in completer.get_synsets(w.lemma(), pos=w.pos):
            for w1 in w0.words():
                _output.append(
                    (
                        w1.lemma(),
                        w.pos.upper(),
                        w0.definition(),
                    )
                )
    return _output


def query() -> t.List[t.Tuple[str, str, str]]:
    cword = vim.eval("expand('<cword>')")  # type: ignore
    # Strip and clean the word
    cword = cword.strip()
    assert isinstance(cword, str)
    # TODO: add cleaning logic here # pylint: disable=fixme
    return get_formatted_synset(cword)


def test_lookup_word__win():
    assert ARTEFACT_NAME is not None, "artefact_name is None"
    assert (
        ARTEFACT_NAME == "oewn:2023"
    ), "Using an unexpected artefact will not work for this test"
    assert get_formatted_synset("win") == [
        ("win", "V", "be the winner in a contest or competition; be victorious"),
        ("acquire", "V", "win something through one's efforts"),
        ("gain", "V", "win something through one's efforts"),
        ("win", "V", "win something through one's efforts"),
        ("gain", "V", "obtain advantages, such as points, etc."),
        ("advance", "V", "obtain advantages, such as points, etc."),
        ("gain ground", "V", "obtain advantages, such as points, etc."),
        ("get ahead", "V", "obtain advantages, such as points, etc."),
        ("pull ahead", "V", "obtain advantages, such as points, etc."),
        ("win", "V", "obtain advantages, such as points, etc."),
        ("make headway", "V", "obtain advantages, such as points, etc."),
        ("come through", "V", "attain success or reach a desired goal"),
        ("win", "V", "attain success or reach a desired goal"),
        ("succeed", "V", "attain success or reach a desired goal"),
        ("bring home the bacon", "V", "attain success or reach a desired goal"),
        ("deliver the goods", "V", "attain success or reach a desired goal"),
        ("garner", "V", "acquire or deserve by one's efforts or actions"),
        ("win", "V", "acquire or deserve by one's efforts or actions"),
        ("earn", "V", "acquire or deserve by one's efforts or actions"),
        ("win", "N", "a victory (as in a race or other competition)"),
        ("win", "N", "something won (especially money)"),
        ("profits", "N", "something won (especially money)"),
        ("winnings", "N", "something won (especially money)"),
    ]


def wordnet_complete(base: str) -> t.List[t.Dict[str, t.Any]]:
    """Main completion function to be called from Vim."""
    # Return completion items
    return completer.get_all_completions(base) + [
        completer.format_completion_item(*s) for s in get_formatted_synset(base)
    ]


def test_wordnet_complete__win():
    assert wordnet_complete("win") == [
        {
            "word": "acquire",
            "kind": "Synonym",
            "menu": "[syn] win something through one's efforts...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain",
            "kind": "Synonym",
            "menu": "[syn] win something through one's efforts...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "advance",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain ground",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "get ahead",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pull ahead",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "make headway",
            "kind": "Synonym",
            "menu": "[syn] obtain advantages, such as points, etc....",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "come through",
            "kind": "Synonym",
            "menu": "[syn] attain success or reach a desired goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "succeed",
            "kind": "Synonym",
            "menu": "[syn] attain success or reach a desired goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "bring home the bacon",
            "kind": "Synonym",
            "menu": "[syn] attain success or reach a desired goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "deliver the goods",
            "kind": "Synonym",
            "menu": "[syn] attain success or reach a desired goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "garner",
            "kind": "Synonym",
            "menu": "[syn] acquire or deserve by one's efforts or actions...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "earn",
            "kind": "Synonym",
            "menu": "[syn] acquire or deserve by one's efforts or actions...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "profits",
            "kind": "Synonym",
            "menu": "[syn] something won (especially money)...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "winnings",
            "kind": "Synonym",
            "menu": "[syn] something won (especially money)...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "romp",
            "kind": "Hyponym",
            "menu": "[hyp] win easily...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "carry",
            "kind": "Hyponym",
            "menu": "[hyp] be successful in...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "take",
            "kind": "Hyponym",
            "menu": "[hyp] obtain by winning...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "sweep",
            "kind": "Hyponym",
            "menu": "[hyp] win an overwhelming victory in or on...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "carry",
            "kind": "Hyponym",
            "menu": "[hyp] win in an election...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "prevail",
            "kind": "Hyponym",
            "menu": "[hyp] prove superior...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "triumph",
            "kind": "Hyponym",
            "menu": "[hyp] prove superior...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "take the cake",
            "kind": "Hyponym",
            "menu": "[hyp] rank first; used often in a negative context...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "cozen",
            "kind": "Hyponym",
            "menu": "[hyp] cheat or trick...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "steal",
            "kind": "Hyponym",
            "menu": "[hyp] steal a base...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "score",
            "kind": "Hyponym",
            "menu": "[hyp] gain points in a game...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "rack up",
            "kind": "Hyponym",
            "menu": "[hyp] gain points in a game...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "tally",
            "kind": "Hyponym",
            "menu": "[hyp] gain points in a game...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pull off",
            "kind": "Hyponym",
            "menu": "[hyp] be successful; achieve a goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "carry off",
            "kind": "Hyponym",
            "menu": "[hyp] be successful; achieve a goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "manage",
            "kind": "Hyponym",
            "menu": "[hyp] be successful; achieve a goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "negociate",
            "kind": "Hyponym",
            "menu": "[hyp] be successful; achieve a goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "bring off",
            "kind": "Hyponym",
            "menu": "[hyp] be successful; achieve a goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pass",
            "kind": "Hyponym",
            "menu": "[hyp] go unchallenged; be approved...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "clear",
            "kind": "Hyponym",
            "menu": "[hyp] go unchallenged; be approved...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "hit the jackpot",
            "kind": "Hyponym",
            "menu": "[hyp] succeed by luck...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "luck out",
            "kind": "Hyponym",
            "menu": "[hyp] succeed by luck...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "peg",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in obtaining a position...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "nail",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in obtaining a position...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "nail down",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in obtaining a position...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pass",
            "kind": "Hyponym",
            "menu": "[hyp] go successfully through a test or a selection proc...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "make it",
            "kind": "Hyponym",
            "menu": "[hyp] go successfully through a test or a selection proc...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "run",
            "kind": "Hyponym",
            "menu": "[hyp] make without a miss...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "work",
            "kind": "Hyponym",
            "menu": "[hyp] have an effect or outcome; often the one desired o...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "act",
            "kind": "Hyponym",
            "menu": "[hyp] have an effect or outcome; often the one desired o...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pan out",
            "kind": "Hyponym",
            "menu": "[hyp] be a success...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "reach",
            "kind": "Hyponym",
            "menu": "[hyp] to gain with effort...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "attain",
            "kind": "Hyponym",
            "menu": "[hyp] to gain with effort...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "accomplish",
            "kind": "Hyponym",
            "menu": "[hyp] to gain with effort...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "achieve",
            "kind": "Hyponym",
            "menu": "[hyp] to gain with effort...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "go far",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in a big way; get to the top...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "make it",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in a big way; get to the top...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "get in",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in a big way; get to the top...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "arrive",
            "kind": "Hyponym",
            "menu": "[hyp] succeed in a big way; get to the top...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "slay",
            "kind": "Hyponym",
            "menu": "[hyp] to succeed greatly...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "hit",
            "kind": "Hyponym",
            "menu": "[hyp] hit the intended target or goal...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "letter",
            "kind": "Hyponym",
            "menu": "[hyp] win an athletic letter...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "first-place finish",
            "kind": "Hyponym",
            "menu": "[hyp] a finish in first place (as in a race)...",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "V",
            "menu": "be the winner in a contest or competition; be victorious",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "acquire",
            "kind": "V",
            "menu": "win something through one's efforts",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain",
            "kind": "V",
            "menu": "win something through one's efforts",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "V",
            "menu": "win something through one's efforts",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "advance",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "gain ground",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "get ahead",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "pull ahead",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "make headway",
            "kind": "V",
            "menu": "obtain advantages, such as points, etc.",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "come through",
            "kind": "V",
            "menu": "attain success or reach a desired goal",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "V",
            "menu": "attain success or reach a desired goal",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "succeed",
            "kind": "V",
            "menu": "attain success or reach a desired goal",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "bring home the bacon",
            "kind": "V",
            "menu": "attain success or reach a desired goal",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "deliver the goods",
            "kind": "V",
            "menu": "attain success or reach a desired goal",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "garner",
            "kind": "V",
            "menu": "acquire or deserve by one's efforts or actions",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "V",
            "menu": "acquire or deserve by one's efforts or actions",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "earn",
            "kind": "V",
            "menu": "acquire or deserve by one's efforts or actions",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "N",
            "menu": "a victory (as in a race or other competition)",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "win",
            "kind": "N",
            "menu": "something won (especially money)",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "profits",
            "kind": "N",
            "menu": "something won (especially money)",
            "dup": 0,
            "empty": 1,
        },
        {
            "word": "winnings",
            "kind": "N",
            "menu": "something won (especially money)",
            "dup": 0,
            "empty": 1,
        },
    ]
