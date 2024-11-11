# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  11th November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
from __future__ import annotations

import re
import subprocess
import sys
import typing as t
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

# Append to path local wordsense.py
sys.path.append(".")


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
    TARGET_LANGUAGE = "en"
else:
    TARGET_LANGUAGE = vim.eval("g:wn_cmp_language")  # type: ignore

ARTEFACT_NAME: t.Optional[str] = None
for dataset_name, item in wn.config.index.items():
    if item.get("language") == TARGET_LANGUAGE:
        ARTEFACT_NAME = dataset_name + ":" + list(item["versions"].keys())[0]
        print(ARTEFACT_NAME)
        break

assert (
    ARTEFACT_NAME is not None
), f"Failed to find a Wordnet dataset for language {TARGET_LANGUAGE}"


"""
WordNet completion plugin for Vim/Neovim.
Provides word completion and documentation based on WordNet semantic relations.
"""

T = TypeVar("T")


class WordClass(Enum):
    """Supported word classes in WordNet."""

    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "a"
    ADVERB = "r"

    @classmethod
    def from_pos(cls, pos: str) -> "WordClass":
        """Convert WordNet POS tag to WordClass."""
        pos_map = {
            "n": cls.NOUN,
            "v": cls.VERB,
            "a": cls.ADJECTIVE,
            "s": cls.ADJECTIVE,  # 's' is also used for adjectives
            "r": cls.ADVERB,
        }
        return pos_map[pos.lower()]

    @property
    def display_name(self) -> str:
        """Get display name for the word class."""
        return self.name


@dataclass(frozen=True)
class CompletionMeta:
    """Metadata for a completion item."""

    word_class: WordClass
    definition: str
    relation_type: Optional[str] = None
    relation_chain: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class CompletionItem:
    """A completion item with all necessary metadata."""

    word: str
    meta: CompletionMeta
    documentation: str

    def to_dict(self) -> Dict[str, Union[str, Dict[str, str]]]:
        """Convert to dictionary format for Vim completion."""
        kind_suffix = f":{self.meta.relation_type}" if self.meta.relation_type else ""
        return {
            "word": self.word,
            "kind": f"{self.meta.word_class.display_name}{kind_suffix}",
            "menu": f"[{self.meta.word_class.display_name}]",
            "documentation": {"kind": "markdown", "value": self.documentation},
        }


class WordNetCompleter:
    """Provides word completions using WordNet semantic relations."""

    def __init__(self, wordnet: wn.Wordnet) -> None:
        """Initialize with a WordNet instance."""
        self.wn = wordnet
        self._seen_combinations: Set[Tuple[str, str]] = set()

    @staticmethod
    @lru_cache(maxsize=1024)
    def _normalize_word(word: str) -> str:
        """Normalize word for lookup, removing non-ASCII chars and special chars."""
        word_lower = word.lower()
        word_lower_rep1 = re.sub(r"[^\x00-\x7F]+", "", word_lower)
        return re.sub(r"\W+", "", word_lower_rep1)

    @lru_cache(maxsize=1024)
    def _get_synsets(self, word: str, pos: str) -> List[wn.Synset]:
        """Get all synsets for a word and POS."""
        return self.wn.synsets(word, pos=pos)

    @lru_cache(maxsize=1024)
    def _format_documentation(
        self,
        word: str,
        word_class: WordClass,
        definition: str,
        relation_type: Optional[str] = None,
        relation_chain: Optional[Tuple[str, ...]] = None,
    ) -> str:
        """Format completion documentation."""
        doc_parts = [f"# {word} [{word_class.display_name}]\n", f"{definition}\n"]

        if relation_type and relation_chain:
            doc_parts.extend(
                [
                    f"**{relation_type.replace('_', ' ').title()}** of: {relation_chain[0]}",
                    f"Chain: {' → '.join(relation_chain)}",
                ]
            )

        return "\n".join(doc_parts)

    def _process_synset(
        self,
        synset: wn.Synset,
        word_class: WordClass,
        base_word: str,
        completions: List[CompletionItem],
    ) -> None:
        """Process a single synset and add its completions."""
        # Process direct meanings
        definition = cast(str, synset.definition())
        for lemma in synset.lemmas():
            key = (lemma, word_class.value)
            if key not in self._seen_combinations:
                self._seen_combinations.add(key)
                meta = CompletionMeta(word_class=word_class, definition=definition)
                doc = self._format_documentation(lemma, word_class, definition)
                completions.append(CompletionItem(lemma, meta, doc))

        # Process relations
        for rel_type, related in [
            ("hypernym", synset.hypernyms()),
            ("hyponym", synset.hyponyms()),
            ("meronym", synset.meronyms()),
            ("holonym", synset.holonyms()),
        ]:
            for rel_synset in related:
                for lemma in rel_synset.lemmas():
                    key = (lemma, word_class.value)
                    if key not in self._seen_combinations:
                        self._seen_combinations.add(key)
                        rel_def = cast(str, rel_synset.definition())
                        chain = (base_word, lemma)
                        meta = CompletionMeta(
                            word_class=word_class,
                            definition=rel_def,
                            relation_type=rel_type,
                            relation_chain=chain,
                        )
                        doc = self._format_documentation(
                            lemma, word_class, rel_def, rel_type, chain
                        )
                        completions.append(CompletionItem(lemma, meta, doc))

    def get_completions(self, word: str) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """Get completions for a word."""
        if not word or len(word) < 2:
            return []

        normalized = self._normalize_word(word)
        completions: List[CompletionItem] = []
        self._seen_combinations.clear()

        # Process each word class
        for word_class in WordClass:
            synsets = self._get_synsets(normalized, word_class.value)
            for synset in synsets:
                self._process_synset(synset, word_class, normalized, completions)

        return [item.to_dict() for item in completions]


if not any(re.findall(r"pytest|py.test", sys.argv[0])):
    import pytest

    @pytest.fixture
    def wordnet_mock():
        """Create a mock WordNet instance for testing."""

        class MockSynset:
            def definition(self):
                return "test definition"

            def lemmas(self):
                return ["test", "example"]

            def hypernyms(self):
                return []

            def hyponyms(self):
                return []

            def meronyms(self):
                return []

            def holonyms(self):
                return []

        class MockWordNet:
            def synsets(
                self, word: str, pos: str
            ) -> List[MockSynset]:  # pylint: disable
                return [MockSynset()]

        return MockWordNet()

    def test_normalize_word():
        """Test word normalization."""
        completer = WordNetCompleter(wordnet_mock())  # type: ignore
        assert completer._normalize_word("Test-Word") == "testword"
        assert completer._normalize_word("Testé") == "test"
        assert completer._normalize_word("test_word") == "testword"

    def test_empty_input():
        """Test handling of empty input."""
        completer = WordNetCompleter(wordnet_mock())  # type: ignore
        assert completer.get_completions("") == []
        assert completer.get_completions("a") == []

    def test_basic_completion(wordnet_mock):
        """Test basic completion functionality."""
        completer = WordNetCompleter(wordnet_mock)
        completions = completer.get_completions("test")
        assert len(completions) > 0
        completion = completions[0]
        assert "word" in completion
        assert "kind" in completion
        assert "menu" in completion
        assert "documentation" in completion

else:
    pass

try:
    # Global instance of the completer
    completer = WordNetCompleter(wn.Wordnet(ARTEFACT_NAME))

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
    completer = WordNetCompleter(wn.Wordnet(ARTEFACT_NAME))


def wordnet_complete(base: str) -> t.List[t.Dict[str, t.Any]]:
    """Main completion function to be called from Vim/Lua."""
    return completer.get_completions(base)
