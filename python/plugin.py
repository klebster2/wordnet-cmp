# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  14th November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
from __future__ import annotations

import re
import subprocess
import sys
import typing as t
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Append to path local wordsense.py
sys.path.append(".")


def _pip_install(package: str = "wn"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import wn
except ImportError:
    _pip_install("wn")
    import wn

from wn import Form  # pylint: disable=wrong-import-position
from wn._db import connect  # pylint: disable=wrong-import-position
from wn._queries import (_qs, _Synset,  # pylint: disable=wrong-import-position
                         _vs)

if not any(re.findall(r"pytest|py.test", sys.argv[0])):
    try:
        import vim  # pylint: disable=import-error

        pytest_active = False
    except Exception as e:
        print("No vim module available outside vim")
        raise e
else:
    vim = None  # type: ignore
    pytest_active = True

ARTEFACT_NAME: str = ""
TARGET_LANGUAGE = "en" if (vim is None) else vim.eval("g:wn_cmp_language")  # type: ignore

for dataset_name, item in wn.config.index.items():
    if item.get("language") == TARGET_LANGUAGE:
        ARTEFACT_NAME = dataset_name + ":" + list(item["versions"].keys())[0]
        break

assert (
    ARTEFACT_NAME != ""
), f"Failed to find a Wordnet dataset for language {TARGET_LANGUAGE}"


# Setup language configuration
TARGET_LANGUAGE = "en" if vim is None else vim.eval("g:wn_cmp_language")


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


@dataclass
class WordSense:  # pylint: disable=too-many-instance-attributes
    """Represents a single sense of a word with its relations."""

    word_class: WordClass
    definition: str
    relation_type: t.Optional[str] = None
    relation_chain: t.Optional[t.Tuple[str, ...]] = None


@dataclass(frozen=True)
class RelationChain:
    """Represents a chain of semantic relations."""

    words: t.Tuple[str, ...]
    relation_types: t.Tuple[str, ...]
    final_definition: str


@dataclass
class SemanticDocument:
    """A document containing all semantic information for a word and its related terms."""

    primary_word: str
    definitions: t.Dict[str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]]
    relation_chains: t.Dict[str, t.List[RelationChain]]


_Form = tuple[str, Optional[str], Optional[str], int]  # form  # id  # script  # rowid


class CustomSynset(wn.Synset):
    """Custom Synset class that handles lexicon IDs properly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lexicon_id = None

    def _get_lexicon_ids(self) -> t.Tuple[int]:
        """Override to avoid lexicon extension lookup."""
        if self._lexicon_id is not None:
            return (self._lexicon_id,)
        return (self._lexid,)


class CustomWordnet(wn.Wordnet):
    """Custom Wordnet class with prefix search capabilities."""

    def __init__(self, lexicon: str):
        super().__init__(lexicon)
        self._lexicon = self._lexicons[0]
        self._lexid = self._lexicon._id

    def _get_lexicon_ids(self) -> t.Tuple[int]:
        """Override to return a proper set of integer IDs."""
        return (self._lexid,)

    def get_lexicon_id(self) -> int:
        """Return the lexicon ID for external use."""
        return self._lexid

    def synsets(
        self,
        form: Optional[str] = None,
        pos: Optional[str] = None,
        ili: Optional[str] = None,
        lexicon: Optional[str] = None,
        normalized: bool = False,
        search_all_forms: bool = True,
        prefix_search: bool = True,
    ) -> list[wn.Synset]:
        """Get synsets with prefix search support."""
        if form is not None and prefix_search:
            results = list(
                self.find_synsets_prefix(
                    forms=[form],
                    pos=pos,
                    ili=ili,
                    normalized=normalized,
                    search_all_forms=search_all_forms,
                )
            )

            synsets = []
            for data in results:
                synset = CustomSynset(data[0], data[1], data[2], self)
                synset._lexicon_id = self._lexid
                synsets.append(synset)
            return synsets

        # For non-prefix searches, convert regular synsets to custom ones
        original_synsets = super().synsets(
            form=form,
            pos=pos,
            ili=ili,
            lexicon=lexicon,
            normalized=normalized,
            search_all_forms=search_all_forms,
        )

        return [
            CustomSynset(s.id, s.pos, s.ili, self, _lexicon_id=self._lexid)
            for s in original_synsets
        ]

    def find_synsets_prefix(
        self,
        id: Optional[str] = None,
        forms: Sequence[str] = (),
        pos: Optional[str] = None,
        ili: Optional[str] = None,
        normalized: bool = False,
        search_all_forms: bool = False,
    ) -> Iterator[_Synset]:
        """Find synsets with prefix matching for forms."""
        conn = connect()
        params: list = []

        query_parts = [
            "SELECT DISTINCT",
            "    ss.id,",
            "    ss.pos,",
            "    (SELECT ilis.id FROM ilis WHERE ilis.rowid = ss.ili_rowid) as ili_id,",
            "    ss.lexicon_rowid,",
            "    ss.rowid",
            "FROM synsets ss",
        ]

        conditions = [f"ss.lexicon_rowid = {self._lexid}"]

        if forms:
            query_parts.extend(
                [
                    "JOIN senses s ON s.synset_rowid = ss.rowid",
                    "JOIN forms f ON f.entry_rowid = s.entry_rowid",
                ]
            )

            form_conditions = []
            for form in forms:
                form_conditions.append("LOWER(f.form) LIKE LOWER(?)")
                params.append(f"{form}%")

            conditions.append(f"({' OR '.join(form_conditions)})")

        if pos:
            conditions.append("ss.pos = ?")
            params.append(pos)

        if ili:
            conditions.append("ss.ili_rowid IN (SELECT rowid FROM ilis WHERE id = ?)")
            params.append(ili)

        query_parts.append(f"WHERE {' AND '.join(conditions)}")
        query_parts.append("ORDER BY ss.id")

        query = "\n".join(query_parts)

        return conn.execute(query, params)


class WordNetCompleter:
    """Provides word completions using WordNet semantic relations."""

    def __init__(self, wordnet: CustomWordnet) -> None:
        """Initialize with a WordNet instance."""
        self.wn = wordnet
        self._cache: t.Dict[str, SemanticDocument] = {}  # Cache for semantic documents
        self.MAX_DEPTH = 1

    def _normalize_word(self, word: str) -> str:
        """Normalize word for lookup."""
        return re.sub(r"[^\w]+", "", word.lower())


class WordNetCompleter:
    """Provides word completions with key semantic relationships."""

    def __init__(self, wordnet: CustomWordnet) -> None:
        self.wn = wordnet
        self._cache: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

    def _normalize_word(self, word: str) -> str:
        return re.sub(r"[^\w]+", "", word.lower())

    def get_completions(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get completions with definitions and semantic relations."""
        if not word or len(word) < 2:
            return []

        normalized = self._normalize_word(word)
        if normalized in self._cache:
            return self._cache[normalized]

        completions = []
        seen_words = set()

        # Get synsets with prefix matching
        synsets = self.wn.synsets(normalized, prefix_search=True)

        # Group synsets by POS for better organization
        noun_synsets = []
        verb_synsets = []
        adj_synsets = []
        adv_synsets = []

        for synset in synsets:
            if synset.pos == "n":
                noun_synsets.append(synset)
            elif synset.pos == "v":
                verb_synsets.append(synset)
            elif synset.pos in ("a", "s"):
                adj_synsets.append(synset)
            elif synset.pos == "r":
                adv_synsets.append(synset)

        def add_completion(
            word: str,
            pos: str,
            rel_type: str,
            definition: str,
            menu_label: str,
            extra_info: str = "",
        ) -> None:
            if word not in seen_words:
                seen_words.add(word)
                doc = [f"# {word} [{pos}]"]
                if definition:
                    doc.append(f"\n{definition}")
                if extra_info:
                    doc.append(f"\n{extra_info}")

                completions.append(
                    {
                        "word": word,
                        "kind": f"{pos}:{rel_type}" if rel_type != "main" else pos,
                        "menu": f"[{menu_label}]",
                        "data": {
                            "pos": pos,
                            "type": rel_type,
                            "definition": definition,
                        },
                        "documentation": {"kind": "markdown", "value": "\n".join(doc)},
                    }
                )

        # Process nouns first (most common)
        for synset in noun_synsets:
            def_text = synset.definition() or ""

            # Direct matches/lemmas
            for lemma in synset.lemmas():
                word = str(lemma)
                add_completion(word, "NOUN", "main", def_text, "N")

            # Hypernyms (broader terms)
            for hyper in synset.hypernyms():
                hyper_def = hyper.definition() or ""
                for lemma in hyper.lemmas():
                    word = str(lemma)
                    add_completion(
                        word,
                        "NOUN",
                        "broader",
                        hyper_def,
                        "N↑",
                        f"**Broader term** for: {normalized}",
                    )

            # Hyponyms (more specific terms)
            for hypo in synset.hyponyms():
                hypo_def = hypo.definition() or ""
                for lemma in hypo.lemmas():
                    word = str(lemma)
                    add_completion(
                        word,
                        "NOUN",
                        "narrower",
                        hypo_def,
                        "N↓",
                        f"**More specific term** for: {normalized}",
                    )

            # Meronyms (part-of relations)
            for mero in synset.meronyms():
                mero_def = mero.definition() or ""
                for lemma in mero.lemmas():
                    word = str(lemma)
                    add_completion(
                        word,
                        "NOUN",
                        "part",
                        mero_def,
                        "N→",
                        f"**Part of**: {normalized}",
                    )

        # Process verbs
        for synset in verb_synsets:
            def_text = synset.definition() or ""

            # Direct matches
            for lemma in synset.lemmas():
                word = str(lemma)
                add_completion(word, "VERB", "main", def_text, "V")

            # Troponyms (manner)
            for trop in synset.hyponyms():
                trop_def = trop.definition() or ""
                for lemma in trop.lemmas():
                    word = str(lemma)
                    add_completion(
                        word,
                        "VERB",
                        "manner",
                        trop_def,
                        "V↓",
                        f"**More specific way** to {normalized}",
                    )

        # Process adjectives
        for synset in adj_synsets:
            def_text = synset.definition() or ""

            # Direct matches
            for lemma in synset.lemmas():
                word = str(lemma)
                add_completion(word, "ADJ", "main", def_text, "A")

            # Similar terms
            if hasattr(synset, "similar_tos"):
                for sim in synset.similar_tos():
                    sim_def = sim.definition() or ""
                    for lemma in sim.lemmas():
                        word = str(lemma)
                        add_completion(
                            word,
                            "ADJ",
                            "similar",
                            sim_def,
                            "A≈",
                            f"**Similar to**: {normalized}",
                        )

        # Process adverbs
        for synset in adv_synsets:
            def_text = synset.definition() or ""
            for lemma in synset.lemmas():
                word = str(lemma)
                add_completion(word, "ADV", "main", def_text, "R")

        # Cache and return results
        self._cache[normalized] = completions
        return completions


def wordnet_complete(base: str) -> t.List[t.Dict[str, t.Any]]:
    """Main completion function to be called from Vim/Lua."""
    return completer.get_completions(base)


if pytest_active:
    import pytest

    @pytest.fixture
    def test_completer():  # pylint: disable=redefined-outer-name
        """Create a WordNetCompleter instance for testing."""
        return WordNetCompleter(CustomWordnet(ARTEFACT_NAME))

    def test_normalize_word(test_completer):  # pylint: disable=redefined-outer-name
        """Test word normalization."""
        assert (
            test_completer._normalize_word(  # pylint: disable=protected-access
                "Test-Word"
            )
            == "testword"
        )
        assert (
            test_completer._normalize_word("Testé")  # pylint: disable=protected-access
            == "test"
        )
        assert (
            test_completer._normalize_word(  # pylint: disable=protected-access
                "test_word"
            )
            == "test_word"
        )

    def test_empty_input(test_completer):  # pylint: disable=redefined-outer-name
        """Test handling of empty input."""
        assert not test_completer.get_completions("")
        assert not test_completer.get_completions("a")

    def test_basic_completion(test_completer):  # pylint: disable=redefined-outer-name
        """Test basic completion functionality."""
        completions = test_completer.get_completions("test")
        assert len(completions) > 0
        completion = completions[0]
        assert "word" in completion
        assert "kind" in completion
        assert "menu" in completion
        assert "documentation" in completion

    @pytest.fixture
    def better_completer():
        return WordNetCompleter(CustomWordnet(ARTEFACT_NAME))

    def test_prefix_search(better_completer):
        """Test prefix-based word search functionality."""
        # Test with "anti" prefix
        anti_completions = better_completer.get_completions("anti")
        anti_words = {completion["word"] for completion in anti_completions}
        expected_anti_words = {
            "antioxidant",
            "antibiotic",
            "antiseptic",
            "antihistamine",
            "antidote",
            "antibody",
            "antivirus",
        }
        assert any(word in anti_words for word in expected_anti_words)

        # Test with "well" prefix
        well_completions = better_completer.get_completions("well")
        well_words = {completion["word"] for completion in well_completions}
        expected_well_words = {
            "well",
            "wellness",
            "wellbeing",
            "well-being",
            "well-known",
            "well-made",
            "well-off",
        }
        assert any(word in well_words for word in expected_well_words)

        # Print found words for debugging
        print("\nFound 'anti' words:", sorted(list(anti_words)))
        print("Found 'well' words:", sorted(list(well_words)))

    def test_partial_word_completion(better_completer):
        """Test completion with partial words."""
        # Test with partial word "comput"
        comput_completions = better_completer.get_completions("comput")
        comput_words = {completion["word"] for completion in comput_completions}
        expected_comput_words = {
            "compute",
            "computer",
            "computation",
            "computing",
            "computational",
            "computerize",
        }
        assert any(word in comput_words for word in expected_comput_words)

        # Print found words for debugging
        print("\nFound 'comput' words:", sorted(list(comput_words)))

    def test_different_pos_completions(better_completer):
        """Test completions across different parts of speech."""
        # Test word that can be noun/verb/adjective
        test_completions = better_completer.get_completions("light")
        pos_types = {completion["kind"] for completion in test_completions}

        # Should find multiple parts of speech
        assert len(pos_types) > 1
        print("\nFound POS types for 'light':", sorted(list(pos_types)))
        print(
            "Found 'light' completions:",
            [(c["word"], c["kind"]) for c in test_completions[:10]],
        )

    def test_semantic_relations(better_completer):
        """Test that semantic relations are included in completions."""
        cat_completions = better_completer.get_completions("cat")
        relation_types = set()

        for completion in cat_completions:
            if "data" in completion and "type" in completion["data"]:
                relation_types.add(completion["data"]["type"])

        # Should find various semantic relations
        expected_relations = {"hypernym", "hyponym", "member_meronym", "main"}
        assert any(rel in relation_types for rel in expected_relations)

        print("\nFound relation types for 'cat':", sorted(list(relation_types)))

else:
    try:

        # Global instance of the completer
        completer = WordNetCompleter(CustomWordnet(ARTEFACT_NAME))

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
        completer = WordNetCompleter(CustomWordnet(ARTEFACT_NAME))
