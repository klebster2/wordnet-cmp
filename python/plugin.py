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
from dataclasses import dataclass
from enum import Enum

# Append to path local wordsense.py
sys.path.append(".")


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


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

# This is the quick and dirty way to get packages installed if they are not already installed
try:
    import wn
    from wn import Form
except ImportError:
    install("wn")
    import wn
    from wn import Form

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


# Setup language configuration
TARGET_LANGUAGE = "en" if vim is None else vim.eval("g:wn_cmp_language")

# Find appropriate WordNet dataset
ARTEFACT_NAME: t.Optional[str] = None
for dataset_name, item in wn.config.index.items():
    if item.get("language") == TARGET_LANGUAGE:
        ARTEFACT_NAME = dataset_name + ":" + list(item["versions"].keys())[0]
        break

assert (
    ARTEFACT_NAME is not None
), f"Failed to find a WordNet dataset for language {TARGET_LANGUAGE}"


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
    word_class: WordClass
    definitions: t.Dict[str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]]
    relation_chains: t.Dict[str, t.List[RelationChain]]


class WordNetCompleter:
    """Provides word completions using WordNet semantic relations."""

    def __init__(self, wordnet: wn.Wordnet) -> None:
        """Initialize with a WordNet instance."""
        self.wn = wordnet
        self._seen_combinations: t.Set[t.Tuple[str, str]] = set()
        self.MAX_DEPTH = 2

    def _normalize_word(self, word: str) -> str:
        """Normalize word for lookup."""
        word_lower = word.lower()
        word_lower_rep1 = re.sub(r"[^\x00-\x7F]+", "", word_lower)
        return re.sub(r"\W+", "", word_lower_rep1)

    def _get_sense_synonyms(
        self, synset: "wn.Synset"
    ) -> t.List[t.Tuple[Form, t.Optional[str]]]:
        """Get all synonyms for a specific word sense with their definitions."""
        return [(lemma, synset.definition()) for lemma in synset.lemmas()]

    def _explore_synset(  # pylint: disable=too-many-locals,R0917
        self,
        synset: "wn.Synset",
        word_class: WordClass,
        current_chain: t.List[str],
        current_relations: t.List[str],
        depth: int = 0,
        seen_synsets: t.Optional[set] = None,
    ) -> t.Tuple[
        t.Dict[str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]],
        t.Dict[str, t.List[RelationChain]],
    ]:
        """
        Recursively explore a synset and its related terms, tracking the chain of relations.
        Returns definitions grouped by word and relation chains.
        """
        if seen_synsets is None:
            seen_synsets = set()

        if depth >= self.MAX_DEPTH or synset.id in seen_synsets:
            return {}, {}

        seen_synsets.add(synset.id)

        # Initialize defaultdict for definitions
        definitions: t.Dict[str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]] = (
            defaultdict(list)
        )
        relation_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        # Get synonyms for this sense
        sense_synonyms = [
            (str(lemma), synset.definition()) for lemma in synset.lemmas()
        ]

        # Add definition and synonyms for each lemma
        for lemma in synset.lemmas():
            lemma_str = str(lemma)
            definitions[lemma_str].append((synset.definition(), sense_synonyms))  # type: ignore

        def process_related(related_synset: "wn.Synset", relation_type: str):
            for lemma in related_synset.lemmas():
                lemma_str = str(lemma)
                if lemma_str in current_chain:
                    continue

                new_chain = current_chain + [lemma_str]
                new_relations = current_relations + [relation_type]

                chain = RelationChain(
                    words=tuple(new_chain),
                    relation_types=tuple(new_relations),
                    final_definition=related_synset.definition(),
                )

                relation_chains[lemma_str].append(chain)

                # Recursively explore if we haven't hit depth limit
                if depth < self.MAX_DEPTH:
                    sub_defs, sub_chains = self._explore_synset(
                        related_synset,
                        word_class,
                        new_chain,
                        new_relations,
                        depth + 1,
                        seen_synsets,
                    )

                    # Merge definitions and chains from recursive call
                    for word, defs in sub_defs.items():
                        definitions[word].extend(defs)
                    for word, chains in sub_chains.items():
                        relation_chains[word].extend(chains)

        # Process relations based on word class
        if (
            word_class == WordClass.NOUN  # pylint: disable=consider-using-in
            or word_class == WordClass.VERB  # pylint: disable=consider-using-in
        ):
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")
            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")
            for meronym in synset.meronyms():
                process_related(meronym, "member_meronym")
            for holonym in synset.holonyms():
                process_related(holonym, "member_holonym")

        elif word_class == WordClass.ADJECTIVE:
            for similar in getattr(synset, "similar_tos", []):
                process_related(similar, "similar")
            # Add other adjective-specific relations here if needed

        elif word_class == WordClass.ADVERB:
            # Add adverb-specific relations here if needed
            pass

        return dict(definitions), dict(relation_chains)

    def build_semantic_document(
        self, word: str, word_class: WordClass
    ) -> SemanticDocument:
        """Build a comprehensive semantic document for a word and its relations."""
        all_definitions: t.Dict[
            str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]
        ] = defaultdict(list)
        all_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        # Collect definitions from all synsets
        synsets = self.wn.synsets(word, pos=word_class.value)
        for synset in synsets:
            definitions, chains = self._explore_synset(
                synset,
                word_class,
                [word],
                [],
                depth=0,
                seen_synsets=set(),
            )

            # Merge definitions
            for word_key, defs in definitions.items():
                all_definitions[word_key].extend(defs)

            # Merge chains
            for target_word, word_chains in chains.items():
                all_chains[target_word].extend(word_chains)

        return SemanticDocument(
            primary_word=word,
            word_class=word_class,
            definitions=dict(all_definitions),
            relation_chains=dict(all_chains),
        )

    def get_completions(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get completions for a word with comprehensive semantic information."""
        if not word or len(word) < 2:
            return []

        normalized = self._normalize_word(word)
        completions = []
        self._seen_combinations.clear()

        # Process each word class
        for word_class in WordClass:
            # Build semantic document for this word class
            doc = self.build_semantic_document(normalized, word_class)
            if doc.definitions:  # Only process if we found any meanings
                # Convert to completion items
                completions.extend(self.format_completion_items(doc))

        return completions

    def _format_documentation(
        self,
        word: str,
        word_class: WordClass,
        senses: t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]],
    ) -> str:
        """Format the documentation to show all senses and their synonyms."""
        doc_parts = [f"# {word} [{word_class.display_name}]\n"]

        for idx, (definition, synonyms) in enumerate(senses, 1):
            # Add sense number and definition
            doc_parts.append(f"## Sense {idx}")
            doc_parts.append(f"{definition}\n")

            # Add synonyms for this sense, excluding the main word
            syn_items = [
                (syn, syn_def)
                for syn, syn_def in synonyms
                if syn.lower() != word.lower()
            ]
            if syn_items:
                doc_parts.append("Synonyms:")
                for syn, syn_def in syn_items:
                    doc_parts.append(f"- {syn}: {syn_def}")
                doc_parts.append("")  # Add empty line after synonyms

        return "\n".join(doc_parts)

    def format_completion_items(  # pylint: disable=too-many-locals
        self, doc: SemanticDocument
    ) -> t.List[t.Dict[str, t.Any]]:
        """Format a semantic document into completion items with rich metadata."""
        items = []
        seen_combinations = set()

        # Process each word and its senses
        for word, senses in doc.definitions.items():
            key = (word, doc.word_class.value)
            if key not in seen_combinations:
                seen_combinations.add(key)

                # Simple menu text
                menu_text = f"[{doc.word_class.display_name}]"

                # Create documentation with all senses and synonyms
                doc_text = self._format_documentation(
                    word, doc.word_class, tuple(senses)
                )  # type: ignore

                items.append(
                    {
                        "word": word,
                        "kind": doc.word_class.display_name,
                        "menu": menu_text,
                        "data": {
                            "word_class": doc.word_class.display_name,
                            "type": "main",
                            "definitions": senses,
                        },
                        "documentation": {"kind": "markdown", "value": doc_text},
                    }
                )

        # Add semantic relations
        relation_markers = {
            "hyponym": ("spec", "More specific form"),
            "hypernym": ("gen", "More general form"),
            "member_meronym": ("mem", "Member component"),
            "part_meronym": ("part", "Part component"),
            "substance_meronym": ("subst", "Made of"),
            "member_holonym": ("memof", "Has member"),
            "part_holonym": ("partof", "Part of"),
            "substance_holonym": ("substof", "Material of"),
            "troponym": ("manner", "More specific way"),
            "similar": ("sim", "Similar term"),
            "antonym": ("opp", "Opposite"),
            "entailment": ("impl", "Implies"),
        }

        for target_word, chains in doc.relation_chains.items():
            key = (target_word, doc.word_class.value)
            if key not in seen_combinations:
                seen_combinations.add(key)
                for chain in chains:
                    relation_type = chain.relation_types[-1]
                    marker, desc = relation_markers.get(
                        relation_type, ("rel", "Related term")
                    )

                    # Simple menu text for relations
                    menu_text = f"[{doc.word_class.display_name}:{marker}]"

                    items.append(
                        {
                            "word": target_word,
                            "kind": f"{doc.word_class.display_name}:{marker}",
                            "menu": menu_text,
                            "data": {
                                "word_class": doc.word_class.display_name,
                                "type": relation_type,
                                "chain": list(chain.words),
                                "relations": list(chain.relation_types),
                                "definition": chain.final_definition,
                            },
                            "documentation": {
                                "kind": "markdown",
                                "value": (
                                    f"# {target_word} [{doc.word_class.display_name}]\n\n"
                                    f"{chain.final_definition}\n\n"
                                    f"**{desc}** of: {doc.primary_word}\n"
                                    f"Chain: {' → '.join(chain.words)}"
                                ),
                            },
                        }
                    )

        return items


if pytest_active:
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
                self, word: str, pos: str  # pylint: disable=unused-argument
            ) -> t.List[MockSynset]:
                return [MockSynset()]

        return MockWordNet()

    def test_normalize_word():
        """Test word normalization."""
        test_completer = WordNetCompleter(wordnet_mock())  # type: ignore
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
            == "testword"
        )

    def test_empty_input():
        """Test handling of empty input."""
        test_completer = WordNetCompleter(wordnet_mock())  # type: ignore
        assert not test_completer.get_completions("")
        assert not test_completer.get_completions("a")

    def test_basic_completion(wordnet_mock):  # pylint: disable=redefined-outer-name
        """Test basic completion functionality."""
        test_completer = WordNetCompleter(wordnet_mock)
        completions = test_completer.get_completions("test")
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
