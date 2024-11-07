# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  2nd November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
import re
import subprocess
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache

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


@dataclass
class RelatedTerm:
    word: str
    definition: str
    relation_type: str


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

try:
    import wn
except ImportError:
    install("wn")
    import wn

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
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "a"
    ADVERB = "r"

    @classmethod
    def from_pos(cls, pos: str) -> "WordClass":
        pos_map = {
            "n": cls.NOUN,
            "v": cls.VERB,
            "a": cls.ADJECTIVE,
            "s": cls.ADJECTIVE,
            "r": cls.ADVERB,
        }
        return pos_map[pos.lower()]

    def to_display_name(self) -> str:
        return self.name


@dataclass
class WordSense:
    """Represents a single sense of a word with its relations."""

    lemma: str
    word_class: WordClass
    definition: str
    sense_number: int
    synonyms: t.List[t.Tuple[str, str]]  # (word, definition)
    hyponyms: t.List[t.Tuple[str, str]]  # (word, definition)
    hypernyms: t.List[t.Tuple[str, str]]  # (word, definition)
    meronyms: t.List[t.Tuple[str, str]]  # (word, definition)
    troponyms: t.List[t.Tuple[str, str]]  # (word, definition)
    similar: t.List[t.Tuple[str, str]]  # (word, definition)


@dataclass
class CompletionItem:
    """A single completion item with all necessary metadata."""

    word: str
    kind: str
    menu: str
    data: t.Dict[str, t.Any]


@dataclass
class RelationChain:
    """Represents a chain of semantic relations."""

    words: t.List[str]  # Chain of words from source to target
    relation_types: t.List[str]  # Chain of relation types
    final_definition: str  # Definition of the target word


@dataclass
class SemanticDocument:
    """A document containing all semantic information for a word and its related terms."""

    primary_word: str
    word_class: WordClass
    definitions: t.List[t.Tuple[str, str]]  # [(word, definition)]
    relation_chains: t.Dict[str, t.List[RelationChain]]  # target_word -> list of chains


import re
import subprocess
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache


class WordClass(Enum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "a"
    ADVERB = "r"

    @classmethod
    def from_pos(cls, pos: str) -> "WordClass":
        pos_map = {
            "n": cls.NOUN,
            "v": cls.VERB,
            "a": cls.ADJECTIVE,
            "s": cls.ADJECTIVE,
            "r": cls.ADVERB,
        }
        return pos_map[pos.lower()]

    def to_display_name(self) -> str:
        return self.name


@dataclass
class RelationChain:
    """Represents a chain of semantic relations."""

    words: t.List[str]  # Chain of words from source to target
    relation_types: t.List[str]  # Chain of relation types
    final_definition: str  # Definition of the target word


@dataclass
class SemanticDocument:
    """A document containing all semantic information for a word and its related terms."""

    primary_word: str
    word_class: WordClass
    definitions: t.List[t.Tuple[str, str]]  # [(word, definition)]
    relation_chains: t.Dict[str, t.List[RelationChain]]  # target_word -> list of chains


class WordNetCompleter:
    def __init__(self, artefact_name: str):
        self.wn = wn.Wordnet(artefact_name)
        self.MAX_DEPTH = 1

    def _normalize_word(self, word: str) -> str:
        word_lower = word.lower()
        word_lower_rep1 = re.sub(r"[^\x00-\x7F]+", "", word_lower)
        word_lower_rep2 = re.sub(r"\W+", "", word_lower_rep1)
        return word_lower_rep2

    def _explore_synset(
        self,
        synset: "wn.Synset",
        word_class: WordClass,
        current_chain: t.List[str],
        current_relations: t.List[str],
        depth: int = 0,
        seen_synsets: t.Optional[set] = None,
    ) -> t.Tuple[t.List[t.Tuple[str, str]], t.Dict[str, t.List[RelationChain]]]:
        """
        Recursively explore a synset and its related terms, tracking the chain of relations.
        """
        if seen_synsets is None:
            seen_synsets = set()

        if depth >= self.MAX_DEPTH or synset.id in seen_synsets:
            return [], {}

        seen_synsets.add(synset.id)

        definitions = [(lemma, synset.definition()) for lemma in synset.lemmas()]
        relation_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        def process_related(related_synset: "wn.Synset", relation_type: str):
            for lemma in related_synset.lemmas():
                new_chain = current_chain + [lemma]
                new_relations = current_relations + [relation_type]

                relation_chains[lemma].append(
                    RelationChain(
                        words=new_chain,
                        relation_types=new_relations,
                        final_definition=related_synset.definition(),
                    )
                )

                if depth < self.MAX_DEPTH - 1:
                    sub_defs, sub_chains = self._explore_synset(
                        related_synset,
                        word_class,
                        new_chain,
                        new_relations,
                        depth + 1,
                        seen_synsets,
                    )
                    definitions.extend(sub_defs)
                    for word, chains in sub_chains.items():
                        relation_chains[word].extend(chains)

        if word_class == WordClass.NOUN:
            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")
            for meronym in synset.meronyms():
                process_related(meronym, "meronym")
        elif word_class == WordClass.VERB:
            for troponym in synset.hyponyms():
                process_related(troponym, "troponym")
        elif word_class == WordClass.ADJECTIVE:
            for similar in synset.get_related():
                process_related(similar, "similar")

        return definitions, relation_chains

    @lru_cache(maxsize=128)
    def build_semantic_document(
        self, word: str, word_class: WordClass
    ) -> SemanticDocument:
        """Build a comprehensive semantic document for a word and its relations."""
        all_definitions = []
        all_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        for synset in self.wn.synsets(word, pos=word_class.value):
            definitions, chains = self._explore_synset(
                synset,
                word_class,
                current_chain=[word],
                current_relations=[],
                depth=0,
                seen_synsets=set(),
            )
            all_definitions.extend(definitions)
            for target_word, word_chains in chains.items():
                all_chains[target_word].extend(word_chains)

        return SemanticDocument(
            primary_word=word,
            word_class=word_class,
            definitions=list(set(all_definitions)),  # Remove duplicates
            relation_chains=dict(all_chains),
        )

    def format_completion_items(
        self, doc: SemanticDocument
    ) -> t.List[t.Dict[str, t.Any]]:
        """Format a semantic document into completion items with rich metadata."""
        items = []

        # Add main word definitions
        for idx, (word, definition) in enumerate(doc.definitions, 1):
            if word == doc.primary_word:
                items.append(
                    {
                        "word": word,
                        "kind": doc.word_class.to_display_name(),
                        "menu": f"[{doc.word_class.to_display_name()}:{idx}] {definition}",
                        "data": {
                            "word_class": doc.word_class.to_display_name(),
                            "sense_number": idx,
                            "type": "main",
                            "chain": [word],
                            "relations": [],
                            "definition": definition,
                        },
                    }
                )

        # Add related words with their relation chains
        relation_markers = {
            "hyponym": "spec",
            "hypernym": "gen",
            "meronym": "part",
            "troponym": "manner",
            "similar": "sim",
        }

        for target_word, chains in doc.relation_chains.items():
            for chain in chains:
                # Create relation path string
                relation_path = " → ".join(
                    f"{w}({relation_markers.get(r, r)})"
                    for w, r in zip(chain.words[:-1], chain.relation_types)
                )

                items.append(
                    {
                        "word": target_word,
                        "kind": f"{doc.word_class.to_display_name()}:{relation_markers.get(chain.relation_types[-1], 'rel')}",
                        "menu": f"[{doc.word_class.to_display_name()}] {chain.final_definition} via {relation_path}",
                        "data": {
                            "word_class": doc.word_class.to_display_name(),
                            "type": chain.relation_types[-1],
                            "chain": chain.words,
                            "relations": chain.relation_types,
                            "definition": chain.final_definition,
                        },
                    }
                )

        return items

    def get_word_completions(self, word: str) -> t.List[t.Dict[str, t.Any]]:
        """Get completions for a word with comprehensive semantic information."""
        if not word or len(word) < 2:
            return []

        normalized = self._normalize_word(word)
        completions = []

        # Process each word class
        for word_class in WordClass:
            # Build semantic document for this word class
            doc = self.build_semantic_document(normalized, word_class)
            if doc.definitions:  # Only process if we found any meanings
                # Convert to completion items
                completions.extend(self.format_completion_items(doc))

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


def wordnet_complete(base: str) -> t.List[t.Dict[str, t.Any]]:
    """Main completion function to be called from Vim/Lua."""
    return completer.get_word_completions(base)
