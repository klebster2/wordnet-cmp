# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  10th November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
import re
import subprocess
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from wn import Form

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
class WordSense:  # pylint: disable=too-many-instance-attributes
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
    def __init__(self, artefact_name: str):
        self.wn = wn.Wordnet(artefact_name)
        self.MAX_DEPTH = 1

    def _normalize_word(self, word: str) -> str:
        """Normalize word for lookup."""
        word_lower = word.lower()
        word_lower_rep1 = re.sub(r"[^\x00-\x7F]+", "", word_lower)
        word_lower_rep2 = re.sub(r"\W+", "", word_lower_rep1)
        return word_lower_rep2

    def _get_sense_synonyms(
        self, synset: "wn.Synset"
    ) -> t.List[t.Tuple[Form, str, None]]:
        """Get all synonyms for a specific word sense with their definitions."""
        return [(lemma, synset.definition()) for lemma in synset.lemmas()]

    def _explore_synset(  # pylint: disable=too-many-branches,too-many-locals,too-many-positional-arguments
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

        # Group definitions by word
        definitions: t.Dict[str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]] = (
            defaultdict(list)
        )

        # Get synonyms for this sense
        sense_synonyms = [(lemma, synset.definition()) for lemma in synset.lemmas()]

        # Add definition and synonyms for each lemma
        for lemma in synset.lemmas():
            definitions[lemma].append((synset.definition(), sense_synonyms))  # type: ignore

        relation_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        def process_related(related_synset: "wn.Synset", relation_type: str):
            for lemma in related_synset.lemmas():
                if lemma in current_chain:
                    continue

                new_chain = current_chain + [lemma]
                new_relations = current_relations + [relation_type]

                chain = RelationChain(
                    words=tuple(new_chain),
                    relation_types=tuple(new_relations),
                    final_definition=related_synset.definition(),  # type: ignore
                )

                relation_chains[lemma].append(chain)

        if word_class == WordClass.NOUN:
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")

            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")

            for meronym in synset.meronyms():
                process_related(meronym, "member_meronym")

            for holonym in synset.holonyms():
                process_related(holonym, "member_holonym")

        elif word_class == WordClass.VERB:
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")

            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")

            for meronym in synset.meronyms():
                process_related(meronym, "member_meronym")

            for holonym in synset.holonyms():
                process_related(holonym, "member_holonym")

        elif word_class == WordClass.ADJECTIVE:
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")

            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")

            for meronym in synset.meronyms():
                process_related(meronym, "member_meronym")

            for holonym in synset.holonyms():
                process_related(holonym, "member_holonym")

        elif word_class == WordClass.ADVERB:
            # Process adverb relations if any
            # Direct hypernyms (more general terms)
            for hypernym in synset.hypernyms():
                process_related(hypernym, "hypernym")

            # Direct hyponyms (more specific terms)
            for hyponym in synset.hyponyms():
                process_related(hyponym, "hyponym")

            # All types of meronyms
            for meronym in synset.meronyms():
                process_related(meronym, "member_meronym")

            # All types of holonyms (inverse of meronyms)
            for holonym in synset.holonyms():
                process_related(holonym, "member_holonym")

        return definitions, relation_chains

    def build_semantic_document(  # pylint: disable=too-many-locals
        self, word: str, word_class: WordClass
    ) -> SemanticDocument:
        """Build a comprehensive semantic document for a word and its relations."""
        all_definitions: t.Dict[
            str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]
        ] = defaultdict(list)
        all_chains: t.Dict[str, t.List[RelationChain]] = defaultdict(list)

        # Collect definitions from all synsets
        for synset in self.wn.synsets(word, pos=word_class.value):
            definitions, chains = self._explore_synset(
                synset,
                word_class,
                current_chain=[word],  # type: ignore
                current_relations=[],  # type: ignore
                depth=0,
                seen_synsets=set(),  # type: ignore
            )

            # Merge definitions
            for word_key, defs in definitions.items():
                all_definitions[word_key].extend(defs)

            # Merge chains
            for target_word, word_chains in chains.items():
                all_chains[target_word].extend(word_chains)

        # Remove duplicates while preserving order
        deduplicated_definitions: t.Dict[
            str, t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]]
        ] = {}
        for word_key, defs in all_definitions.items():
            # Use a set to track unique definitions
            seen_defs = set()
            unique_defs = []
            for definition, synonyms in defs:
                if definition not in seen_defs:
                    seen_defs.add(definition)
                    unique_defs.append((definition, synonyms))
            deduplicated_definitions[word_key] = unique_defs

        return SemanticDocument(
            primary_word=word,
            word_class=word_class,
            definitions=deduplicated_definitions,  # pylint: disable=no-member
            relation_chains=dict(all_chains),
        )

    def _format_documentation(
        self,
        word: str,
        word_class: WordClass,
        senses: t.List[t.Tuple[str, t.List[t.Tuple[str, str]]]],
    ) -> str:
        """Format the documentation to show all senses and their synonyms."""
        doc_parts = [f"# {word} [{word_class.to_display_name()}]\n"]

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
                menu_text = f"[{doc.word_class.to_display_name()}]"

                # Create documentation with all senses and synonyms
                doc_text = self._format_documentation(
                    word, doc.word_class, tuple(senses)
                )  # type: ignore

                items.append(
                    {
                        "word": word,
                        "kind": doc.word_class.to_display_name(),
                        "menu": menu_text,
                        "data": {
                            "word_class": doc.word_class.to_display_name(),
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
                    menu_text = f"[{doc.word_class.to_display_name()}:{marker}]"

                    items.append(
                        {
                            "word": target_word,
                            "kind": f"{doc.word_class.to_display_name()}:{marker}",
                            "menu": menu_text,
                            "data": {
                                "word_class": doc.word_class.to_display_name(),
                                "type": relation_type,
                                "chain": list(chain.words),
                                "relations": list(chain.relation_types),
                                "definition": chain.final_definition,
                            },
                            "documentation": {
                                "kind": "markdown",
                                "value": (
                                    f"# {target_word} [{doc.word_class.to_display_name()}]\n\n"
                                    f"{chain.final_definition}\n\n"
                                    f"**{desc}** of: {doc.primary_word}\n"
                                    f"Chain: {' → '.join(chain.words)}"
                                ),
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
                completions.extend(self.format_completion_items(doc))  # type: ignore

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
