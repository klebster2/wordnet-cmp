# pylint: disable=missing-module-docstring
import dataclasses
import typing as t


@dataclasses.dataclass
class WordSense:
    """
    A word sense is a meaning of a word. It is a combination of the word, the
    part of speech, the definition, and the synonyms, hypernyms, hyponyms, and
    meronyms of the word sense.
    """

    lemma: str
    pos: str
    definition: str
    synonyms: t.List[t.Tuple[str, str]]  # (word, definition)
    hypernyms: t.List[t.Tuple[str, str]]  # (word, definition)
    hyponyms: t.List[t.Tuple[str, str]]  # (word, definition)
    meronyms: t.List[t.Tuple[str, str]]  # (word, definition)
