# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  2nd November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
import subprocess
import sys


def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# This is the quick and dirty way to get packages installed if they are not already installed
try:
    import vim
except Exception as e:
    print("No vim module available outside vim")
    raise e

try:
    import wn
except ImportError:
    install("wn")
    import wn

LANGUAGE_TO_WORDNET_ARTEFACT = {
    item["language"]: dataset_name
    for dataset_name, item in wn.config.index.items()
    if item.get("language")
}

ARTEFACT_NAME = LANGUAGE_TO_WORDNET_ARTEFACT.get(vim.eval("g:wn_cmp_language"), "mul")
try:
    spec = wn.Wordnet(ARTEFACT_NAME)

except wn.Error as e:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "wn",
            "download",
            ARTEFACT_NAME,
        ]
    )
    spec = wn.Wordnet(ARTEFACT_NAME)


def query():
    cword = vim.eval("expand('<cword>')")  # type: ignore
    # Strip and clean the word
    cword = cword.strip()
    assert isinstance(cword, str)
    # TODO: add cleaning logic here
    return cword
