# Title:        wordnet-cmp
# Description:  A plugin to help users Define, Use, and Research words.
# Last Change:  2nd November 2024
# Maintainer:   klebster2 <https://github.com/klebster2>
# Imports Python modules to be used by the plugin.
import subprocess
import sys

def install(package: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def query() -> :
    cword = vim.eval("expand('<cword>')")  # type:ignore
    # Strip and clean the word
    cword = cword.strip()
    assert isinstance(cword, str)
    # TODO: add cleaning logic here 
    return (cword)
