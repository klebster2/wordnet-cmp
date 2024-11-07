# wordnet-cmp

A Vim/Neovim plugin that provides WordNet-based completions through nvim-cmp, offering synonyms, hyponyms, and meronyms.

## Features

- WordNet integration with nvim-cmp for intelligent word suggestions
- Provides:
  - Synonyms (words with similar meanings)
  - Hyponyms (more specific terms)
  - Meronyms (part-of relationships)
- Configurable completion triggers
- Support for both Vim and Neovim
- Python-powered WordNet lookups

## Prerequisites

- Vim with Python3 support `+python3` or Neovim with Python3 provider
- Python 3.8 or higher
- nvim-cmp installed and configured
- Python packages:
  [`wn`](https://github.com/goodmami/wn)

## Installation

### Using [vim-plug](https://github.com/junegunn/vim-plug)

```vim
Plug 'hrsh7th/nvim-cmp'  " Required dependency
Plug 'klebster2/wordnet-cmp'
```

### Using [packer.nvim](https://github.com/wbthomason/packer.nvim)

```lua
use {
  'klebster2/wordnet-cmp',
  requires = {
    'hrsh7th/nvim-cmp'
  }
}
```

### Using [lazy.nvim](https://github.com/folke/lazy.nvim)

```lua
{
  'klebster2/wordnet-cmp',
  dependencies = {
    'hrsh7th/nvim-cmp'
  }
}
```

## Configuration

### Neovim (init.lua)

```lua
-- Set configuration options (optional)
vim.g.wn_cmp_language = 'en'  -- Default language
vim.g.wn_cmp_min_word_length = 3  -- Minimum word length to trigger completion
vim.g.wn_cmp_max_items = 50  -- Maximum number of completion items

-- Add WordNet to nvim-cmp sources
require('cmp').setup({
  sources = {
    -- Your other sources...
    { name = 'wordnet' }
  }
})
```

### Vim (vimrc)

```vim
" Set configuration options (optional)
let g:wn_cmp_language = 'en'
let g:wn_cmp_min_word_length = 3
let g:wn_cmp_max_items = 50

" Configure nvim-cmp with WordNet source
lua << EOF
require('cmp').setup({
  sources = {
    -- Your other sources...
    { name = 'wordnet' }
  }
})
EOF
```

## Usage

The plugin provides completions automatically through nvim-cmp. Start typing a word, and completion suggestions will appear, including:

- Synsets (marked with [NOUN] [VERB] [ADJECTIVE] [ADJECTIVE SATELLITE] [ADVERB])
- Hyponyms (marked with [Hyponym])
- Meronyms (marked with [Meronym])
- Synonyms (marked with [Synonym])

## Configuration Options

| Option                     | Default | Description                                         |
| -------------------------- | ------- | --------------------------------------------------- |
| `g:wn_cmp_language`        | 'en'    | WordNet language (currently only English supported) |
| `g:wn_cmp_min_word_length` | 3       | Minimum word length to trigger completion           |
| `g:wn_cmp_max_items`       | 50      | Maximum number of completion items to show          |

## Troubleshooting

### Common Issues

1. **Plugin not working**

   - Ensure Vim/Neovim has Python3 support:
     ```vim
     :echo has('python3')  " Should return 1
     ```
   - Check Python packages are installed based on the python vim is using:
     ```bash
     python -m pip list | grep -E "wn|nltk|vim-client"
     ```

2. **No completions appearing**

   - Verify nvim-cmp is properly configured
   - Check the minimum word length setting
   - Ensure the WordNet source is registered
   - Create an issue explaining the problem

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

The same as the [VIM license](https://github.com/vim/vim/blob/master/LICENSE)

## Credits

- wn python library database and resources
- nvim-cmp for the completion engine
