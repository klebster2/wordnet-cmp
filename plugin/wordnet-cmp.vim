if !has('python3')
  echo 'vim has to be compiled with +python3 to run this'
  finish
endif

if exists('g:loaded_wordnet_cmp')
    finish
endif
let g:loaded_wordnet_cmp = 1

" Set default configuration values
if !exists('g:wn_cmp_language')
    let g:wn_cmp_language = 'en'
endif

if !exists('g:wn_cmp_min_word_length')
    let g:wn_cmp_min_word_length = 3
endif

if !exists('g:wn_cmp_max_items')
    let g:wn_cmp_max_items = 50
endif

let s:plugin_root_dir = fnamemodify(resolve(expand('<sfile>:p')), ':h')

python3 << EOF
import sys
from os.path import normpath, join
import vim
plugin_root_dir = vim.eval('s:plugin_root_dir')
python_root_dir = normpath(join(plugin_root_dir, '..', 'python'))
sys.path.insert(0, python_root_dir)
import plugin
EOF

" add language, and comma separated keep keys arg
function! s:wordnetcmp()
    " Register the source with nvim-cmp
    lua << LUA_EOF
    local cmp = require('cmp')
    local source = {}

    source.new = function()
        return setmetatable({}, { __index = source })
    end

    source.get_trigger_characters = function()
        return {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        }
    end

    source.get_keyword_pattern = function()
        return [[\w\+]]
    end

    source.complete = function(self, params, callback)
        -- Call Python function to get completions
        local line = vim.fn.getline('.')
        local original_start = vim.fn.col('.') - 1
        local start = original_start
        while start > 0 and string.match(line:sub(start, start), '%S') do
            start = start - 1
        end
        local query_word = line:sub(start + 1, vim.fn.col('.') - 1)
        if #query_word < 3 then return end  --- Short input requires a lot of processing, so let's skip it.
        local items = vim.fn.py3eval('plugin.wordnet_complete(0, "' .. query_word .. '")')

        -- Convert items to nvim-cmp format
        local cached_items = {}
        ---- Count items that have the same word and POS
        local cmp_items = {}
        for _, item in ipairs(items) do
            local pos = ""
            if item.kind == 'N' then
                pos = 'NOUN'
            elseif item.kind == 'V' then
                pos = 'VERB'
            elseif item.kind == 'A' then
                pos = 'ADJECTIVE'
            elseif item.kind == 'S' then
                pos = 'ADJECTIVE SATELLITE'
            elseif item.kind == 'R' then
                pos = 'ADVERB'
            else
                pos = item.kind
            end
            cached_items[item.word .. pos] = (cached_items[item.word .. pos] or 0) + 1
            table.insert(cmp_items, {
                label = item.word .. ' [' .. pos .. '] (' .. cached_items[item.word .. pos] .. ')',
                kind = cmp.lsp.CompletionItemKind.Text,
                documentation = item.menu,
                textEdit = {
                  newText = query_word,
                  filterText = query_word,
                  range = {
                    ['start'] = {
                      line = params.context.cursor.row - 1,
                        character = original_start,
                    },
                    ['end'] = {
                        line = params.context.cursor.row - 1,
                        character = params.context.cursor.col - 1,
                    }
                  },
                },
              })
        end

        callback({ items = cmp_items, isIncomplete = false })
    end

    -- Register the source
    cmp.register_source('wordnet', source.new())
LUA_EOF
endfunction

" Autoload the setup function
call s:wordnetcmp()
