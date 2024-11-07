if exists('g:loaded_wordnet_cmp')
    finish
endif
"let g:loaded_wordnet_cmp = 1

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

function! s:wordnetcmp()
    lua << LUA_EOF
    local cmp = require('cmp')

    local source = {}

    -- Helper functions
    local function format_menu_source(word_class)
        return string.format("WORDNET [%s]", word_class)
    end

    local function create_rich_documentation(word, word_class, definitions)
        local doc = {}
        table.insert(doc, string.format("# %s [%s]\n", word, word_class))
        
        for i, def in ipairs(definitions) do
            table.insert(doc, string.format("%d. _%s_", i, def))
        end
        
        return table.concat(doc, "\n\n")
    end

    source.new = function()
        return setmetatable({}, { __index = source })
    end

    source.get_trigger_characters = function()
        return {
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
        }
    end

    source.get_keyword_pattern = function()
        return [[\k\+]]
    end
    source.complete = function(self, params, callback)
        local line = vim.fn.getline('.')
        local col = vim.fn.col('.')
        local current_word_start = vim.fn.match(line:sub(1, col-1), [[\k*$]])
        local query_word = line:sub(current_word_start + 1, col - 1)
        
        if #query_word < vim.g.wn_cmp_min_word_length then
            callback({ items = {}, isIncomplete = true })
            return
        end
        
        -- Get items from Python
        local ok, items = pcall(vim.fn.py3eval, string.format('plugin.wordnet_complete("%s")', query_word))
        if not ok then
            vim.notify('Error getting completions: ' .. tostring(items), vim.log.levels.ERROR)
            callback({ items = {}, isIncomplete = true })
            return
        end

        -- Group by word and word class
        local main_senses = {}
        local related_items = {}

        -- First pass: organize main senses and related terms
        for _, item in ipairs(items) do
            if item.data.type == "main" then
                local key = string.format("%s_%s", item.word, item.data.word_class)
                if not main_senses[key] then
                    main_senses[key] = {
                        word = item.word,
                        word_class = item.data.word_class,
                        definitions = {},
                        textEdit = item.textEdit
                    }
                end
                table.insert(main_senses[key].definitions, item.data.definition)
            else
                table.insert(related_items, {
                    label = item.word,
                    kind = cmp.lsp.CompletionItemKind.Text,
                    detail = format_menu_source(item.data.word_class),
                    menu = table.concat(item.data.chain, " → "),
                    documentation = {
                        kind = 'markdown',
                        value = string.format("# %s\n\n_%s_\n\n**Chain:**\n%s",
                            item.word,
                            item.data.definition,
                            table.concat(item.data.chain, " → ")
                        )
                    },
                    filterText = query_word,
                    sortText = string.format("B%s%s",
                        item.data.word_class,
                        string.format("%03d", #(item.data.chain or {}))
                    ),
                    textEdit = item.textEdit
                })
            end
        end

        -- Create result list
        local result = {}

        -- Add main senses (separated by word class)
        for _, sense in pairs(main_senses) do
            -- Create numbered definitions list
            local def_list = {}
            for i, def in ipairs(sense.definitions) do
                table.insert(def_list, string.format("%d. _%s_", i, def))
            end
            
            table.insert(result, {
                label = sense.word,
                kind = cmp.lsp.CompletionItemKind.Class,
                detail = format_menu_source(sense.word_class),
                menu = string.format("%d definitions", #sense.definitions),
                documentation = {
                    kind = 'markdown',
                    value = string.format("# %s [%s]\n\n%s",
                        sense.word,
                        sense.word_class,
                        table.concat(def_list, "\n\n")
                    )
                },
                filterText = sense.word,
                sortText = string.format("A%s%s", sense.word, sense.word_class),
                textEdit = sense.textEdit
            })
        end

        -- Add related terms if we have any
        callback({
            items = result,
            isIncomplete = false
        })
    end

    -- Register the source
    cmp.register_source('wordnet', source.new())
LUA_EOF
endfunction

call s:wordnetcmp()
