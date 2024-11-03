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

function! s:wordnetcmp()
    lua << LUA_EOF
    local cmp = require('cmp')
    local source = {}

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
        return [[\w\+]]
    end

    source.complete = function(self, params, callback)
        local line = vim.fn.getline('.')
        local original_start = vim.fn.col('.') - 1
        local start = original_start
        while start > 0 and string.match(line:sub(start, start), '%w') do
            --- Better still; ensure start matches alphanumeric by using \w as in:
            start = start - 1
        end
        local query_word = line:sub(start + 1, vim.fn.col('.') - 1)
        if #query_word < 3 then return end

        local items = vim.fn.py3eval('plugin.wordnet_complete("' .. query_word .. '")')

        -- Group items by word and POS
        local grouped_items = {}
        local seen_definitions = {}  -- Track seen definitions per word+POS

        for _, item in ipairs(items) do
            local pos = ""
            if item.kind == 'N' then
                pos = 'Noun'
            elseif item.kind == 'V' then
                pos = 'Verb'
            elseif item.kind == 'A' then
                pos = 'Adjective'
            elseif item.kind == 'S' then
                pos = 'Adjective Satellite'
            elseif item.kind == 'R' then
                pos = 'Adverb'
            else
                pos = item.kind
            end

            local key = item.word .. '_' .. pos
            seen_definitions[key] = seen_definitions[key] or {}

            -- Only process if this definition hasn't been seen before
            if not seen_definitions[key][item.menu] then
                seen_definitions[key][item.menu] = true

                if not grouped_items[key] then
                    grouped_items[key] = {
                        word = item.word,
                        pos = pos,
                        senses = {},
                        count = 0
                    }
                end
                
                grouped_items[key].count = grouped_items[key].count + 1
                table.insert(grouped_items[key].senses, {
                    sense_num = grouped_items[key].count,
                    definition = item.menu
                })
            end
        end

        -- Convert grouped items to cmp items
        local cmp_items = {}
        for _, group in pairs(grouped_items) do
            -- Combine all senses into a single documentation string
            local doc = ""
            for _, sense in ipairs(group.senses) do
                doc = doc .. string.format("%d. %s\n", sense.sense_num, sense.definition)
            end
            -- make bold the query word
            doc = string.gsub(doc, query_word, string.format("**%s**", query_word ))

            table.insert(cmp_items, {
                label = string.format("%s [%s] (%d)", group.word, group.pos, group.count),
                kind = cmp.lsp.CompletionItemKind.Text,
                documentation = {
                    kind = 'markdown',
                    value = string.format("**%s** _%s_\n\n%s", group.word, group.pos, doc)
                },
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

call s:wordnetcmp()
