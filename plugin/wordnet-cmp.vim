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
        return ""
    end

    local function create_rich_documentation(word, word_class, definition, sense_synonyms)
        local doc = {}
        table.insert(doc, string.format("# %s [%s]\n", word, word_class))
        table.insert(doc, string.format("_%s_\n", definition))
        
        if sense_synonyms and #sense_synonyms > 0 then
            table.insert(doc, "\n**Synonyms:**")
            for _, syn in ipairs(sense_synonyms) do
                local syn_word = syn[1]
                table.insert(doc, string.format("- %s", syn_word))
            end
        end
        
        return table.concat(doc, "\n")
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
    -- Source file formatting in your plugin.vim
    local function format_menu_source(word_class)
        return "[" .. word_class .. "]"
    end

    local function create_rich_documentation(item)
        local doc = {}
        local word = item.word
        local word_class = item.data.word_class
        local definition = item.data.definition
        
        -- Main header
        table.insert(doc, string.format("# %s [%s]\n", word, word_class))
        
        -- Main definition
        table.insert(doc, string.format("_%s_\n", definition))
        
        -- Handle sense-specific synonyms
        if item.data.sense_synonyms and #item.data.sense_synonyms > 0 then
            table.insert(doc, "\n**Synonyms:**")
            for _, syn in ipairs(item.data.sense_synonyms) do
                local syn_word, syn_def = syn[1], syn[2]
                if syn_word ~= word then  -- Don't include the word itself as its own synonym
                    table.insert(doc, string.format("- %s: _%s_", syn_word, syn_def))
                end
            end
        end
        
        -- Handle semantic relations (meronyms, hyponyms, etc.)
        if item.data.chain and #item.data.chain > 1 then
            -- Get relation type and display name
            local relation_type = item.data.type
            local relation_display = {
                hyponym = "Types/Specific Forms",
                hypernym = "General Categories",
                meronym = "Parts/Components",
                troponym = "Ways to",
                similar = "Similar Terms",
            }
            
            local relation_name = relation_display[relation_type] or relation_type:gsub("^%l", string.upper)
            
            table.insert(doc, string.format("\n**%s:**", relation_name))
            
            -- Build relation chain display
            local chain = item.data.chain
            local chain_str = table.concat(chain, " → ")
            table.insert(doc, string.format("- Chain: %s", chain_str))
            if item.data.definition then
                table.insert(doc, string.format("- Definition: _%s_", item.data.definition))
            end
        end
        
        return table.concat(doc, "\n")
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

        -- Transform items for completion
        local result = {}
        
        for _, item in ipairs(items) do
            -- if word starts with query_word
            if item.word:find("^" .. query_word) then
                local completion_item = {
                    label = item.word,
                    kind = cmp.lsp.CompletionItemKind.Text,
                    detail = item.menu,
                    documentation = item.documentation,
                    filterText = query_word,
                    sortText = string.format("%s%s%s",
                        item.data.type == "main" and "A" or
                        item.data.type == "synonym" and "B" or "C",
                        item.data.word_class,
                        item.word
                    )
                }
                table.insert(result, completion_item)
            end
        end

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
