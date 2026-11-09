# llm_experiment

Experiments with LLM tokenization. Results are stored in `results/`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install tiktoken   # Linux/macOS: .venv/bin/python
```

## Experiment 1 — the same phrase in different languages

Script: `exp01_tokenizer_ru_en.py` · Data: `results/exp01_tokenizer_ru_en.json`

One sentence ("Artificial intelligence is changing how people work and learn") in four variants:

| Key        | Text                                                                 | Chars | Words |
|------------|----------------------------------------------------------------------|-------|-------|
| `en`       | Artificial intelligence is changing how people work and learn.       | 62    | 9     |
| `ru`       | Искусственный интеллект меняет то, как люди работают и учатся.       | 62    | 9     |
| `ru_latin` | Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut i uchatsya.  | 67    | 9     |
| `kk`       | Жасанды интеллект адамдардың жұмыс істеу және оқу тәсілін өзгертіп жатыр. | 73 | 10 |

Tokenizers: `tiktoken` 0.14.0 (OpenAI BPE encodings).

### Token counts

| Variant    | `cl100k_base` tokens | ratio vs EN | `o200k_base` tokens | ratio vs EN |
|------------|----------------------|-------------|---------------------|-------------|
| `en`       | 11                   | 1.00        | 10                  | 1.00        |
| `ru`       | 30                   | 2.73        | 16                  | 1.60        |
| `ru_latin` | 27                   | 2.45        | 26                  | 2.60        |
| `kk`       | 59                   | 5.36        | 19                  | 1.90        |

Characters per token: `cl100k_base` — en 5.64, ru 2.07, ru_latin 2.48, kk 1.24; `o200k_base` — en 6.20, ru 3.88, ru_latin 2.58, kk 3.84.

### Token splits

`<xx>` is a raw byte: the token is only a fragment of a multi-byte UTF-8 character.

```
cl100k en:       Art | ificial |  intelligence |  is |  changing |  how |  people |  work |  and |  learn | .
cl100k ru:       И | с | ку | с | ств | ен | ный |  ин | т | ел | л | ект |  мен | я | ет |  то | , |  как |  лю | д | и |  работ | а | ют |  и |  у | ч | ат | ся | .
cl100k ru_latin: Is | kus | st | ven | ny | j |  int | elle | kt |  men | ya | et |  to | , |  kak |  ly | udi |  rab | ot | ay | ut |  i |  u | ch | ats | ya | .
cl100k kk:       <d0> | <96> | ас | анд | ы |  ин | т | ел | л | ект |  а | д | ам | д | ар | д | ы | <d2> | <a3> |  ж | <d2> | <b1> | м | ы | с |  і | ст | е | у |  ж | <d3> | <99> | не |  о | <d2> | <9b> | у |  т | <d3> | <99> | с | і | л | і | н |   | <d3> | <a9> | з | г | ер | т | і | п |  ж | ат | ы | р | .

o200k  en:       Artificial |  intelligence |  is |  changing |  how |  people |  work |  and |  learn | .
o200k  ru:       И | ск | ус | ственный |  интеллект |  меня | ет |  то | , |  как |  люди |  работают |  и |  уч | атся | .
o200k  ru_latin: Is | kus | st | ven | ny | j |  int | elle | kt |  men | ya | et |  to | , |  kak |  ly | udi |  rab | ot | ay | ut |  i |  uch | ats | ya | .
o200k  kk:       Ж | ас | анды |  интеллект |  адамдар | дың |  жұмыс |  і | сте | у |  және |  оқу |  тәс | іл | ін |  өзгер | тіп |  жатыр | .
```

### Findings

- **Russian vs English.** With `cl100k_base` Russian costs 2.7x more tokens; Russian words are cut into syllable-like pieces, sometimes single letters. The newer `o200k_base` (bigger vocabulary) narrows this to 1.6x: `интеллект`, `работают` become single tokens.
- **Russian in Latin letters (translit).** It does *not* help: 27 / 26 tokens, about as bad as Cyrillic under `cl100k_base` and clearly worse than Cyrillic under `o200k_base` (26 vs 16). Transliterated words are not real words for the tokenizer, so they are chopped into meaningless fragments (`Is | kus | st | ven | ny | j`). It is also longer in characters (67 vs 62).
- **Kazakh.** The worst case under `cl100k_base`: 59 tokens (5.4x English), 1.24 chars/token. Kazakh-specific letters (`ұ`, `ң`, `ө`, `ә`, `қ`, `ғ`) are not in the vocabulary, so they fall apart into raw bytes (`<d2> | <b1>`): one letter costs 2 tokens. `o200k_base` fixes most of this (19 tokens, 1.9x), with letters like `ұ`, `қ`, `ө` now covered.
- **Takeaway.** The tokenizer vocabulary matters more than the language itself: the gap between languages shrank a lot between `cl100k_base` and `o200k_base`, and the Kazakh result went from 59 to 19 tokens.

### Caveats

- One short sentence per language, so exact numbers are anecdotal; the Kazakh sentence was written by Claude (not checked by a native speaker) and is a free translation, not a word-for-word one (10 words vs 9).
- These are OpenAI tokenizers, not Claude's. Claude's counts will differ.

## Experiment 2 — randomness

Raw answers: `results/exp02_coffee_names.md`

Prompt `Придумай название для кофейни`, asked 3 times, each in a fresh session with empty context (model: Claude Sonnet 5.5).

| | Chat 1 | Chat 2 | Chat 3 |
|---|---|---|---|
| Format | 8 numbered names + reasons | 15 names in 4 categories | 8 numbered names + reasons |
| Final pick | «Зерно и Час» | «Третья чашка» (top-3 list) | «Крема» |

Comparison:
- **The three final picks are all different**, so the "answer" is not stable across runs.
- **Exact repeats:** «Зерно и Час» (chats 1 and 3, both as item 1) and «Обжарка №7» (chats 1 and 3). 29 unique names out of 31 total.
- **Near repeats:** «Крем» / «Крема» appears in all three chats, «Обжарка» in all three (with and without «№7»). The same neighbourhood of ideas (grain, roast, crema, time, "Ко") comes up each time: the model's high-probability region is narrow, the exact choice within it is random.
- **Structure differs too:** the number of options (8 vs 15), grouping, and the offer of follow-up help vary from run to run.
- Surprisingly the single most "typical" idea was not the one picked as best every time: «Крема» is the favourite of chat 3 but only a side option elsewhere.

Caveats:
- Runs were made as Claude Code subagents (no API key / claude.ai chat available), which carry a default Claude Code system prompt; a real fresh chat in claude.ai may behave slightly differently.
- Only 3 samples; sampling temperature is the default and was not controlled.
