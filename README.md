# llm_experiment

Experiments with LLM tokenization and randomness. Results are stored in `results/`.

## Summary

| # | Experiment | Key result | Details |
|---|------------|------------|---------|
| 1 | Same sentence in EN / RU / RU in Latin letters / KK through 8 tokenizers (OpenAI, Qwen3, DeepSeek-V3, Llama 3, Gemma 3, Mistral Nemo, GLM-4.5) | English is 10–11 tokens everywhere; Russian costs 1.5–2.7x more; Russian in Latin letters 2.3–2.6x with every tokenizer; Kazakh ranges from 19 to 59 tokens depending on the tokenizer. Claude not measured yet (needs `ANTHROPIC_API_KEY`). | [Experiment 1](#experiment-1--the-same-phrase-in-different-languages-many-tokenizers) |
| 2 | Randomness: "Придумай название для кофейни" x3 in fresh sessions | Three different final picks; some names and word roots repeat (Зерно и Час, Обжарка, Крема). | [Experiment 2](#experiment-2--randomness) |
| 3 | Randomness: name for a horned skull mascot of a tavern x3 | Three different final picks, no identical full name in all lists, but the same building blocks (Рогволд, Бальтазар, Скальд, «Рог-»). | [Experiment 3](#experiment-3--randomness-name-for-a-horned-skull-mascot) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
# optional, for Claude token counts: export ANTHROPIC_API_KEY=...
```

## Experiment 1 — the same phrase in different languages, many tokenizers

Script: `exp01_tokenizer_ru_en.py` · Data (with all token ids and splits): `results/exp01_tokenizer_ru_en.json`

One sentence ("Artificial intelligence is changing how people work and learn") in four variants:

| Key        | Text                                                                 | Chars | Words |
|------------|----------------------------------------------------------------------|-------|-------|
| `en`       | Artificial intelligence is changing how people work and learn.       | 62    | 9     |
| `ru`       | Искусственный интеллект меняет то, как люди работают и учатся.       | 62    | 9     |
| `ru_latin` | Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut i uchatsya.  | 67    | 9     |
| `kk`       | Жасанды интеллект адамдардың жұмыс істеу және оқу тәсілін өзгертіп жатыр. | 73 | 10 |

Tokenizers:
- OpenAI, `tiktoken` 0.14.0: `cl100k_base` (GPT-4), `o200k_base` (GPT-4o / GPT-5 family).
- Open-weight models, `tokenizer.json` from Hugging Face: Qwen3 (`Qwen/Qwen3-8B`), DeepSeek-V3, Llama 3.x (Llama-3.2, via the ungated `unsloth` mirror), Gemma 3 (`unsloth` mirror), Mistral Nemo ("Tekken"), GLM-4.5.
- **Claude (Anthropic): not measured yet.** Claude's tokenizer is not published; the only way is the `count_tokens` API, which needs `ANTHROPIC_API_KEY`. The code is already in the script and runs automatically when the key is set (models `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-1`; the fixed per-message overhead of the API is subtracted).

### Token counts (ratio vs English in brackets)

| Tokenizer      | en | ru          | ru_latin    | kk          |
|----------------|----|-------------|-------------|-------------|
| `cl100k_base`  | 11 | 30 (2.73)   | 27 (2.45)   | 59 (5.36)   |
| `o200k_base`   | 10 | 16 (1.60)   | 26 (2.60)   | 19 (1.90)   |
| Qwen3          | 11 | 20 (1.82)   | 27 (2.45)   | 44 (4.00)   |
| DeepSeek-V3    | 10 | 20 (2.00)   | 26 (2.60)   | 40 (4.00)   |
| Llama 3.x      | 11 | 21 (1.91)   | 26 (2.36)   | 49 (4.45)   |
| Gemma 3        | 10 | 16 (1.60)   | 25 (2.50)   | 29 (2.90)   |
| Mistral Nemo   | 11 | 19 (1.73)   | 25 (2.27)   | 30 (2.73)   |
| GLM-4.5        | 11 | 16 (1.45)   | 27 (2.45)   | 46 (4.18)   |

Example splits of the Russian sentence (`<xx>` = raw byte, a fragment of a multi-byte UTF-8 character; full splits for all tokenizers are in the JSON):

```
cl100k_base: И | с | ку | с | ств | ен | ный |  ин | т | ел | л | ект |  мен | я | ет |  то | , |  как |  лю | д | и |  работ | а | ют |  и |  у | ч | ат | ся | .
o200k_base:  И | ск | ус | ственный |  интеллект |  меня | ет |  то | , |  как |  люди |  работают |  и |  уч | атся | .
gemma3:      И | скус | ственный |  интел | лект |  меня | ет |  то | , |  как |  люди |  работают |  и |  уча | тся | .
```

### Findings

- **English is stable:** 10–11 tokens in every tokenizer.
- **Russian costs 1.5–2.7x more.** The newest/most multilingual vocabularies are best: GLM-4.5, `o200k_base` and Gemma 3 need 16 tokens, the Qwen3 / DeepSeek / Llama / Mistral group 19–21, and the old `cl100k_base` 30.
- **Russian in Latin letters (translit) never helps.** 25–27 tokens everywhere (2.3–2.6x English), because the tokenizer sees non-words and chops them into fragments (`Is | kus | st | ven | ny | j`). Only for `cl100k_base` it is slightly cheaper than Cyrillic (27 vs 30); in all newer tokenizers Cyrillic is cheaper (16–21).
- **Kazakh shows the biggest spread: 19 to 59 tokens for the same sentence.** `o200k_base` (19), Gemma 3 (29) and Mistral Nemo (30) handle it well. Llama 3.x, Qwen3, DeepSeek-V3 and GLM-4.5 need 40–49 tokens (4.0–4.5x English). Llama 3 still falls back to raw bytes for the Kazakh-specific letters (`ұ`, `ә`, `қ`, `ө`, `ң`); Qwen3 has single tokens for those letters but cuts the words into single letters.
- **A bigger or newer vocabulary is not automatically better for a language:** what matters is how much Kazakh/Russian text the vocabulary was built on. Llama 3 (128k vocabulary) is better than `cl100k_base` for Russian but still poor for Kazakh.

### Caveats

- One short sentence per language, so exact numbers are anecdotal. The Kazakh sentence was written by Claude (not checked by a native speaker) and is a free translation (10 words vs 9).
- Llama 3.x and Gemma 3 were loaded from `unsloth` mirrors of the official tokenizers; the mirrors are expected to be identical, but this was not verified against the official gated repositories.
- Token counts are for the bare text, without chat templates or special tokens.

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

## Experiment 3 — randomness: name for a horned skull mascot

Raw answers: `results/exp03_skull_mascot_names.md`

Prompt `Придумай имя черепу животного с рогами, который является маскотом, висящего в баре-таверне`, asked 3 times, each in a fresh session with empty context (model: Claude Sonnet 5.5).

| | Chat 1 | Chat 2 | Chat 3 |
|---|---|---|---|
| Final pick | Рогволд Костяной | Скальд Рогатый | Бальтазар Рогатый |
| Names offered | 9 (flat list) | 19 (4 groups, incl. English) | 18 (4 groups, incl. English) |

Comparison:
- **The three final picks are all different**, and there is not a single identical full name that appears in all three lists. Across the chats there are 46 names in total.
- **But the building blocks repeat:**
  - «Рогволд» is in all three chats (Рогволд Костяной / Рогволд Безмолвный / Рогволд).
  - «Бальтазар» is in all three chats (Бальтазар Безгубый / Бальтазар Рогоносец / Бальтазар Рогатый).
  - «Скальд» in 2 of 3 (Скальд Рогатый / Скальд Безглазый); «Grimhorn» in 2 of 3 (and «Грим Рогатый»); «Рогоносец» in 2 of 3.
  - Name patterns: «Рог-» root, the suffix «Рогатый/Рогач», «Кост-» (Костоглот, Костеглав, Костяной Боря, Костян), «Хмель-/Хмельной», "Старина/Дядя".
- **Same as in experiment 2:** the model keeps coming back to the same small set of "obvious" ideas (Norse-sounding names with a horn pun), but which one it picks as the best varies. Names are assembled from the same parts in different combinations.
- **Structure is stable here:** each chat gave one favourite plus alternatives, and every chat ended with an offer to clarify the animal (deer / ram / bull / goat) and the style of the tavern. Only the number of options (9 vs 19 vs 18) and the grouping differ. Chat 1 addressed the user informally («ты»), the others formally.

Caveats:
- Same as experiment 2: runs were made as Claude Code subagents, not claude.ai chats; only 3 samples; sampling temperature was not controlled.
- «Overlap» was judged by hand on word roots, not with a formal metric.
