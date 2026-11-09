# llm_experiment

Experiments with LLM tokenization. Results are stored in `results/`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install tiktoken   # Linux/macOS: .venv/bin/python
```

## Experiment 1 — the same phrase in Russian and English

Script: `exp01_tokenizer_ru_en.py` · Data: `results/exp01_tokenizer_ru_en.json`

Phrases (62 characters, 9 words each):
- RU: «Искусственный интеллект меняет то, как люди работают и учатся.»
- EN: "Artificial intelligence is changing how people work and learn."

Tokenizers: `tiktoken` 0.14.0 (OpenAI BPE encodings).

| Encoding      | RU tokens | EN tokens | RU/EN ratio | RU chars/token | EN chars/token |
|---------------|-----------|-----------|-------------|----------------|----------------|
| `cl100k_base` | 30        | 11        | 2.73        | 2.07           | 5.64           |
| `o200k_base`  | 16        | 10        | 1.60        | 3.88           | 6.20           |

Token splits:

```
cl100k_base RU: И | с | ку | с | ств | ен | ный |  ин | т | ел | л | ект |  мен | я | ет |  то | , |  как |  лю | д | и |  работ | а | ют |  и |  у | ч | ат | ся | .
cl100k_base EN: Art | ificial |  intelligence |  is |  changing |  how |  people |  work |  and |  learn | .
o200k_base  RU: И | ск | ус | ственный |  интеллект |  меня | ет |  то | , |  как |  люди |  работают |  и |  уч | атся | .
o200k_base  EN: Artificial |  intelligence |  is |  changing |  how |  people |  work |  and |  learn | .
```

Findings:
- The same sentence costs 2.7x more tokens in Russian than in English with `cl100k_base`; Russian words are split into syllable-like fragments, sometimes down to single letters.
- The newer `o200k_base` (larger vocabulary) narrows the gap to 1.6x: whole words such as `интеллект`, `работают` become single tokens.
- English stays almost flat (11 → 10 tokens).
- Note: these are OpenAI tokenizers, not Claude's. Claude's tokenizer is different, so absolute counts will differ, but the RU > EN tendency is expected to hold.
