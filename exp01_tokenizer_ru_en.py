"""Experiment 1: tokenize the same phrase in several languages with many tokenizers.

Run:  python exp01_tokenizer_ru_en.py
Writes results/exp01_tokenizer_ru_en.json and prints a summary.

Tokenizers:
  * tiktoken (OpenAI):            cl100k_base, o200k_base
  * Hugging Face tokenizer.json:  Qwen3, DeepSeek-V3, Llama 3.2, Gemma 3, Mistral Nemo, GLM-4.5
  * Anthropic (Claude) - only if ANTHROPIC_API_KEY is set; uses the free
    /v1/messages/count_tokens endpoint (Claude's tokenizer is not published).
"""
import json
import os
import sys
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PHRASES = {
    "en": "Artificial intelligence is changing how people work and learn.",
    "ru": "Искусственный интеллект меняет то, как люди работают и учатся.",
    "ru_latin": "Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut i uchatsya.",
    "kk": "Жасанды интеллект адамдардың жұмыс істеу және оқу тәсілін өзгертіп жатыр.",
}
TIKTOKEN_ENCODINGS = ["cl100k_base", "o200k_base"]  # GPT-4 / GPT-4o, GPT-5 family
HF_TOKENIZERS = {  # label -> repo with tokenizer.json (Llama/Gemma via ungated unsloth mirrors)
    "qwen3": "Qwen/Qwen3-8B",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "gemma3": "unsloth/gemma-3-1b-it",
    "mistral_nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "glm4_5": "zai-org/GLM-4.5",
}
CLAUDE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]


def show_piece(raw: bytes) -> str:
    """Text of a token; if it is a fragment of a multi-byte char, show hex bytes like <d0 b6>."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex(" ") + ">"


def _gpt2_byte_decoder():
    """Inverse of the GPT-2 byte->unicode map used by byte-level BPE tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _gpt2_byte_decoder()


def show_hf_token(tok: str) -> str:
    if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):  # sentencepiece byte fallback
        return f"<{tok[3:5].lower()}>"
    tok = tok.replace("▁", " ")  # sentencepiece space marker
    if all(ch in BYTE_DECODER for ch in tok):  # byte-level BPE
        return show_piece(bytes(BYTE_DECODER[ch] for ch in tok))
    return tok


def row(text, pieces, ids=None):
    n = len(pieces)
    return {
        "text": text,
        "chars": len(text),
        "words": len(text.split()),
        "tokens": n,
        "chars_per_token": round(len(text) / n, 2),
        "token_ids": ids,
        "token_pieces": pieces,
    }


def run_tiktoken(name):
    enc = tiktoken.get_encoding(name)
    out = {}
    for lang, text in PHRASES.items():
        ids = enc.encode(text)
        out[lang] = row(text, [show_piece(enc.decode_single_token_bytes(i)) for i in ids], ids)
    return out


def run_hf(repo):
    path = hf_hub_download(repo, "tokenizer.json")
    tk = Tokenizer.from_file(path)
    out = {}
    for lang, text in PHRASES.items():
        e = tk.encode(text, add_special_tokens=False)
        out[lang] = row(text, [show_hf_token(t) for t in e.tokens], list(e.ids))
    return out


def run_claude(model, client):
    """count_tokens adds a fixed per-message overhead; estimate it with a 1-token text '.'."""
    def count(text):
        r = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
        return r.input_tokens

    overhead = count(".") - 1
    out = {"_overhead_tokens": overhead}
    for lang, text in PHRASES.items():
        raw = count(text)
        n = raw - overhead
        out[lang] = {
            "text": text, "chars": len(text), "words": len(text.split()),
            "tokens": n, "tokens_raw_api": raw,
            "chars_per_token": round(len(text) / n, 2),
        }
    return out


def main():
    results = {}
    for name in TIKTOKEN_ENCODINGS:
        results[name] = run_tiktoken(name)
    for label, repo in HF_TOKENIZERS.items():
        results[label] = run_hf(repo)
        results[label]["_source"] = repo

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        for model in CLAUDE_MODELS:
            try:
                results[model] = run_claude(model, client)
            except Exception as e:  # unknown model, no access, ...
                print(f"[skip] {model}: {type(e).__name__}: {e}")
    else:
        print("[info] ANTHROPIC_API_KEY not set - Claude models skipped\n")

    # ratios vs English + console summary
    langs = list(PHRASES)
    print(f"{'tokenizer':<20}" + "".join(f"{l:>10}" for l in langs) + "   ratio vs EN: " + " ".join(langs[1:]))
    for name, r in results.items():
        en = r["en"]["tokens"]
        ratios = {l: round(r[l]["tokens"] / en, 2) for l in langs if l != "en"}
        r["token_ratio_vs_en"] = ratios
        print(f"{name:<20}" + "".join(f"{r[l]['tokens']:>10}" for l in langs)
              + "   " + " ".join(f"{ratios[l]:.2f}" for l in langs[1:]))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "exp01_tokenizer_ru_en.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
