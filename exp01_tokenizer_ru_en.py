"""Experiment 1: tokenize the same phrase in Russian and English.

Run:  python exp01_tokenizer_ru_en.py
Writes results/exp01_tokenizer_ru_en.json and prints a summary.
"""
import json
import sys
from pathlib import Path

import tiktoken

sys.stdout.reconfigure(encoding="utf-8")

PHRASES = {
    "ru": "Искусственный интеллект меняет то, как люди работают и учатся.",
    "en": "Artificial intelligence is changing how people work and learn.",
}
ENCODINGS = ["cl100k_base", "o200k_base"]  # GPT-4 / GPT-4o families


def tokenize(enc, text):
    ids = enc.encode(text)
    # decode each id separately; a multi-byte char may be split across tokens
    pieces = [enc.decode_single_token_bytes(i) for i in ids]
    return ids, pieces


def main():
    results = {}
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        results[enc_name] = {}
        print(f"=== {enc_name} ===")
        for lang, text in PHRASES.items():
            ids, pieces = tokenize(enc, text)
            shown = [p.decode("utf-8", errors="replace") for p in pieces]
            results[enc_name][lang] = {
                "text": text,
                "chars": len(text),
                "words": len(text.split()),
                "tokens": len(ids),
                "chars_per_token": round(len(text) / len(ids), 2),
                "token_ids": ids,
                "token_pieces": shown,
            }
            print(f"[{lang}] {len(text)} chars, {len(text.split())} words -> {len(ids)} tokens "
                  f"({len(text) / len(ids):.2f} chars/token)")
            print("   " + " | ".join(shown))
        ratio = results[enc_name]["ru"]["tokens"] / results[enc_name]["en"]["tokens"]
        results[enc_name]["ru_to_en_token_ratio"] = round(ratio, 2)
        print(f"   RU/EN token ratio: {ratio:.2f}\n")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "exp01_tokenizer_ru_en.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
