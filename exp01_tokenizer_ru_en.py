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
    "ru_latin": "Iskusstvennyj intellekt menyaet to, kak lyudi rabotayut i uchatsya.",
    "kk": "Жасанды интеллект адамдардың жұмыс істеу және оқу тәсілін өзгертіп жатыр.",
}
ENCODINGS = ["cl100k_base", "o200k_base"]  # GPT-4 / GPT-4o families


def tokenize(enc, text):
    ids = enc.encode(text)
    # decode each id separately; a multi-byte char may be split across tokens
    pieces = [enc.decode_single_token_bytes(i) for i in ids]
    return ids, pieces


def show_piece(raw: bytes) -> str:
    """Text of a token; if it is a fragment of a multi-byte char, show hex bytes like <d0 b6>."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex(" ") + ">"


def main():
    results = {}
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        results[enc_name] = {}
        print(f"=== {enc_name} ===")
        for lang, text in PHRASES.items():
            ids, pieces = tokenize(enc, text)
            shown = [show_piece(p) for p in pieces]
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
        en_tokens = results[enc_name]["en"]["tokens"]
        ratios = {
            lang: round(results[enc_name][lang]["tokens"] / en_tokens, 2)
            for lang in PHRASES
            if lang != "en"
        }
        results[enc_name]["token_ratio_vs_en"] = ratios
        print("   token ratio vs EN: " + ", ".join(f"{k}={v:.2f}" for k, v in ratios.items()) + "\n")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "exp01_tokenizer_ru_en.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
