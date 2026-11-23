#!/usr/bin/env python3
# ruff: noqa: RUF001  (the corpus holds look-alike characters on purpose)
"""Tokenizer validation: tinyllm's tokenizer against llama.cpp's, token for token, on a mixed corpus.

    uv run python tools/tokenizer_check.py Qwen3.8-27B-UD-Q4_K_XL.gguf

Starts the reference llama.cpp server on the CPU (tools/build_ref_llamacpp.sh) with the same file and tokenizes
every text with both; control tokens written in the text are parsed as such on both sides. The corpus covers
prose, numbers, whitespace runs, code, many scripts, emoji, control tokens and a rendered chat prompt. Exits
non-zero on any difference.
"""

from __future__ import annotations

import argparse
import sys

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure you'll see it's what they've done; don't worry, we'd rather wait.",
    "Numbers: 0 7 42 123 1234 12345 1234567890 3.14159 -2.5e-10 1,000,000 0x1F 2026-11-23 12:30:45",
    "Spaces:  two   three    four\ttab\t\ttabs\nnewline\n\nblank\n\n\nthree   \n trailing   ",
    "    indented code\n        deeper\n\treal tab\r\nwindows line\r\n",
    'def fib(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
    "    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    '{"name": "get_weather", "arguments": {"city": "Paris", "days": [1, 2, 3], "ok": true, "x": null}}',
    "#include <vector>\nint main() { std::vector<int> v{1,2,3}; for (auto& x : v) x <<= 2; return v[0] != 4; }",
    "<div class=\"a\" id='b'>&lt;tag&gt; &amp; text</div><!-- comment --><br/>",
    r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+)@(?:[a-z0-9-]+\.)+[a-z]{2,}$",
    "for f in *.gguf; do echo \"$f\" | sed -e 's/\\.gguf$//'; done 2>&1 | tee -a log.txt",
    "SELECT u.id, COUNT(*) AS n FROM users u JOIN orders o ON o.user_id = u.id WHERE o.total > 9.99 GROUP BY u.id;",
    "中文分词测试：今天天气很好，我们去公园散步吧。人工智能正在改变世界。",
    "日本語のテキスト：吾輩は猫である。名前はまだ無い。カタカナとひらがなと漢字。",
    "한국어 문장입니다. 오늘 날씨가 좋네요!",
    "النص العربي: مرحبا بالعالم، كيف حالك اليوم؟",
    "עברית: שלום עולם",
    "हिन्दी पाठ: नमस्ते दुनिया, आप कैसे हैं?",
    "Русский текст: Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Ελληνικά: Γειά σου Κόσμε. ΑΒΓΔΕ αβγδε.",
    "ภาษาไทย: สวัสดีชาวโลก",
    "Tiếng Việt: Xin chào thế giới, hôm nay trời đẹp quá!",
    "Emoji: 😀😃😄 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 ❤️ 🇮🇹 🧑🏿‍💻 ✅ 🚀",
    "Combining: é ñ ä ẛ̣ Zalgo: Z͑ͫÂL̀G̃O",
    "Symbols: ∀x∈ℝ: x² ≥ 0 ⇒ √(x²) = |x| · ∑ ∫ ∞ ≈ ≠ ≤ ≥ € £ ¥ © ® ™ § ¶ † ‡ • … ‰ ′ ″",
    "Mixed: Hello世界123abc！？。、「」『』（）《》",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "ab" * 60 + " " + "ha" * 40 + " " + "!?" * 30,
    "URL: https://example.com/path/to/page.html?q=search+terms&lang=en#section-2 mailto:a.b@c.de",
    "Markdown: # Title\n\n- **bold** and *italic* and `code`\n- [link](http://x.y)\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "LaTeX: $\\frac{a}{b} + \\sqrt{x^2 + y^2} = \\int_0^\\infty e^{-t} \\, dt$",
    "",
    " ",
    "\n",
    "a",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="GGUF path, or a file name under the model directory")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    from tinyllm import device

    device.configure("cpu")
    from tinyllm.frontend.template import ChatTemplate
    from tinyllm.oracle.llamacpp import ReferenceServer
    from tinyllm.tg.tokenizer import GGUFTokenizer
    from tinyllm.weights import gguf
    from tinyllm.weights.resolve import find_model

    path = find_model(args.model)
    ours = GGUFTokenizer(gguf.read(path, max_array_items=1 << 24, header_only=True).kv)
    corpus = list(CORPUS)
    if ours.chat_template:
        weather = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
        messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "¿Qué tiempo hace? 天气?"}]
        corpus.append(ChatTemplate(ours.chat_template).render(messages, tools=[weather]))
    failures = 0
    with ReferenceServer(path, threads=args.threads) as reference:
        for i, text in enumerate(corpus):
            mine, theirs = ours.encode(text), reference.tokenize(text)
            if mine != theirs:
                failures += 1
                at = next(
                    (j for j, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b),
                    min(len(mine), len(theirs)),
                )
                print(f"DIFF {i:2d}: {len(mine)} vs {len(theirs)} tokens, first difference at {at}: {text[:50]!r}")
    print(f"{len(corpus) - failures} of {len(corpus)} texts tokenized identically ({path.name})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
