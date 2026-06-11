#!/usr/bin/env python3
import argparse
import json
import os
import struct

from transformers import AutoTokenizer


def create_tokenizer(model_dir: str, output: str = None) -> str:
    if output is None:
        output = os.path.join(model_dir, "tokenizer.bin")

    if os.path.exists(output):
        print(f"Tokenizer already exists: {output}")
        return output

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found under {model_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    llm = config.get("text_config", config)
    vocab_size = int(llm["vocab_size"])
    print(f"Config vocab_size: {vocab_size}")

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

    tv = getattr(tok, "vocab_size", None)
    if tv is not None and \
            tv != vocab_size:
        print(f"Note: tokenizer.vocab_size={tv} != config vocab_size; "
              f"exporting {vocab_size} rows to match checkpoint")


    def id_to_utf8_bytes(idx: int) -> bytes:
        piece = tok.convert_ids_to_tokens(idx)
        if piece is None:
            return b""

        content = getattr(piece, "content", piece)
        if not isinstance(content, str):
            content = str(content)

        s = tok.convert_tokens_to_string([content])
        return s.encode("utf-8")


    tokens = []
    for i in range(vocab_size):
        tokens.append(id_to_utf8_bytes(i))

    max_token_length = max((len(t) for t in tokens), default=0)
    print(f"Max token length: {max_token_length}")

    with open(output, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for bs in tokens:
            f.write(struct.pack("fI", 0.0, len(bs)))
            f.write(bs)

    print(f"Created tokenizer: {output} ({vocab_size} tokens)")
    return output


def main():
    parser = argparse.ArgumentParser(
            description="Create tokenizer.bin for Dolen"
            )
    parser.add_argument(
        "model_path",
        type=str,
        help="model path"
        )

    args = parser.parse_args()

    create_tokenizer(args.model_path)


if __name__ == "__main__":
    main()

