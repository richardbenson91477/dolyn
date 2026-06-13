#!/usr/bin/env python3
import os
import struct
from transformers import AutoTokenizer


def create_tokenizer(model_dir: str, out_path: str = None) -> str:
    if out_path is None:
        out_path = os.path.join(model_dir, "tokenizer.bin")

    print(f"Info: out_path = \"{out_path}\"")

    if os.path.exists(out_path):
        print(f"Error: tokenizer already exists")
        return out_path


    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

    vocab_size = getattr(tok, "vocab_size", None)

    print(f"Info: vocab_size = {vocab_size}")


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

    with open(out_path, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for bs in tokens:
            f.write(struct.pack("fI", 0.0, len(bs)))
            f.write(bs)

    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
            description="create binary-format tokenizer"
            )
    parser.add_argument(
        "model_path",
        type=str,
        help="model path"
        )

    args = parser.parse_args()

    create_tokenizer(args.model_path, "./tokenizer.bin")


if __name__ == "__main__":
    main()

