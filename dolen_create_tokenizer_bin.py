#!/usr/bin/env python3
import os
import struct
import json
from transformers import AutoTokenizer

def create_tokenizer(model_dir: str, out_path: str = None) -> str:
    if out_path is None:
        out_path = os.path.join(model_dir, "tokenizer.bin")

    print(f"Info: out_path = \"{out_path}\"")

    if os.path.exists(out_path):
        print(f"Error: \"{out_path}\"  already exists")
        return out_path

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    
    # Use len(tok) to get the true number of tokens including added special tokens
    hf_vocab_size = len(tok)
    print(f"Info: HF vocab_size (len(tok)) = {hf_vocab_size}")

    config_path = os.path.join(model_dir, "config.json")
    model_vocab_size = hf_vocab_size
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
            # Check for nested text_config (common in multimodal models)
            model_vocab_size = config.get("vocab_size", None)
            if model_vocab_size is None and "text_config" in config:
                model_vocab_size = config["text_config"].get("vocab_size", None)
                
            if model_vocab_size is None:
                model_vocab_size = hf_vocab_size
            
    print(f"Info: Model vocab_size = {model_vocab_size}")


    def id_to_utf8_bytes(idx: int) -> bytes:
        piece = tok.convert_ids_to_tokens(idx)
        if piece is None:
            return b" "
        
        content = getattr(piece, "content", piece)
        if not isinstance(content, str):
            content = str(content)
        
        # Many BPE tokenizers represent a leading space with 'Ġ' (U+0120), 
        # SentencePiece (used by LLaMA) uses '▁' (U+2581) for space.
        # we manually replace these markers with a real ASCII space (0x20) here.
        content = content.replace("\u0120", " ")   \
                         .replace("\u2581", " ")   \
                         .replace("\u010a", "\n")  # 'Ċ' is often used for newline
        
        return content.encode("utf-8")


    tokens = []
    for i in range(hf_vocab_size):
        tokens.append(id_to_utf8_bytes(i))

    # Pad to target size to ensure we meet the model's expected vocab size
    target_vocab_size = max(model_vocab_size, hf_vocab_size)
    if target_vocab_size > hf_vocab_size:
        dummy_token = b"<pad>"
        for _ in range(target_vocab_size - hf_vocab_size):
            tokens.append(dummy_token)

    max_token_length = max((len(t) for t in tokens), default=0)
    print(f"Max token length: {max_token_length}")

    with open(out_path, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for bs in tokens:
            f.write(struct.pack("I", len(bs)))
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

