from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import os
from pathlib import Path

from upwash.core.data import prepare_token_shards

DEV_REPO = "cerulean-works/fineweb-edu-dev"
DEV_REVISION = "2484afe000a6fdf442ba2396391a3bec12644903"
DEV_FILES = "data/validation-*.parquet"

TOKENIZER_REPO = "meta-llama/Llama-3.2-1B"
TOKENIZER_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"

WORK_ROOT = Path("/work") / os.environ["USER"]
CACHE_DIR = str(WORK_ROOT / ".cache/huggingface")
OUTPUT_DIR = str(WORK_ROOT / "fineweb-edu-dev-tokens")

def main():
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", 
                           revision=TOKENIZER_REVISION, cache_dir=CACHE_DIR)
    tok = Tokenizer.from_file(path)
    bos_id = tok.token_to_id("<|begin_of_text|>")
    eos_id = tok.token_to_id("<|end_of_text|>")
    assert (bos_id, eos_id) == (128000, 128001), (bos_id, eos_id)

    def tokenize(text):
        return [bos_id]+ tok.encode(text, add_special_tokens=False).ids
    
    summary = prepare_token_shards(DEV_REPO,
                                   revision=DEV_REVISION,
                                   data_files=DEV_FILES,
                                   output_dir=OUTPUT_DIR,
                                   tokenize=tokenize,
                                   eos_id=eos_id,
                                   exclude_ids=set(),
                                   cache_dir=CACHE_DIR)
    print(summary)


if __name__ == "__main__":
    main()
