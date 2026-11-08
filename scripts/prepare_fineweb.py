from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from upwash.core.data import load_dev_ids, prepare_token_shards

DATA_REPO = "HuggingFaceFW/fineweb-edu"
DATA_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATA_FILES = "sample/10BT/000_00000.parquet" 


DEV_REPO = "cerulean-works/fineweb-edu-dev"
DEV_REVISION = "2484afe000a6fdf442ba2396391a3bec12644903"
DEV_FILENAME = "dev-ids.txt"

TOKENIZER_REPO = "meta-llama/Llama-3.2-1B"
TOKENIZER_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"

CACHE_DIR = ".cache/huggingface"
OUTPUT_DIR = ".cache/tokens"

def main():
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", 
                           revision=TOKENIZER_REVISION, cache_dir=CACHE_DIR)
    tok = Tokenizer.from_file(path)
    eos_id = tok.token_to_id("<|end_of_text|>")
    assert eos_id == 128001, eos_id

    def tokenize(text):
        return tok.encode(text, add_special_tokens=False).ids
    
    dev_ids = load_dev_ids(DEV_REPO, revision=DEV_REVISION, filename=DEV_FILENAME, cache_dir=CACHE_DIR)

    summary = prepare_token_shards(DATA_REPO,
                                   revision=DATA_REVISION,
                                   data_files=DATA_FILES,
                                   output_dir=OUTPUT_DIR,
                                   tokenize=tokenize,
                                   eos_id=eos_id,
                                   exclude_ids=dev_ids,
                                   cache_dir=CACHE_DIR)
    print(summary)


if __name__ == "__main__":
    main()