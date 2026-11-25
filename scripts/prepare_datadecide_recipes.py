"""Build a read-only Lance fragment manifest for five pinned DataDecide recipes."""

import argparse
import ast
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

from scripts.datadecide_paloma import atomic_json
from scripts.train_dclm_150m import open_datasets

DATASET = ("s3://flock-datasets/datadecide", 1)
REGISTRY_COMMIT = "7094aab0448096f4994cae881edbd629d6c2f3de"
REGISTRY_URL = f"https://raw.githubusercontent.com/allenai/OLMo/{REGISTRY_COMMIT}/olmo/data/named_data_mixes.py"
NATIVE_TOKENIZER = ("allenai/DataDecide-dclm-baseline-1B", "d897fc22d7bcb3db6fdfb0d9d9dd4ec048ea62b5")
DCLM_QC = "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top"
RECIPES = {
    "falcon-and-cc-qc-orig-10p": ("cc_og_eli5_oh_top10p", "falcon_og_eli5_oh_top10p"),
    "falcon-and-cc-qc-10p": ("cc_eli5_oh_top10p", "falcon_eli5_oh_top10p"),
    "falcon-and-cc-qc-20p": ("cc_eli5_oh_top20p", "falcon_eli5_oh_top20p"),
    "dclm-baseline-qc-20p": (DCLM_QC + "20p",),
    "dclm-baseline-qc-10p": (DCLM_QC + "10p",),
}


def source_files(registry):
    wanted = {source for sources in RECIPES.values() for source in sources}
    components = {}
    for node in ast.walk(ast.parse(registry)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            name = key.value if isinstance(key, ast.Constant) else None
            if name not in wanted or name in components:
                continue
            # This pinned registry uses string lists and simple range-based f-strings.
            if isinstance(value, ast.List):
                paths = ast.literal_eval(value)
            else:
                allowed = (ast.ListComp, ast.JoinedStr, ast.FormattedValue, ast.Constant, ast.Name,
                           ast.Load, ast.Store, ast.comprehension, ast.Call)
                if (not isinstance(value, ast.ListComp)
                        or any(not isinstance(item, allowed) for item in ast.walk(value))
                        or any(not isinstance(item.func, ast.Name) or item.func.id != "range"
                               for item in ast.walk(value) if isinstance(item, ast.Call))):
                    raise ValueError(f"Unsupported expression in Ai2 registry for {name}")
                paths = eval(compile(ast.Expression(value), REGISTRY_URL, "eval"),
                             {"__builtins__": {}, "range": range})
            if not paths or len(paths) != len(set(paths)) or any(not path.endswith(".npy") for path in paths):
                raise ValueError(f"Invalid file list for {name}")
            components[name] = paths
    if set(components) != wanted:
        raise ValueError(f"Missing Ai2 components: {wanted - components.keys()}")
    for name, members in RECIPES.items():
        paths = [path for member in members for path in components[member]]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Repeated source files for {name}")
    return components


def index_files(dataset, components):
    fragments = dataset.get_fragments()
    if [fragment.fragment_id for fragment in fragments] != sorted(fragment.fragment_id for fragment in fragments):
        raise ValueError("Lance v1 fragments are not in physical order")
    columns = ["_hf_source_file", "_hf_split", "source_token_offset", "source_token_count"]

    @lru_cache(maxsize=None)
    def edges(position):
        fragment = fragments[position]
        last = fragment.count_rows() - 1
        left, right = fragment.take([0, last], columns=columns).to_pylist()
        return left, right

    files = {}
    prefixes = {path.rsplit("/", 1)[0] + "/" for paths in components.values() for path in paths}
    expected = {path for paths in components.values() for path in paths}
    with ThreadPoolExecutor(max_workers=24) as pool:
        for prefix in sorted(prefixes):
            # The pinned import groups and sorts rows by source-file path.
            first = bisect_left(range(len(fragments)), prefix, key=lambda p: edges(p)[1]["_hf_source_file"])
            stop = bisect_left(range(len(fragments)), prefix + "\uffff",
                               key=lambda p: edges(p)[0]["_hf_source_file"])
            for position, (left, right) in zip(range(first, stop), pool.map(edges, range(first, stop)), strict=True):
                path = left["_hf_source_file"]
                if path != right["_hf_source_file"]:
                    raise ValueError(f"Mixed file boundary in fragment {fragments[position].fragment_id}")
                if path not in expected or not path.startswith(prefix):
                    continue
                if left["_hf_split"] != "train" or right["_hf_split"] != "train":
                    raise ValueError(f"Non-training fragment for {path}")
                start = left["source_token_offset"]
                end = right["source_token_offset"] + right["source_token_count"]
                entry = files.setdefault(path, {"path": path, "tokens": 0, "fragments": []})
                if start != entry["tokens"] or end <= start:
                    raise ValueError(f"Discontinuous source offsets for {path}: {start}, {end}")
                entry["fragments"].append([fragments[position].fragment_id, start, end])
                entry["tokens"] = end
            print(f"INDEX {prefix} fragments={stop - first}", flush=True)
    if missing := expected - files.keys():
        raise ValueError(f"{len(missing)} Ai2 files missing in Lance v1, first: {min(missing)}")
    return files


def prepare(path):
    with urlopen(REGISTRY_URL, timeout=30) as response:
        registry = response.read()
    components = source_files(registry)
    (dataset,) = open_datasets(DATASET)
    files = index_files(dataset, components)
    cache = "/work/asciibase64/upwash/datadecide-paloma/cache/huggingface/hub"
    repo, revision = NATIVE_TOKENIZER
    tokenizer = Path(hf_hub_download(repo, "tokenizer.json", revision=revision, cache_dir=cache))
    groups = {
        source: {"name": source, "tokens": sum(files[name]["tokens"] for name in names),
                 "files": [files[name] for name in names]}
        for source, names in components.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {
        "dataset": DATASET, "registry_commit": REGISTRY_COMMIT,
        "native_tokenizer": {
            "repo": repo, "revision": revision, "path": str(tokenizer),
            "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(), "eos": 50279,
        },
        "recipes": RECIPES, "groups": groups,
    })
    for name, members in RECIPES.items():
        print("RECIPE", name, [(part, len(groups[part]["files"]), groups[part]["tokens"]) for part in members])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    prepare(parser.parse_args().manifest)
