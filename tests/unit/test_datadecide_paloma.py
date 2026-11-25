import math
from types import SimpleNamespace

import pytest
import torch

from scripts.datadecide_paloma import DOCUMENTS, SEEDS, STEP, atomic_json, ranking_rows
from scripts.paloma_validation import evaluate


def test_paloma_scores_total_nll_per_utf8_byte_across_windows():
    class UniformModel(torch.nn.Module):
        def forward(self, inputs):
            return SimpleNamespace(logits=torch.zeros((*inputs.shape, 2)))

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[0, 1, 0] if text == "éx" else [1, 0])

    model = UniformModel()
    result = evaluate(model, Tokenizer(), ["éx", "中"], "cpu", prefix_token_id=1, seq_len=2)
    assert model.training
    assert result["eval/loss"] == pytest.approx(math.log(2))
    assert result["eval/bpb"] == pytest.approx(5 / 6)
    assert (result["eval/tokens"], result["eval/bytes"]) == (5, 6)


def test_ranking_validates_all_seeds_and_preserves_full_precision_ties(tmp_path):
    recipes = []
    jobs = {}
    for rank in range(1, 11):
        slug = f"recipe-{rank}"
        repo = f"allenai/{slug}"
        revisions = {seed: f"{rank}-{seed}" for seed in SEEDS}
        recipes.append({
            "macro_rank": rank, "recipe": slug, "repo": repo, "slug": slug, "revisions": revisions,
        })
        jobs[str(rank)] = str(rank + 100)
        (tmp_path / slug).mkdir()
        for seed, revision in revisions.items():
            atomic_json(tmp_path / slug / f"{seed}.json", {
                "repo": repo, "revision": revision, "seed": seed, "step": STEP,
                "validation_sha256": "test", "documents": DOCUMENTS,
                "slurm_job_id": jobs[str(rank)], "eval/bytes": 100, "eval/tokens": 75,
                "eval/loss": 1.0, "eval/bpb": float(1 if rank in (2, 3) else rank + 1),
            })
    manifest = {"recipes": recipes, "step": STEP, "validation": {"sha256": "test", "bytes": 100}}
    rows = ranking_rows(tmp_path, manifest, jobs)
    assert [row["macro_rank"] for row in rows[:3]] == [2, 3, 1]
    assert [row["paloma_rank"] for row in rows[:3]] == [1, 1, 3]
    assert rows[0]["seed_sd"] == 0

    (tmp_path / "recipe-2" / "large-aux-2.json").unlink()
    with pytest.raises(ValueError, match="recipe-2 / large-aux-2 / Slurm 102"):
        ranking_rows(tmp_path, manifest, jobs)
