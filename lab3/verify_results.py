import csv
from hashlib import sha256
import json
import re
from math import isclose

import nbformat

from rag import HERE, hit_rate, original_prompt, review_hash


def read(name):
    return json.loads((HERE / "results" / f"{name}.json").read_text(encoding="utf-8"))


def table(name):
    with (HERE / "results" / f"{name}_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify():
    path = HERE / "rag_experiments.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(isinstance(c.execution_count, int) for c in code_cells)
    assert all("execution" not in c.metadata for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    counts = {"01_coverage": 12, "02_size": 60, "03_overlap": 15, "04_search": 20,
              "05_citations": 5, "06_prompts_comparison": 20, "07_dialogue": 9}
    group_fields = {"01_coverage": ["variant"], "02_size": ["size", "overlap"],
                    "03_overlap": ["overlap"], "04_search": ["method", "kind"]}
    for name, size in counts.items():
        rows = read(name + "_details")
        assert len(rows) == size
        for r in rows:
            assert r["fragments"] is not None and len(r["fragments"]) <= 3
            assert r["top3_ids"] == [f["chunk_id"] for f in r["fragments"]]
            assert [f["rank"] for f in r["fragments"]] == list(range(1, len(r["fragments"]) + 1))
            assert r["fingerprint"] == review_hash(r["answer"], r["fragments"])
            assert len(set(r["top3_ids"])) == len(r["top3_ids"])
        if name in group_fields:
            for summary in table(name):
                group = [r for r in rows if all(str(r[k]) == summary[k] for k in group_fields[name])]
                assert group and len(group) == int(summary["questions"])
                assert sum(r["hit_at_3"] for r in group) == int(summary["hits"])
                assert isclose(hit_rate([r["hit_at_3"] for r in group]), float(summary["hit_rate_at_3_pct"]))
        if name in {"01_coverage", "05_citations", "06_prompts_comparison", "07_dialogue"}:
            assert all(r["answer"] for r in rows)
        if name in {"01_coverage", "07_dialogue"}:
            assert all(r["score_0_2"] in {0, 1, 2} for r in rows)
    both = read("06_prompts_comparison_details")
    assert read("06_prompts")["original_sha256"] == sha256(original_prompt().encode()).hexdigest()
    for first, strict in zip(both[::2], both[1::2], strict=True):
        assert first["id"] == strict["id"] and first["fragments"] == strict["fragments"]
        for row in (first, strict):
            if row["in_pdf"]:
                assert row["score_0_2"] in {0, 1, 2}
            else:
                assert row["answer_without_evidence"] in {"refusal", "unsupported", "invalid"}
    for row in read("05_citations_details"):
        assert all(isinstance(link["supports_claim"], bool) for link in row["citations"])
    indexes = read("indexes")
    assert len(indexes) == 9 and all(x["created"] for x in indexes.values())
    dialogue = read("07_session_checks")
    assert dialogue["isolated"] and dialogue["window"] == 3 and dialogue["eviction"] and dialogue["clear"]
    for p in HERE.iterdir():
        if p.suffix not in {".py", ".txt", ".md", ".ipynb", ".json"}:
            continue
        content = p.read_text(encoding="utf-8")
        assert "\u2014" not in content, (p.name, "длинное тире")
        assert not re.search(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\" ]+", content), p.name
    result = {"code_cells": len(code_cells), "executed_without_errors": True,
              "verified_indexes": len(indexes), "experiment_rows": counts,
              "hit_rate_reconciled": True, "session_isolation": True,
              "review_annotations_validated": True, "source_text_style_checked": True}
    (HERE / "results" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(verify())
