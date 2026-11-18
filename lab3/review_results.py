from collections import defaultdict

import pandas as pd

from experiments import Experiments, RESULTS, read_json, save
from rag import HERE, review_hash


def label_rows(rows, reviews):
    for row in rows:
        review = reviews.get(row["review_key"], {})
        assert row["fingerprint"] == review_hash(row["answer"], row["fragments"])
        if review.get("fingerprint") != row["fingerprint"]:
            continue
        if "score" in review:
            assert review["score"] in (0, 1, 2) and row["answer"] is not None
            row["score_0_2"] = review["score"]
        if "absence" in review:
            assert review["absence"] in ("refusal", "unsupported", "invalid") and row.get("in_pdf") is False
            row["answer_without_evidence"] = review["absence"]
        if "hit_at_3" in review:
            assert isinstance(review["hit_at_3"], bool) and row["fragments"] is not None
            row["hit_at_3"] = review["hit_at_3"]
        if "citation_support" in review:
            assert len(row["citations"]) == len(review["citation_support"])
            for link, supported in zip(row["citations"], review["citation_support"], strict=True):
                assert isinstance(supported, bool) and (link["exists"] or not supported)
                link["supports_claim"] = supported
        if "rewrite_ok" in review:
            assert row["turn"] > 1 and isinstance(review["rewrite_ok"], bool)
            row["rewrite_ok"] = review["rewrite_ok"]
        row["review_note"] = review.get("note")


def finalize_review():
    reviews = read_json(HERE / "review.json", {})
    baseline = read_json(RESULTS / "baseline.json", None)
    if baseline:
        label_rows([baseline], reviews)
        save("baseline", baseline)
    names = ["01_coverage", "02_size", "03_overlap", "04_search", "05_citations",
             "06_prompts_comparison", "07_dialogue"]
    tables = {}
    for name in names:
        rows = read_json(RESULTS / f"{name}_details.json", [])
        if not rows:
            continue
        label_rows(rows, reviews)
        save(f"{name}_details", rows)
        if name == "05_citations":
            items = []
            for row in rows:
                for link in row["citations"] or [{}]:
                    items.append({"id": row["id"], "format_link_found": bool(link),
                                  **{k: link.get(k) for k in ("citation", "exists", "page_correct", "supports_claim")}})
        elif name == "07_dialogue":
            keys = ["session", "turn", "question", "query", "history_used", "raw_query_hit",
                    "hit_at_3", "top3_ids", "score_0_2", "rewrite_ok"]
            items = [{k: row.get(k) for k in keys} for row in rows]
        else:
            keys_map = {"01_coverage": ("variant",), "02_size": ("overlap", "size"),
                        "03_overlap": ("overlap",), "04_search": ("method", "kind"),
                        "06_prompts_comparison": ("prompt",)}
            groups = defaultdict(list)
            keys = keys_map[name]
            for row in rows:
                groups[tuple(row[k] for k in keys)].append(row)
            path = RESULTS / f"{name}_summary.csv"
            precomputed = pd.read_csv(path).to_dict("records")
            items = []
            for old in precomputed:
                group = groups[tuple(old[k] for k in keys)]
                clean_old = {k: (v if pd.notna(v) else None) for k, v in old.items()}
                if name == "06_prompts_comparison":
                    score = Experiments.aggregate([r for r in group if r["in_pdf"]])
                    absent = [r["answer_without_evidence"] for r in group if not r["in_pdf"]]
                    complete = all(a is not None for a in absent)
                    item = {**clean_old, "mean_score_answerable": score["mean_score"],
                            "answerable_assessed": score["assessed"],
                            "correct_refusals": absent.count("refusal") if complete else None,
                            "unsupported_answers": absent.count("unsupported") if complete else None,
                            "invalid_outputs": absent.count("invalid") if complete else None}
                else:
                    item = {**clean_old, **Experiments.aggregate(group)}
                    if name == "01_coverage":
                        for flag in ("early", "late"):
                            item[f"{flag}_hits"] = Experiments.aggregate(
                                [r for r in group if r["group"] == flag])["hits"]
                items.append(item)
        frame = pd.DataFrame(items)
        frame.to_csv(RESULTS / f"{name}_summary.csv", index=False, encoding="utf-8-sig")
        tables[name] = frame
    return tables


if __name__ == "__main__":
    for name, table in finalize_review().items():
        print(name)
        print(table.to_string(index=False))
