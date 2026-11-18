import json
import os

import pandas as pd

from questions import ABSENT, BANK, BOUNDARY, CITED, DIALOGUES, EARLY, LATE, MAIN, PARAPHRASES
from rag import (BinaryBoW, Dialogue, K, HERE, STRICT, check_citations, dense_search, duplicates,
                 evidence_hit, generate, hit_rate, normalized, pack, review_hash, split_pages)

RESULTS = HERE / os.environ.get("LAB3_RESULTS", "results")
RESULTS.mkdir(parents=True, exist_ok=True)


def save(name, value):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class Experiments:
    def __init__(self, pages, llm, embeddings, original):
        self.pages, self.llm, self.embeddings, self.original = pages, llm, embeddings, original
        self.base_chunks = split_pages(pages)
        self.indexes = {}
        self.index_stats = {}
        self.review = read_json(HERE / "review.json", {})

    def build(self, name, chunks):
        if name not in self.indexes:
            self.indexes[name] = dense_search(chunks, self.embeddings) if self.embeddings is not None else None
            self.index_stats[name] = {"chunks": len(chunks), "created": self.indexes[name] is not None}
        return self.indexes[name]

    def base(self):
        return self.build("full-1000-200", self.base_chunks)

    def verify_bank(self):
        pages = {p.metadata["page_number"]: p.page_content for p in self.pages}
        for key, item in BANK.items():
            for e in item["evidence"]:
                assert normalized(e["quote"]) in normalized(pages[e["page"]]), key
        for key in EARLY + LATE:
            hit = any(evidence_hit(BANK[key], doc) for doc in self.base_chunks[:5])
            assert hit == (key in EARLY), key
        boundaries = split_pages(self.pages, 1000, 0)
        positions = []
        for key in BOUNDARY:
            e = BANK[key]["evidence"][0]
            page = pages[e["page"]]
            a = normalized(page).index(normalized(e["quote"]))
            b = a + len(normalized(e["quote"]))
            cuts = [len(normalized(page[:d.metadata["start_index"]])) for d in boundaries
                    if d.metadata["page_number"] == e["page"] and d.metadata["start_index"] > 0]
            cut = next(x for x in cuts if a < x < b)
            positions.append({"id": key, "page": e["page"], "evidence_begin": a,
                              "evidence_end": b, "boundary": cut, "quote": e["quote"]})
        save("question_bank", {"positive": BANK, "early": EARLY, "late": LATE,
                               "size": MAIN, "boundary_cases": positions, "cited": CITED,
                               "paraphrases": PARAPHRASES, "dialogues": DIALOGUES, "absent": ABSENT})
        return positions

    def row(self, key, search, *, question=None, template=None, documents=None, answer=None,
            record_key=None, case=None, query=None):
        case = BANK[key] if case is None else case
        question = question or case["question"]
        query = query or question
        if documents is None and search is not None:
            documents = search(query)
        fragments = pack(documents, case) if documents is not None else None
        if template is not None and documents is not None and self.llm is not None:
            answer = generate(self.llm, query, documents, template)
        hash_value = review_hash(answer, fragments)
        review = self.review.get(record_key, {})
        if review.get("fingerprint") != hash_value:
            review = {}
        match = None if fragments is None or not case["evidence"] else any(
            d["contains_annotated_evidence"] for d in fragments)
        return {
            "id": key, "review_key": record_key, "fingerprint": hash_value,
            "question": question, "query": query, "reference": case["answer"], "evidence": case["evidence"],
            "hit_at_3": review.get("hit_at_3", match), "hit_annotated_passage": match,
            "fragments": fragments, "top3_ids": None if documents is None else [
                d.metadata["chunk_id"] for d in documents],
            "context_chars": None if fragments is None else sum(len(d["text"]) for d in fragments),
            "near_duplicate_pairs": None if documents is None else duplicates(documents),
            "answer": answer, "score_0_2": review.get("score"), "answer_without_evidence": review.get("absence"),
            "review_note": review.get("note"),
        }

    @staticmethod
    def aggregate(rows):
        hits = [r["hit_at_3"] for r in rows]
        scores = [r["score_0_2"] for r in rows if r["score_0_2"] is not None]
        return {"questions": len(rows), "hits": sum(hits) if all(h is not None for h in hits) else None,
                "hit_rate_at_3_pct": hit_rate(hits) if all(h is not None for h in hits) else None,
                "mean_score": round(sum(scores) / len(scores), 3) if len(scores) == len(rows) and rows else None,
                "assessed": len(scores)}

    def baseline(self):
        return self.row("D01", self.base(), template=self.original, record_key="base/D01")

    def coverage(self):
        rows, summary = [], []
        for variant, chunks, search in [
            ("first_5", self.base_chunks[:5], self.build("first-5", self.base_chunks[:5])),
            ("full", self.base_chunks, self.base()),
        ]:
            group = []
            for key in EARLY + LATE:
                group.append({**self.row(key, search, template=self.original, record_key=f"e1/{variant}/{key}"),
                              "variant": variant, "group": "early" if key in EARLY else "late"})
            rows.extend(group)
            summary.append({"variant": variant, "chunks": len(chunks),
                            "early_hits": self.aggregate(group[:3])["hits"],
                            "late_hits": self.aggregate(group[3:])["hits"], **self.aggregate(group)})
        return rows, summary

    def size(self, overlap):
        rows, summary = [], []
        for size in (500, 1000, 2000):
            chunks = split_pages(self.pages, size, overlap)
            search = self.build(f"full-{size}-{overlap}", chunks)
            group = [{**self.row(key, search, record_key=f"e2/{overlap}/{size}/{key}"),
                      "size": size, "overlap": overlap} for key in MAIN]
            rows.extend(group)
            totals = [row["context_chars"] for row in group if row["context_chars"] is not None]
            summary.append({"size": size, "overlap": overlap, "chunks": len(chunks),
                            "mean_context_chars": round(sum(totals) / len(totals)) if totals else None,
                            **self.aggregate(group)})
        return rows, summary

    def overlap(self):
        rows, summary = [], []
        for overlap in (0, 100, 200):
            chunks = split_pages(self.pages, 1000, overlap)
            search = self.build(f"full-1000-{overlap}", chunks)
            group = [{**self.row(key, search, record_key=f"e3/{overlap}/{key}"),
                      "overlap": overlap} for key in BOUNDARY]
            rows.extend(group)
            summary.append({"overlap": overlap, "chunks": len(chunks),
                            "questions_with_duplicates": sum(bool(r["near_duplicate_pairs"]) for r in group) if search else None,
                            "duplicates_pairs": sum(len(r["near_duplicate_pairs"]) for r in group) if search else None,
                            **self.aggregate(group)})
        return rows, summary

    def compare_search(self):
        rows, summary = [], []
        bow = BinaryBoW(self.base_chunks)
        assert not bow("qzxvunknownlexemetest")
        for method, search in [("BoW", bow), ("OllamaEmbeddings", self.base())]:
            for kind in ("exact", "paraphrase"):
                group = [{**self.row(key, search, question=paraphrase if kind == "paraphrase" else None,
                                     record_key=f"e4/{method}/{kind}/{key}"), "method": method, "kind": kind}
                         for key, paraphrase in PARAPHRASES.items()]
                rows.extend(group)
                summary.append({"method": method, "kind": kind, **self.aggregate(group)})
        return rows, summary

    def citations(self):
        rows, summary = [], []
        search = self.base()
        for key in CITED:
            docs = search(BANK[key]["question"]) if search else None
            row = self.row(key, None, documents=docs, template=STRICT, record_key=f"e5/{key}")
            links = check_citations(row["answer"], docs) if row["answer"] is not None else []
            review = self.review.get(row["review_key"], {})
            supports = review.get("citation_support", []) if review.get("fingerprint") == row["fingerprint"] else []
            for i, link in enumerate(links):
                link["supports_claim"] = supports[i] if i < len(supports) else (False if not link["exists"] else None)
            rows.append({**row, "citations": links})
            summary.extend([{"id": key, "format_link_found": True, **{k: l[k] for k in
                             ["citation", "exists", "page_correct", "supports_claim"]}} for l in links] or
                           [{"id": key, "format_link_found": False if row["answer"] is not None else None,
                             "citation": None, "exists": None, "page_correct": None, "supports_claim": None}])
        return rows, summary

    def absence(self):
        rows, checked, summary = [], [], []
        search = self.base()
        for key, (question, terms, note) in ABSENT.items():
            checked.append({"id": key, "question": question, "reviewed_pages": len(self.pages),
                            "term_pages": {term: [p.metadata["page_number"] for p in self.pages
                                                  if normalized(term) in normalized(p.page_content)]
                                           for term in terms}, "conclusion": note})
        save("06_full_document_check", checked)
        cases = [(key, BANK[key], True) for key in CITED] + [
            (key, {"question": q, "answer": "Информация не приведена в PDF.", "evidence": []}, False)
            for key, (q, _, _) in ABSENT.items()]
        for key, case, available in cases:
            docs = search(case["question"]) if search else None
            for kind, template in [("original", self.original), ("strict", STRICT)]:
                rows.append({**self.row(key, None, documents=docs, template=template, case=case,
                                        record_key=f"e6/{kind}/{key}"), "prompt": kind, "in_pdf": available})
            assert rows[-1]["fragments"] == rows[-2]["fragments"]
        for kind in ("original", "strict"):
            group = [r for r in rows if r["prompt"] == kind]
            absent = [r["answer_without_evidence"] for r in group if not r["in_pdf"]]
            complete = all(a is not None for a in absent)
            score = self.aggregate([r for r in group if r["in_pdf"]])
            summary.append({"prompt": kind, "mean_score_answerable": score["mean_score"],
                            "answerable_assessed": score["assessed"],
                            "correct_refusals": absent.count("refusal") if complete else None,
                            "unsupported_answers": absent.count("unsupported") if complete else None,
                            "invalid_outputs": absent.count("invalid") if complete else None})
        return rows, summary

    def dialogue(self):
        search = self.base()
        rows, checks = [], None
        if search is not None and self.llm is not None:
            chat = Dialogue(search, self.llm, STRICT)
            # Чередование сессий проверяет, что предыдущий чужой обмен не используется.
            for turn in range(3):
                for session, sequence in DIALOGUES.items():
                    key, utterance = sequence[turn]
                    query, chunks, answer, count = chat.ask(session, utterance)
                    assert count == turn
                    bare = search(utterance) if turn else chunks
                    rows.append({**self.row(key, None, question=utterance, query=query, documents=chunks,
                                            answer=answer, record_key=f"e7/{session}/{turn+1}"),
                                 "session": session, "turn": turn + 1, "history_used": count,
                                 "raw_query_hit": any(evidence_hit(BANK[key], d) for d in bare),
                                 "raw_query_fragments": pack(bare, BANK[key])})
            before = {name: chat.history(name) for name in DIALOGUES}
            query, _, _, used = chat.ask("instruct", "Which base model are we discussing?")
            assert used == 3 and len(chat.history("instruct")) == 3
            assert chat.history("instruct")[:2] == before["instruct"][1:]
            chat.clear("instruct")
            assert chat.history("instruct") == [] and chat.history("training") == before["training"]
            original = "What is InstructGPT?"
            reset_query, _, _, used = chat.ask("instruct", original)
            assert reset_query == original and used == 0
            checks = {"isolated": True, "window": 3, "eviction": True, "clear": True,
                      "clear_preserves_other_session": True, "fourth_turn_query": query}
        save("07_session_checks", checks)
        summary = [{k: r[k] for k in ("session", "turn", "question", "query", "history_used",
                                      "raw_query_hit", "hit_at_3", "top3_ids", "score_0_2")} for r in rows]
        return rows, summary


def publish(name, result):
    rows, summary = result
    save(f"{name}_details", rows)
    frame = pd.DataFrame(summary)
    frame.to_csv(RESULTS / f"{name}_summary.csv", index=False, encoding="utf-8-sig")
    return frame
