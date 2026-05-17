"""Chicory retrieval backend for EnterpriseRAG-Bench.

Queries a Chicory memory instance for each benchmark question,
maps retrieved memories back to dataset_doc_uuids, generates
answers via LLM, and writes the benchmark-compatible JSONL output.

CLI interface matches the benchmark's other retrieval backends
(vector_retrieval.py, bm25_retrieval.py) for drop-in compatibility.

Usage:
    python -m chicory.bench.chicory_retrieval \
        --questions-file /path/to/questions.jsonl \
        --output answers_chicory.jsonl \
        --db-path ~/.chicory/bench.db \
        --top-k 10 \
        --parallelism 4 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from chicory.bench.utils import (
    append_result,
    build_memory_uuid_map,
    load_already_processed,
    load_questions,
)

logger = logging.getLogger(__name__)

# ── Question classification & document drill-down ──────────────────

_DETAIL_PATTERNS = [
    re.compile(r"\bwhat (?:is|are|was|were) the (?:default|exact|specific|target|recommended)\b", re.I),
    re.compile(r"\bwhat (?:is|are|was|were) the (?:name|metric|threshold|limit|percentage|duration|period|value|score|count|number|size|rate|cost|price|fee|credit)\b", re.I),
    re.compile(r"\bhow (?:long|many|much|often)\b", re.I),
    re.compile(r"\bwhat (?:percentage|ratio|fraction|proportion)\b", re.I),
    re.compile(r"\bwhat (?:device|mode|version|firmware|configuration|setting)\b", re.I),
    re.compile(r"\bwhat (?:metric|counter|gauge|alert)\b", re.I),
    re.compile(r"\b(?:percentage|percent|latency|threshold|timeout|limit|credit|fee|price|cost|budget|cap)\w*\b.*\b(?:propos|specif|set|target|report|observ|measur)", re.I),
    re.compile(r"\b(?:root cause|misconfiguration|bug|error|failure mode)\b", re.I),
]

_STOPWORDS = frozenset(
    "a an the is are was were what which how does did do will would could "
    "should can may might have has had been being for of in on at to by with "
    "from and or but not this that these those it its they their them he she "
    "his her about after before between into through during when where who "
    "whom whose why new recent specific default current proposed recommended "
    "according describe".split()
)

_TERM_SPLIT = re.compile(r"[^a-zA-Z0-9_.\-/]+")


def classify_question(question: str) -> tuple[str, list[str]]:
    """Classify a question as 'detail' or 'broad' and extract search terms.

    Returns (category, terms) where terms are lowercased keywords
    useful for paragraph-level matching inside retrieved documents.
    """
    is_detail = any(p.search(question) for p in _DETAIL_PATTERNS)

    raw_tokens = _TERM_SPLIT.split(question)
    terms = [
        t.lower() for t in raw_tokens
        if len(t) > 2 and t.lower() not in _STOPWORDS
    ]

    # Preserve multi-word quoted phrases or hyphenated compounds
    for match in re.finditer(r'"([^"]+)"', question):
        terms.append(match.group(1).lower())

    return ("detail" if is_detail else "broad"), terms


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n")


def extract_relevant_sections(
    content: str,
    terms: list[str],
    context_lines: int = 0,
) -> str:
    """Extract only sentences containing search terms.

    Splits content into sentences, keeps only those with at least one
    term hit, plus up to `context_lines` neighboring sentences on each
    side for readability. Returns them in original order.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(content) if s.strip()]
    if not sentences or not terms:
        return content

    hit_indices: set[int] = set()
    for idx, sent in enumerate(sentences):
        lower = sent.lower()
        if any(t in lower for t in terms):
            hit_indices.add(idx)

    if not hit_indices:
        return content

    keep: set[int] = set()
    for idx in hit_indices:
        for offset in range(-context_lines, context_lines + 1):
            neighbor = idx + offset
            if 0 <= neighbor < len(sentences):
                keep.add(neighbor)

    return " ".join(sentences[i] for i in sorted(keep))

ANSWER_GEN_PROMPT = """\
You are a helpful and precise assistant that generates answers based on the \
provided documents. The documents came from a retrieval system which is \
imperfect. Base your answer purely on the documents and do not make up any \
information. Many of the documents provided are likely to be irrelevant. \
Be concise and only provide information directly relevant to the query.

## Context Documents
{context_documents}

## Question
{question}

## Answer
Output your answer below, do not include any additional text or formatting:
"""


def retrieve_for_question(
    orchestrator,
    question: str,
    memory_to_uuid: dict[str, str],
    top_k: int = 10,
    method: str = "hybrid",
) -> list[str]:
    """Query Chicory and return deduplicated dataset_doc_uuids."""
    result = orchestrator.handle_retrieve_memories(
        query=question,
        method=method,
        top_k=top_k * 3,
    )

    memories = result.get("memories", [])

    seen: set[str] = set()
    doc_uuids: list[str] = []
    for mem in memories:
        mid = mem.get("id", "")
        uuid = memory_to_uuid.get(mid)
        if uuid and uuid not in seen:
            seen.add(uuid)
            doc_uuids.append(uuid)
        if len(doc_uuids) >= top_k:
            break

    return doc_uuids


def format_retrieved_context(
    doc_uuids: list[str],
    uuid_to_memories: dict[str, list[str]],
    db,
    question_category: str = "broad",
    search_terms: list[str] | None = None,
) -> str:
    """Load memory content and format as the benchmark's context string.

    For 'detail' questions, drills down into each document and extracts
    only the paragraphs most relevant to the search terms.
    """
    parts: list[str] = []

    for i, uuid in enumerate(doc_uuids, 1):
        memory_ids = uuid_to_memories.get(uuid, [])
        if not memory_ids:
            continue

        content_parts: list[str] = []
        title = ""
        for mid in memory_ids:
            row = db.execute(
                "SELECT content, summary FROM memories WHERE id = ?",
                (mid,),
            ).fetchone()
            if row:
                if not title and row["summary"]:
                    title = row["summary"]
                content_parts.append(row["content"])

        content = "\n\n".join(content_parts)

        if question_category == "detail" and search_terms:
            content = extract_relevant_sections(content, search_terms)

        parts.append(
            f"--- Document {i} (ID: {uuid}) ---\n"
            f"Title: {title}\n\n"
            f"{content}"
        )

    return "\n\n".join(parts)


def generate_answer(
    orchestrator,
    context: str,
    question: str,
) -> str:
    """Generate an answer using the configured LLM client."""
    from chicory.llm import create_llm_client

    prompt = ANSWER_GEN_PROMPT.format(
        context_documents=context,
        question=question,
    )

    try:
        llm = create_llm_client(orchestrator._config)
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are a precise question-answering assistant. Answer based only on the provided documents.",
        )
        answer_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                answer_parts.append(block.text)
        return " ".join(answer_parts).strip()
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return f"Error generating answer: {e}"


def process_question(
    orchestrator,
    question: dict[str, Any],
    memory_to_uuid: dict[str, str],
    uuid_to_memories: dict[str, list[str]],
    top_k: int,
    method: str,
    output_path: str,
    write_lock: threading.Lock,
) -> dict[str, Any]:
    """Process a single benchmark question end-to-end."""
    qid = question["question_id"]
    query = question["question"]

    category, terms = classify_question(query)

    doc_uuids = retrieve_for_question(
        orchestrator, query, memory_to_uuid, top_k=top_k, method=method,
    )

    context = format_retrieved_context(
        doc_uuids, uuid_to_memories, orchestrator._db,
        question_category=category,
        search_terms=terms,
    )

    answer = generate_answer(orchestrator, context, query)

    result = {
        "question_id": qid,
        "answer": answer,
        "document_ids": doc_uuids,
        "question_category": category,
    }

    if "question_type" in question:
        result["question_type"] = question["question_type"]
    if "source_types" in question:
        result["source_types"] = question["source_types"]

    append_result(output_path, result, write_lock)
    return result


def run_benchmark(
    questions_file: str,
    output_path: str,
    db_path: str | None = None,
    top_k: int = 10,
    method: str = "hybrid",
    parallelism: int = 1,
    resume: bool = False,
    limit: int | None = None,
    question_ids: list[str] | None = None,
    sync: bool = False,
) -> dict[str, Any]:
    """Run the full Chicory retrieval benchmark.

    Returns stats dict with timing and counts.
    """
    from chicory.config import load_config
    from chicory.orchestrator.orchestrator import Orchestrator

    db = Path(db_path).expanduser() if db_path else Path.home() / ".chicory" / "bench.db"
    config = load_config(db_path=db)
    orchestrator = Orchestrator(config)

    if sync:
        logger.info("Running network sync before benchmark...")

        def _sync_progress(step: str, detail: str) -> None:
            logger.info("[sync:%s] %s", step, detail)

        sync_stats = orchestrator.run_sync(on_step=_sync_progress)
        logger.info("Sync complete in %.1fs", sync_stats.get("total_seconds", 0))

    questions = load_questions(questions_file, limit=limit, question_ids=question_ids)
    logger.info("Loaded %d questions", len(questions))

    if resume:
        already_done = load_already_processed(output_path)
        questions = [q for q in questions if q["question_id"] not in already_done]
        logger.info("After resume filter: %d remaining", len(questions))

    memory_to_uuid = build_memory_uuid_map(orchestrator._db)
    uuid_to_memories: dict[str, list[str]] = {}
    for mid, uuid in memory_to_uuid.items():
        uuid_to_memories.setdefault(uuid, []).append(mid)

    logger.info(
        "UUID index: %d unique documents, %d memory chunks",
        len(uuid_to_memories), len(memory_to_uuid),
    )

    stats: dict[str, Any] = {
        "total_questions": len(questions),
        "completed": 0,
        "errors": 0,
        "avg_docs_retrieved": 0.0,
        "elapsed_seconds": 0.0,
    }

    t0 = time.time()
    write_lock = threading.Lock()
    doc_counts: list[int] = []

    if parallelism <= 1:
        for q in questions:
            try:
                result = process_question(
                    orchestrator, q, memory_to_uuid, uuid_to_memories,
                    top_k, method, output_path, write_lock,
                )
                doc_counts.append(len(result["document_ids"]))
                stats["completed"] += 1
                logger.info(
                    "[%d/%d] %s → %d docs",
                    stats["completed"], len(questions),
                    q["question_id"], len(result["document_ids"]),
                )
            except Exception as e:
                stats["errors"] += 1
                logger.error("Error on %s: %s", q["question_id"], e)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    process_question,
                    orchestrator, q, memory_to_uuid, uuid_to_memories,
                    top_k, method, output_path, write_lock,
                ): q
                for q in questions
            }
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    doc_counts.append(len(result["document_ids"]))
                    stats["completed"] += 1
                    logger.info(
                        "[%d/%d] %s → %d docs",
                        stats["completed"], len(questions),
                        q["question_id"], len(result["document_ids"]),
                    )
                except Exception as e:
                    stats["errors"] += 1
                    logger.error("Error on %s: %s", q["question_id"], e)

    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    if doc_counts:
        stats["avg_docs_retrieved"] = round(sum(doc_counts) / len(doc_counts), 2)

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Chicory retrieval backend for EnterpriseRAG-Bench"
    )
    parser.add_argument(
        "--questions-file", required=True,
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--output", default="answers_chicory.jsonl",
        help="Output JSONL path (default: answers_chicory.jsonl)",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Chicory DB path (default: ~/.chicory/bench.db)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of documents to retrieve per question (default: 10)",
    )
    parser.add_argument(
        "--method", default="hybrid", choices=["hybrid", "semantic", "tag"],
        help="Chicory retrieval method (default: hybrid)",
    )
    parser.add_argument(
        "--parallelism", type=int, default=1,
        help="Parallel workers (default: 1)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip questions already in the output file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max questions to process",
    )
    parser.add_argument(
        "--question-id", nargs="*", default=None,
        help="Process specific question IDs only",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Run full network sync before starting the benchmark",
    )
    args = parser.parse_args()

    result = run_benchmark(
        questions_file=args.questions_file,
        output_path=args.output,
        db_path=args.db_path,
        top_k=args.top_k,
        method=args.method,
        parallelism=args.parallelism,
        resume=args.resume,
        limit=args.limit,
        question_ids=args.question_id,
        sync=args.sync,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
