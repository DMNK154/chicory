"""Claude tool_use JSON schema definitions for Chicory memory tools."""

CHICORY_TOOLS = [
    {
        "name": "store_memory",
        "description": (
            "Store a new memory with tags and importance rating. "
            "Use this when the conversation contains information worth remembering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Be specific and self-contained.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant tags. Use existing tags when possible.",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How important is this memory? 0.0=trivial, 1.0=critical",
                },
                "summary": {
                    "type": "string",
                    "description": "Optional one-line summary for quick reference.",
                },
                "author": {
                    "type": "string",
                    "description": "Who/what wrote this memory (provenance byline).",
                },
                "origin": {
                    "type": "string",
                    "description": "Where the memory came from (session, source, context).",
                },
            },
            "required": ["content", "tags"],
        },
    },
    {
        "name": "retrieve_memories",
        "description": (
            "Search for relevant memories. Use this when you need to recall "
            "information from past conversations or find connections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filter.",
                },
                "method": {
                    "type": "string",
                    "enum": ["semantic", "tag", "hybrid"],
                    "description": "Retrieval method. Default: hybrid.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max results. Default: 10.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_trends",
        "description": (
            "View current tag trend signals: what topics are heating up, "
            "cooling down, or accelerating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific tags to check. Omit for all active tags.",
                },
            },
        },
    },
    {
        "name": "get_phase_space",
        "description": (
            "View the phase space: where each tag sits on the "
            "trend-temperature vs retrieval-frequency plane."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_synchronicities",
        "description": (
            "View detected synchronicity events: meaningful coincidences where "
            "retrieval patterns diverge from trend patterns. Includes effective_strength "
            "(decayed and reinforcement-boosted), reinforcement data, and overall "
            "synchronicity velocity (level/velocity/jerk of event activity)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max events to return. Default: 10.",
                },
                "unacknowledged_only": {
                    "type": "boolean",
                    "description": "Only show new/unseen events. Default: false.",
                },
            },
        },
    },
    {
        "name": "get_meta_patterns",
        "description": (
            "View detected higher-order meta-patterns: recurring themes "
            "across synchronicity events."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_lattice_resonances",
        "description": (
            "View the prime Ramsey lattice: synchronicity events organized by "
            "angular position across prime scales. Shows resonances (structurally "
            "entangled events) and the void profile (latent attractor themes). "
            "Returns top items by strength; use max_items to control count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_items": {
                    "type": "integer",
                    "description": "Max positions/resonances to return (default 10).",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "deep_retrieve",
        "description": (
            "Recursively retrieve memories by following association chains. "
            "Starts with a standard query, then expands through related memories. "
            "Deeper recursion levels progressively favor older memories, surfacing "
            "the deep past through semantic association."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Initial natural language search query.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filter for the initial retrieval.",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum recursion depth. Default: 3.",
                },
                "per_level_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Results per recursion level. Default: 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_last_trace",
        "description": (
            "Get the activation trace from the most recent retrieval. "
            "Shows top tag activations with strengths, per-memory score "
            "breakdowns (semantic, tag, lattice, glyph), and timing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "compare_traces",
        "description": (
            "Compare the last two retrieval traces. Shows tag activation "
            "deltas, memories that appeared or disappeared, per-component "
            "score deltas for shared memories, and rank changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "librarian_status",
        "description": (
            "Librarian overview: write-gate calibration stats (verdict counts "
            "and score ranges), sweep patrol progress, quarantined memory "
            "count, and subject roster summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "librarian_review",
        "description": (
            "The librarian's queue: quarantined memories awaiting judgment "
            "and open sweep findings (near-duplicates, tag drift, missing "
            "embeddings). Use librarian_resolve to act on items."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max items per section (default 20).",
                },
            },
        },
    },
    {
        "name": "librarian_resolve",
        "description": (
            "Apply a judgment from librarian_review. For a quarantined "
            "memory: pass memory_id plus action (release_critical, "
            "release_reference, or archive). For a sweep finding: pass "
            "finding_id to mark it resolved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Quarantined memory to act on.",
                },
                "action": {
                    "type": "string",
                    "enum": ["release_critical", "release_reference", "archive"],
                    "description": "Judgment to apply to the memory.",
                },
                "finding_id": {
                    "type": "integer",
                    "description": "Sweep finding to mark resolved.",
                },
            },
        },
    },
    {
        "name": "librarian_maintain",
        "description": (
            "Run one librarian maintenance pass now: sweep the next batch "
            "of the shelf-reading patrol, rebuild the subject roster, and "
            "process the demand ledger (FAQ promotion, commission filing). "
            "The roster rebuild is warm by default, inheriting the standing "
            "tree's shape so shelf paths stay stable as addresses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cold_rebuild": {
                    "type": "boolean",
                    "description": (
                        "Re-derive the subject tree from the geometry "
                        "instead of inheriting the standing shape. This "
                        "WILL move shelf paths and retire shelves, "
                        "orphaning addresses that digests and commissions "
                        "quote. Deliberate reset only."
                    ),
                },
            },
        },
    },
    {
        "name": "librarian_consult",
        "description": (
            "The front desk's structured consultation: FAQ fast path for "
            "known recurring questions, subject dispatch to matching "
            "shelves, general hybrid retrieval, and open research tickets "
            "near the query. Prefer this when answering a patron question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The patron's question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Results per channel (default 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "librarian_faq",
        "description": (
            "FAQ desk management. list (status candidate/active/stale/"
            "retired), approve (activate a candidate, optional answer "
            "rewrite + author byline), retire. Only approved entries are "
            "served on the consult fast path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "approve", "retire"],
                },
                "faq_id": {"type": "integer"},
                "answer": {"type": "string"},
                "author": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_closure",
        "description": (
            "Specialist closure — settle what a piece of work overturned. "
            "supersede: record that one memory refutes a claim in another "
            "(nothing is deleted; the older one afterwards arrives marked). "
            "close: the full reverse-order unwind for a finished report — "
            "supersede what it refuted, stale dependent FAQ answers, "
            "resolve findings it explains, close the ticket. list, retract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["list", "supersede", "close", "retract"]},
                "memory_id": {"type": "string"},
                "superseding_id": {"type": "string"},
                "report_memory_id": {"type": "string"},
                "commission_id": {"type": "integer"},
                "claim": {"type": "string"},
                "all_claims": {
                    "type": "boolean",
                    "description": (
                        "retract only: withdraw every correction between the "
                        "pair, not just the named claim. A pair can carry "
                        "several independent corrections, so this must be "
                        "deliberate."
                    ),
                },
                "reason": {"type": "string"},
                "scope": {"type": "string", "enum": ["claim", "whole"]},
                "actor": {"type": "string"},
                "supersedes": {"type": "array", "items": {"type": "object"}},
                "resolve_findings": {"type": "array",
                                     "items": {"type": "integer"}},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_desk",
        "description": (
            "The library's standing report — start here. What changed "
            "since you were last here, what needs your judgment, what was "
            "learned, and the state of the building. action='ack' marks "
            "the moment seen; reading alone never acknowledges."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["report", "ack"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_subjects",
        "description": (
            "Address the subject hierarchy. tree: the roster as an outline. "
            "show: one shelf, addressed loosely — a path, a name, or just a "
            "topic ('gnosticism') resolves to the most specific shelf "
            "holding it; ambiguity returns candidates. children: direct "
            "children. rename (path + name): give a shelf a human name, "
            "which survives roster rebuilds. Prefer `label` over `name` "
            "when speaking to a person."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["tree", "show", "children", "rename"],
                },
                "path": {"type": "string"},
                "name": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_ingress",
        "description": (
            "Receipted ingress dwell: hold content under custody before "
            "committing it to memory, giving temporal detection a window. "
            "hold (issue receipt + examine the dwelling window), review, "
            "settle, commit (needs tags), discard, examine, stats. "
            "Findings are advisory, not verdicts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["hold", "examine", "review", "settle",
                             "commit", "discard", "stats"],
                },
                "content": {"type": "string"},
                "receipt": {"type": "string"},
                "source": {"type": "string"},
                "origin": {"type": "string"},
                "author": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "number"},
                "summary": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_recall",
        "description": (
            "Cross-domain analogic recall: counterparts that occupy the "
            "same role in other subjects that the query occupies in its "
            "home subject (offset transport between subject centroids). "
            "Results are conjectures, labeled register: analogy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {
                    "type": "integer",
                    "description": "Counterparts per subject (default 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "librarian_commissions",
        "description": (
            "The commission queue: research tickets on the library's known "
            "gaps. list (by status), file (question), claim, report "
            "(attach a stored report memory), close."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "file", "claim", "report", "close"],
                },
                "commission_id": {"type": "integer"},
                "question": {"type": "string"},
                "report_memory_id": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
]
