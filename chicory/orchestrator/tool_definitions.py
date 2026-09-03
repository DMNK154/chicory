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
                "actor": {
                    "type": "string",
                    "description": (
                        "Who is asking. Retrieval is not read-only here: it records demand and reweights the centroid geometry the shelf tree is derived from. Leave unset (defaults to 'patron') when relaying a person's own question. Set 'research' when querying on your own behalf - gathering sources for an FAQ answer, a commission report, or a digest - so the queries are recorded but not counted as the library's demand. Set 'probe' for tests and diagnostics."
                    ),
                },
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
                "actor": {
                    "type": "string",
                    "description": (
                        "Who is asking. Retrieval is not read-only here: it records demand and reweights the centroid geometry the shelf tree is derived from. Leave unset (defaults to 'patron') when relaying a person's own question. Set 'research' when querying on your own behalf - gathering sources for an FAQ answer, a commission report, or a digest - so the queries are recorded but not counted as the library's demand. Set 'probe' for tests and diagnostics."
                    ),
                },
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
                "actor": {
                    "type": "string",
                    "description": (
                        "Who is asking. Retrieval is not read-only here: it records demand and reweights the centroid geometry the shelf tree is derived from. Leave unset (defaults to 'patron') when relaying a person's own question. Set 'research' when querying on your own behalf - gathering sources for an FAQ answer, a commission report, or a digest - so the queries are recorded but not counted as the library's demand. Set 'probe' for tests and diagnostics."
                    ),
                },
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
            "The FAQ desk — the card file of crystallized answers, and the "
            "queue of drafts waiting on judgment. Only 'active' entries are "
            "served on the consult fast path; a 'candidate' is a mechanical "
            "draft nobody has verified yet. Workflow: list to see the queue "
            "and the tally across states, show to read one card with its "
            "sources, then approve with a rewritten answer or retire it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "show", "audit", "approve", "retire", "stale"],
                    "description": (
                        "list — the queue for one status, plus counts across "
                        "all four. show — one card hydrated with its "
                        "supporting memories' text, provenance, custody and "
                        "any blockers; this is what judgment needs, since "
                        "list returns ids only. approve — activate it, "
                        "normally with a rewritten answer and a byline. "
                        "audit — check every live card against the promotion bar as it "
                        "stands now, since tuning the bar leaves earlier "
                        "cards behind; pass demote to return the ones "
                        "below it to candidate. "
                        "retire — permanent; nothing reopens it. stale — "
                        "pull a served card back for rebuilding when the "
                        "question is still good but the answer is not."
                    ),
                },
                "faq_id": {"type": "integer"},
                "answer": {
                    "type": "string",
                    "description": (
                        "The approved answer. The mechanical draft is a "
                        "citation list, not prose — rewrite it."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": "Byline for the approval, e.g. specialist:<subject>.",
                },
                "supporting_memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Re-ground the card on approve. The draft rests on "
                        "whatever the promotion retrieval returned; if you "
                        "rewrote the answer from better sources, pass them "
                        "here so the served card's provenance matches its "
                        "text and serve verifies the right memories."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": ["candidate", "active", "stale", "retired"],
                    "description": "Which queue to list. Defaults to candidate.",
                },
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "meta_shelves",
        "description": (
            "The desk where a subject the library cannot retrieve by becomes "
            "a shelf. The gap net groups concepts that belong together — the "
            "Kabbalah vocabulary, the Tractatus in German, Leviathan in "
            "period spelling — and files each group as a PROPOSAL. A person "
            "confirms, because the discriminator cannot be automated: a "
            "game's config file scores better on every available measure "
            "than a real subject does. Confirming can also COMPLETE a group "
            "the grouping could not reach, from the ring that 'show' lists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "show", "confirm", "retire",
                             "materialize", "propose"],
                    "description": (
                        "list (default) — the proposal queue. show — one "
                        "charter, its members, and the ring of concepts "
                        "around it that could be added. confirm — assert "
                        "these concepts are one subject. retire — this is "
                        "not a subject. materialize — IRREVERSIBLE: creates "
                        "a tag per member so the roster derives the shelf; "
                        "there is no un-tag path. propose — re-run the pass."
                    ),
                },
                "slug": {"type": "string", "description": "Which charter."},
                "name": {
                    "type": "string",
                    "description": (
                        "Rename the charter's permanent address at confirm "
                        "time. No '+' or '/'."
                    ),
                },
                "display_name": {"type": "string"},
                "rationale": {
                    "type": "string",
                    "description": (
                        "Why these are one subject — the part only a person "
                        "knows, which the corpus does not encode anywhere."
                    ),
                },
                "add": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Concepts to add. Each must sit in the same material "
                        "and either share memories with the core or be a "
                        "spelling variant of a member."
                    ),
                },
                "drop": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "librarian_gaps",
        "description": (
            "The net of what this library is MISSING. Every other graph "
            "here connects what it holds; this one connects concepts the "
            "shelves discuss constantly and cannot be retrieved by, joined "
            "when the same documents raise them. Built by the wandering "
            "librarian's concept_gaps errand from what it read, not from "
            "the index. Use it to answer 'what does the library talk about "
            "but cannot find', and to see whether a bad answer was a "
            "holdings gap or an indexing one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["clusters", "region", "neighbors", "list", "rebuild"],
                    "description": (
                        "clusters (default) — the missing SUBJECTS, being "
                        "connected components of the net; a single "
                        "unfindable word is a nuisance, a component of them "
                        "is one problem with one fix. neighbors — what else "
                        "the material around a term raises (needs subject). "
                        "list — individual gaps, most concentrated first. "
                        "rebuild — recompute edges from the walk's "
                        "observations."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": "The concept to step out from, for neighbors.",
                },
                "limit": {"type": "integer"},
                "min_confirmations": {
                    "type": "integer",
                    "description": (
                        "How many passes must have seen a gap before it "
                        "counts. Recurrence is what separates a concept "
                        "from a one-off capitalised word."
                    ),
                },
            },
        },
    },
    {
        "name": "librarian_tag",
        "description": (
            "Attach a tag to memories that already exist. This is the one "
            "act the gap map licenses and nothing else can perform: the "
            "map says a concept is discussed in thousands of memories and "
            "no tag holds it, and this closes that. Pass concept= to take "
            "the target set from the wandering librarian's own reading "
            "rather than from a retrieval — retrieval could not find these, "
            "which is why the concept is a gap. Previews by default."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["preview", "apply"],
                    "description": (
                        "preview (default) changes nothing and reports what "
                        "would be attached. Tagging thousands of memories "
                        "moves tag centroids, and the shelf tree is derived "
                        "from those, so this is a considered act."
                    ),
                },
                "tag": {"type": "string", "description": "The tag to attach."},
                "concept": {
                    "type": "string",
                    "description": (
                        "Take the memories from this recorded concept gap "
                        "— the ones the walk actually read the term in."
                    ),
                },
                "memory_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Explicit targets, instead of concept.",
                },
                "author": {
                    "type": "string",
                    "description": "Byline recorded on every attachment.",
                },
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
