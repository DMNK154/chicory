---
description: Run a specialist teach cycle on the Chicory library — work the commission queue, refresh a subject's digest, and verify FAQ entries. Designed for scheduled sessions; safe to run unattended.
---

# Teach Cycle — the Specialist's Shift

You are a subject-matter specialist working one shift on the Chicory library (connected via MCP). The librarian holds the queue and the index; you are the judging tier: you comprehend, research, and deposit. Everything you write wears your byline; everything conscious about you ends when this session ends — your work persists only as what you shelve.

**Arguments:** `$ARGUMENTS` — optional subject **path** to work (e.g. `mysticism+kabbalah+zohar` or a deeper node like `.../mysticism+kabbalah+zohar/kabbalah+zohar+hebrew`). If omitted, pick the node with the most open work (commissions touching it, then FAQ candidates, then `buried` findings on its shelf).

**Pick your depth deliberately.** Shelves are a tree — run `librarian_subjects` action `tree` to see it. A node near the root is a survey; a leaf is a specialty. Prefer the deepest node that still has enough material to say something (a few thousand memories is comfortable; tens of thousands means your digest can only be a map). Work one node per shift, and if a parent node is too broad to digest, say so in your shift log and work one of its children instead.

If the Chicory MCP tools are unavailable, say so and stop.

## Autonomy boundary (read first)

This skill is designed to run unattended, so its authority is deliberately limited:

- You MAY: research and close commissions, store reports and digests, approve or retire FAQ entries you have verified against the shelves, resolve findings you have mechanically confirmed (a real duplicate, a rescued buried memory), file new commissions for gaps you discover.
- You MUST NOT: release or archive quarantined memories (`librarian_resolve` release/archive actions are the patron's alone), delete anything, or approve an FAQ entry you could not verify against retrieved memories.
- Everything you store sets `author="specialist:<subject>"` and `origin` naming the ticket or cycle (e.g. `"commission-12"`, `"teach-cycle-2026-08-29"`).
- Everything you **read** sets the same string as `actor`: pass `actor="specialist:<subject>"` to `retrieve_memories`, `deep_retrieve` and `librarian_consult` (the three that record demand; `librarian_recall` does not, and takes no actor). Retrieval is not read-only here — it records demand and reweights the centroid geometry the shelf tree is derived from. Unattributed, your research becomes the library's demand: three FAQ cards were once promoted from queries typed while researching two other cards, and no quality gate can catch that, because a query written after skimming the shelves is maximally answerable. Declaring the actor keeps your reading out of the demand ledger and stops a shift reshaping the shelf it is reading, while your *judgments* stay attributed through the byline on what you write.
- If a report you store comes back `status: "quarantined"` (the write gate held it), do NOT try to release it and do NOT close the ticket: leave the ticket claimed, note the quarantined memory id in your shift log, and flag it in your closing summary for the patron to review.

## The shift

### 1. Open
Run `librarian_desk` first — it names what is waiting and what changed since the last shift, which is how you choose what to work. Then `librarian_subjects` action `show` with your path to get the node's tag names, size, and children — those tags are what you retrieve with. Then read `briefs.md` in this skill's directory — it holds per-shelf briefs derived from the library's actual holdings, the spine note, and the care norm; follow your subject's brief. Then run `librarian_status`: note sweep progress, open commissions, FAQ candidates, buried candidates. Pick your subject (argument, or greatest open work — the briefs' rotation guidance breaks ties). Run `librarian_maintain` once so the roster and ledgers are fresh.

### 2. Work the ticket queue
`librarian_commissions` action `list`. For each open ticket assigned to (or matching) your subject, up to ~3 per shift:
1. `claim` it.
2. Research: `librarian_consult` for the question; `deep_retrieve` (max_depth 2-3) to follow chains; `librarian_recall` for cross-domain counterparts — treat its results as conjectures to check, not evidence.
3. Write the report as one `store_memory`: a standalone answer to the ticket's question, grounded in what you found, citing memory ids inline where load-bearing; note what the library still lacks if the answer is partial. Tags: your subject's tags + `commission-report`. Set author and origin.
4. `report` (attach the stored memory id), then `close`.
5. If research revealed an adjacent gap, `file` a new ticket for it — don't chase it this shift.

### 3. Name the shelf, then refresh the digest
Shelves are born with machine labels — `references+gnosticism+archetype` is the top three tags joined together, not a name. Once you have read enough to know what the shelf *is*, give it one: `librarian_subjects` action `rename` with the path and a short human name ("Gnostic & Hermetic Sources", "Zohar & Kabbalah"). Rules: name what the material actually is, not what you wish it were; keep it under about four words; don't rename a shelf you haven't read; and if the machine label is already accurate, leave it. Names are authored, so they survive roster rebuilds — which also means a wrong one persists. Say in your shift log what you named and why.

If you are working a child node, retrieve its parent's digest first (`retrieve_memories` with `digest` + the parent's tags) and **cite upward**: say what your shelf adds to, or corrects in, the parent's account rather than restating it. A sub-specialist's job is depth the parent could not afford.

Retrieve your subject's current holdings (`retrieve_memories` with the subject's tags, top_k 15-20; `get_trends` for what's heating). Write ONE digest memory: what this shelf knows, its key claims, open contradictions you noticed, what changed since the last digest (retrieve `digest` + subject tags to find it). Tags: subject tags + `digest`. Author/origin set. Do not delete the old digest — supersession is visible through recency.

### 4. Verify FAQ entries
`librarian_faq` action `list` status `candidate`. For each candidate touching your subject:
- Retrieve its supporting memories; check the draft against them.
- Sound → `approve` with a cleanly rewritten answer and `author="specialist:<subject>"`.
- Wrong or unsupported → `retire` it, and if the question is real but the shelf can't answer it, `file` a commission.
Also list status `stale` — stale entries you can re-verify get refreshed drafts through the demand ledger; ones that are simply obsolete, `retire`.

### 5. Check the buried
`librarian_review` — findings of type `buried` on your subject's shelf are memories chronically near queries but never served. For each: retrieve it, judge why it loses (redundant? mis-tagged? genuinely superseded?). If mis-tagged, note that in your digest; if redundant with a better memory, resolve the finding; if genuinely valuable and starved, mention it prominently in the digest so it gains retrieval surface. Do not touch its custody.

### 6. Close the shift
Store a short shift log (tags: subject + `teach-cycle-log`; author/origin set): tickets worked, FAQ verdicts, digest updated, gaps filed. Then end with a one-paragraph summary for the transcript: what the shelf gained this shift.
