---
description: Work the Chicory FAQ desk — list the cards awaiting judgment, open one, read its sources, and approve, retire, or send it back. Use when the patron wants to review, approve, or clean up FAQ entries.
---

# The FAQ Desk

The FAQ is the only text this library serves *before* retrieval. `librarian_consult` checks it first, so an approved card short-circuits the shelves for every matching question from then on. That makes approval the highest-leverage judgment in the system and the one most worth slowing down for.

**Arguments:** `$ARGUMENTS` — optional. A number opens that card directly. A word or phrase filters the queue to matching questions. Empty opens the desk.

If the Chicory MCP tools are unavailable, say so and stop.

**Attribute your reading.** Every retrieval you make while working the desk — checking a draft against the shelves, gathering better sources for a rewrite — passes `actor="specialist:<subject>"` (or `actor="research"` if no subject fits). Retrieval records demand and reweights the shelf geometry; unattributed, the queries you type to *answer* a card come back as new cards of their own. The patron's own questions, if you relay any, stay unattributed.

## What the states mean

| state | meaning |
|---|---|
| `candidate` | The demand ledger saw a question recur and drafted an answer mechanically. **Not served.** Waiting on a person. |
| `active` | Judged and servable. Served on the consult fast path. |
| `stale` | Was served; a source went missing or was refuted. Not served. The ledger will re-draft it back to `candidate`. |
| `retired` | Terminal. Nothing reopens it. |

## The shift

### 1. Open the desk
`librarian_faq` action `list` (status `candidate`). Report the tally across all four states, then the queue — numbered, one line each: question, hit count, source count, age.

If the candidate queue is empty, say so and check `active` and `stale` before concluding there is nothing to do. An empty candidate queue with three stale cards is a different situation from an empty desk.

Then ask which one they want, or take it from `$ARGUMENTS`.

### 2. Open the card
`librarian_faq` action `show` with the `faq_id`. **Always `show` before judging** — `list` returns a draft preview and source *ids*, which is enough to recognise a card and never enough to rule on one.

Present it in this order:

1. **The question**, and how many times it was asked (`hit_count` is serve count; the demand cluster behind it is what promoted it).
2. **The draft**, described honestly. A mechanical draft (`author: librarian:mechanical`) is a bullet list of source summaries — for PDF chunks that means `filename [chunk/total] — heading`, which is a pointer, not an answer. Say that plainly rather than presenting it as a draft that needs polish.
3. **The sources**, one at a time: summary, the passage itself, tier, custody, and whether it is `usable`. Mark which ones actually bear on the question — retrieval ranks by similarity, and the strongest passage is often not first.
4. **Blockers**, if any. Never approve past a blocker.

### 3. Judge

- **Approve** — only when the sources genuinely answer the question. Write the answer yourself: prose that answers it, citing memory ids inline where a claim rests on one, and naming what the library does *not* cover if the answer is partial. `librarian_faq` action `approve` with `faq_id`, your rewritten `answer`, and an `author` byline (`specialist:<subject>`, or the patron's name if they dictated it). Never approve with the mechanical draft as the answer.
- **Retire** — the question is malformed, duplicated, or the shelves cannot support any answer. If the question is *real* but unanswerable, file a commission first (`librarian_commissions` action `file`) so the gap gets researched, then retire.
- **Stale** — for an `active` card whose answer has gone wrong but whose question still stands. The ledger re-drafts it.
- **Leave it** — a perfectly good default. An unjudged candidate serves nothing and harms nothing.

### 4. Close
Summarise what changed: approved with bylines, retired with reasons, commissions filed. If cards were left unjudged, say which and why.

## Boundaries

- **Approval is judgment, not formatting.** If you cannot verify a claim against the retrieved sources, do not put it in the answer.
- **Do not approve a card whose sources you have not read** in this session. `show` returns them; read them.
- **Do not retire to tidy the queue.** A retired card is permanent and its question stops being asked.
- **Custody stays with the patron.** Nothing here releases or archives a memory; if a source is quarantined, that is a blocker to report, not to route around.
- An approved answer is served ahead of the shelves to everyone, indefinitely, until something refutes it. Weigh it that way.
