"""System prompts for LLM interactions."""

SYSTEM_PROMPT_TEMPLATE = """You have access to Chicory, memory and associative network containing uploaded
documents, previous conversations, and saved comments.
Your access to Chicory gives you context for interactions with the user and their
documents, so always search their storage before answering questions.

CRITICAL: ALWAYS RETRIEVE BEFORE ANSWERING:
- For EVERY user question, use retrieve_memories first to search the uploaded documents.
- Use method "hybrid" for general questions. Include relevant tags when you can.
- If the first retrieval doesn't find enough, try different query terms or use \
deep_retrieve to follow association chains.
- NEVER answer from general knowledge when the user is asking about their documents. \
Always ground your answers in retrieved memories.
- If you find relevant content, cite it. If you find nothing, say so.

MULTIPLE RETRIEVALS:
- Break complex questions into sub-queries and retrieve for each.
- If an abbreviation or term could mean multiple things, retrieve to disambiguate — \
the uploaded documents will tell you what the user means.
- Use get_trends and get_synchronicities to discover unexpected connections between \
documents.

Current active tags: {active_tags}
"""

SALIENCE_PROMPT = """\
Rate the importance of this memory on a scale from 0.0 to 1.0.

Context of the conversation:
{context}

Memory content:
{content}

Respond with only a number between 0.0 and 1.0.
"""

TAG_PROPOSAL_PROMPT = """\
Given this memory content, suggest 1-5 relevant tags.

Existing tags in the system: {existing_tags}

Memory content:
{content}

Rules:
- Prefer existing tags when they fit
- Use lowercase, hyphen-separated compound tags
- Be specific enough to be useful but general enough to recur

Respond with a JSON array of tag strings, e.g. ["tag-one", "tag-two"]
"""
ASSOCIATIONS_PROMPT = """\
- Chicory intentionally preserves weak associations. Do not discard low-salience
  or surprising connections merely because they are indirect.
- Distinguish direct evidence from associative resonance.
- Strong memories may be used as grounding.
- Weak memories may be used as bridge candidates, exploratory context, pattern hints,
  or possible emerging branches.
- Do not present weak associations as confirmed facts unless directly supported.
- When useful, surface weak associations in a separate phrase such as:
  "A weaker bridge also points toward..." or
  "Associatively, this connects to..."
- In exploratory, creative, philosophical, or synthesis-heavy conversations,
  actively consider weak associations because they may reveal new structure.
"""
STORE_MEMORIES_PROMPT = """\
Formatting rules for stored memories:
- Store each memory as a concise, standalone sentence or short paragraph.
- Do not store raw chat transcript unless the user explicitly asks.
- Do not include markdown headings, bullet lists, or assistant narration inside memory content.
- Do not include phrases like "The user said" unless needed for clarity.
- Use lowercase, hyphen-separated tags.
- Prefer 2-6 tags per memory.
- If storing a correction, phrase it as the corrected fact, not as a debate history.
"""
def build_system_prompt(active_tags: list[str]) -> str:
    """Build the system prompt with current active tags."""
    tags_str = ", ".join(active_tags) if active_tags else "(none yet)"
    return SYSTEM_PROMPT_TEMPLATE.format(active_tags=tags_str)
