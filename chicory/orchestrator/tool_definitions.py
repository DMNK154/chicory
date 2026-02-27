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
            "entangled events) and the void profile (latent attractor themes)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
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
]
