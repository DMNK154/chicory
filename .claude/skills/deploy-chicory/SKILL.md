---
description: Deploy a Chicory MCP memory server into the current project
allowed-tools: Bash(*), PowerShell(*), Read(*), Write(*), Edit(*), Glob(*), Grep(*)
---

# Deploy Chicory

Set up a Chicory MCP memory server for the current project. This configures the MCP connection, ensures the database is ready, and optionally ingests the codebase.

**Arguments:** `$ARGUMENTS` — optional path(s) to ingest. If provided, these are the directories or files to ingest after setup. If omitted, ask the user what they want to ingest.

## Context

- Chicory package location: `C:/chicory`
- Python with chicory installed: `C:/chicory/.venv/Scripts/python`
- MCP module: `chicory.mcp.server` (stdio transport)
- Default DB location: `~/.chicory/chicory.db` (override via `CHICORY_DB_PATH` env var)
- The project to deploy into is the **current working directory**: !`cd`

## Steps

### 1. Write `.mcp.json` in the project root

Create (or update) `.mcp.json` in the current working directory with a `chicory` server entry:

```json
{
  "mcpServers": {
    "chicory": {
      "type": "stdio",
      "command": "C:/chicory/.venv/Scripts/python",
      "args": ["-m", "chicory.mcp.server"],
      "env": {
        "CHICORY_DB_PATH": "<project-specific DB path>"
      }
    }
  }
}
```

**DB path convention:** `~/.chicory/<project-name>.db` where `<project-name>` is the current directory's basename, lowercased and hyphenated. For example, if deploying into `C:\Users\wrwar\projects\MyApp`, the DB path is `C:/Users/wrwar/.chicory/myapp.db`.

If `.mcp.json` already exists and has other servers defined, **merge** the chicory entry — don't overwrite the file. If a `chicory` entry already exists, ask the user before overwriting.

### 2. Ensure the database directory exists

Run: `mkdir -p ~/.chicory` (or PowerShell equivalent). The SQLite database and schema are created automatically on first server startup — no manual init needed.

### 3. Codebase ingestion

**Stop and ask the user for permission** before ingesting. Explain that ingestion will:
- Scan source files for structural summaries (classes, functions, imports)
- Store them as memories so they can be retrieved via `retrieve_memories` instead of re-reading files
- Take a few minutes depending on codebase size

**Ingestion target:** If `$ARGUMENTS` was provided, use that as the path(s) to ingest. Otherwise, ask the user which directory/directories they want to ingest (suggest the project root as default, but let them specify subdirectories, specific patterns, etc.).

If the user approves, tell them to restart Claude Code first (so the new MCP server loads), then use the `ingest_codebase` MCP tool with:
- `path`: the target directory they specified
- `file_patterns`: if the user specified specific patterns (e.g. `["*.py", "src/**/*.ts"]`)
- `exclude_patterns`: if the user wants to skip certain directories

### 4. Summary

Report what was configured:
- MCP config path
- Database path
- Ingestion target path(s)
- Whether ingestion is pending (needs restart first)

Remind the user they need to **restart Claude Code** for the MCP server to be picked up.
