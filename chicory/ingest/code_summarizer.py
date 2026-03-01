"""Code-aware summarization — extract structure from source files.

Uses AST parsing for Python, regex-based extraction for other languages.
Produces compact summaries suitable for memory storage and retrieval.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional


def summarize_file(path: Path, base_dir: Path | None = None) -> Optional[str]:
    """Summarize a source file into a compact structural description.

    Returns None if the file type is unsupported or the file is empty.
    """
    suffix = path.suffix.lower()

    summarizers = {
        ".py": _summarize_python,
        ".js": _summarize_javascript,
        ".ts": _summarize_typescript,
        ".jsx": _summarize_javascript,
        ".tsx": _summarize_typescript,
        ".java": _summarize_java,
        ".go": _summarize_go,
        ".rs": _summarize_rust,
        ".c": _summarize_c_cpp,
        ".cpp": _summarize_c_cpp,
        ".h": _summarize_c_cpp,
        ".rb": _summarize_ruby,
        ".sh": _summarize_shell,
        ".md": _summarize_markdown,
        ".yaml": _summarize_config,
        ".yml": _summarize_config,
        ".toml": _summarize_config,
        ".json": _summarize_json,
        ".sql": _summarize_sql,
        ".css": _summarize_css,
        ".html": _summarize_html,
    }

    summarizer = summarizers.get(suffix)
    if summarizer is None:
        return None

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    if not source.strip():
        return None

    rel_path = str(path.relative_to(base_dir)) if base_dir else str(path)
    rel_path = rel_path.replace("\\", "/")

    try:
        summary = summarizer(source, rel_path)
    except Exception:
        # Fall back to basic summary on any parse error
        summary = _summarize_basic(source, rel_path, suffix)

    return summary


def _summarize_python(source: str, rel_path: str) -> str:
    """Summarize a Python file using AST parsing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _summarize_basic(source, rel_path, ".py")

    lines = [f"# {rel_path}"]
    lines.append(f"Lines: {len(source.splitlines())}")

    # Module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        lines.append(f"Purpose: {first_line}")

    # Imports (grouped)
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
    if imports:
        # Deduplicate and show top-level packages
        top_level = sorted(set(imp.split(".")[0] for imp in imports if imp))
        lines.append(f"Imports: {', '.join(top_level)}")

    # Module-level constants and assignments
    constants = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
    if constants:
        lines.append(f"Constants: {', '.join(constants[:10])}")

    # Classes
    classes = [n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)]
    if classes:
        lines.append("")
        lines.append("## Classes")
        for cls in classes:
            cls_doc = ast.get_docstring(cls)
            doc_part = f" — {cls_doc.split(chr(10))[0]}" if cls_doc else ""

            # Bases
            bases = []
            for base in cls.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.unparse(base))
            base_str = f"({', '.join(bases)})" if bases else ""

            lines.append(f"- {cls.name}{base_str}{doc_part}")

            # Methods
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for method in methods:
                sig = _python_signature(method)
                method_doc = ast.get_docstring(method)
                doc_part = f" — {method_doc.split(chr(10))[0]}" if method_doc else ""
                prefix = "async " if isinstance(method, ast.AsyncFunctionDef) else ""
                lines.append(f"  - {prefix}{method.name}{sig}{doc_part}")

    # Module-level functions
    functions = [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if functions:
        lines.append("")
        lines.append("## Functions")
        for func in functions:
            sig = _python_signature(func)
            func_doc = ast.get_docstring(func)
            doc_part = f" — {func_doc.split(chr(10))[0]}" if func_doc else ""
            prefix = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
            lines.append(f"- {prefix}{func.name}{sig}{doc_part}")

    return "\n".join(lines)


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract a compact function signature from an AST node."""
    args = node.args
    parts = []

    # Regular args (skip 'self' and 'cls')
    all_args = args.args
    for i, arg in enumerate(all_args):
        if arg.arg in ("self", "cls"):
            continue
        annotation = ""
        if arg.annotation:
            try:
                annotation = f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        parts.append(f"{arg.arg}{annotation}")

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    # Return type
    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    return f"({', '.join(parts)}){ret}"


def _summarize_javascript(source: str, rel_path: str) -> str:
    """Summarize JS/JSX using regex extraction."""
    return _summarize_js_ts(source, rel_path, "JavaScript")


def _summarize_typescript(source: str, rel_path: str) -> str:
    """Summarize TS/TSX using regex extraction."""
    return _summarize_js_ts(source, rel_path, "TypeScript")


def _summarize_js_ts(source: str, rel_path: str, lang: str) -> str:
    """Shared JS/TS summarizer."""
    lines = [f"# {rel_path}", f"Language: {lang}", f"Lines: {len(source.splitlines())}"]

    # Imports
    imports = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]", source)
    if imports:
        top_pkgs = sorted(set(
            imp.split("/")[0] for imp in imports
            if not imp.startswith(".")
        ))
        if top_pkgs:
            lines.append(f"External imports: {', '.join(top_pkgs[:15])}")

    # Exports
    exports = re.findall(r"export\s+(?:default\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+)", source)
    if exports:
        lines.append(f"Exports: {', '.join(exports[:15])}")

    # Classes
    classes = re.findall(r"class\s+(\w+)(?:\s+extends\s+(\w+))?", source)
    if classes:
        lines.append("")
        lines.append("## Classes")
        for name, base in classes:
            ext = f"({base})" if base else ""
            lines.append(f"- {name}{ext}")

    # Functions (named, exported, and arrow)
    funcs = re.findall(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", source
    )
    arrow_funcs = re.findall(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*\w+\s*)?=>",
        source,
    )
    all_funcs = funcs + arrow_funcs
    if all_funcs:
        lines.append("")
        lines.append("## Functions")
        for name, params in all_funcs[:20]:
            params_short = _shorten_params(params)
            lines.append(f"- {name}({params_short})")

    # Interfaces / Types (TS)
    interfaces = re.findall(r"(?:export\s+)?interface\s+(\w+)", source)
    types = re.findall(r"(?:export\s+)?type\s+(\w+)\s*=", source)
    if interfaces or types:
        lines.append("")
        lines.append("## Types")
        for name in interfaces:
            lines.append(f"- interface {name}")
        for name in types:
            lines.append(f"- type {name}")

    return "\n".join(lines)


def _summarize_java(source: str, rel_path: str) -> str:
    """Summarize Java using regex extraction."""
    lines = [f"# {rel_path}", "Language: Java", f"Lines: {len(source.splitlines())}"]

    pkg = re.search(r"package\s+([\w.]+);", source)
    if pkg:
        lines.append(f"Package: {pkg.group(1)}")

    imports = re.findall(r"import\s+([\w.]+);", source)
    if imports:
        pkgs = sorted(set(imp.rsplit(".", 1)[0] for imp in imports))
        lines.append(f"Imports: {', '.join(pkgs[:10])}")

    classes = re.findall(
        r"(?:public|private|protected)?\s*(?:abstract|static|final)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?",
        source,
    )
    if classes:
        lines.append("")
        lines.append("## Classes")
        for name, base, ifaces in classes:
            parts = [name]
            if base:
                parts[0] += f"({base})"
            if ifaces:
                parts[0] += f" implements {ifaces.strip()}"
            lines.append(f"- {parts[0]}")

    methods = re.findall(
        r"(?:public|private|protected)\s+(?:static\s+)?(?:[\w<>\[\],\s]+)\s+(\w+)\s*\(([^)]*)\)",
        source,
    )
    if methods:
        lines.append("")
        lines.append("## Methods")
        for name, params in methods[:20]:
            params_short = _shorten_params(params)
            lines.append(f"- {name}({params_short})")

    return "\n".join(lines)


def _summarize_go(source: str, rel_path: str) -> str:
    """Summarize Go using regex extraction."""
    lines = [f"# {rel_path}", "Language: Go", f"Lines: {len(source.splitlines())}"]

    pkg = re.search(r"package\s+(\w+)", source)
    if pkg:
        lines.append(f"Package: {pkg.group(1)}")

    imports = re.findall(r'"([^"]+)"', source[:source.find("func ") if "func " in source else len(source)])
    if imports:
        lines.append(f"Imports: {', '.join(imports[:10])}")

    structs = re.findall(r"type\s+(\w+)\s+struct", source)
    interfaces = re.findall(r"type\s+(\w+)\s+interface", source)
    if structs or interfaces:
        lines.append("")
        lines.append("## Types")
        for s in structs:
            lines.append(f"- struct {s}")
        for i in interfaces:
            lines.append(f"- interface {i}")

    funcs = re.findall(r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)", source)
    if funcs:
        lines.append("")
        lines.append("## Functions")
        for recv_name, recv_type, name, params in funcs[:20]:
            if recv_type:
                lines.append(f"- ({recv_type}).{name}({_shorten_params(params)})")
            else:
                lines.append(f"- {name}({_shorten_params(params)})")

    return "\n".join(lines)


def _summarize_rust(source: str, rel_path: str) -> str:
    """Summarize Rust using regex extraction."""
    lines = [f"# {rel_path}", "Language: Rust", f"Lines: {len(source.splitlines())}"]

    mods = re.findall(r"(?:pub\s+)?mod\s+(\w+)", source)
    if mods:
        lines.append(f"Modules: {', '.join(mods)}")

    uses = re.findall(r"use\s+([\w:]+)", source)
    if uses:
        crates = sorted(set(u.split("::")[0] for u in uses if u != "crate"))
        if crates:
            lines.append(f"Uses: {', '.join(crates[:10])}")

    structs = re.findall(r"(?:pub\s+)?struct\s+(\w+)", source)
    enums = re.findall(r"(?:pub\s+)?enum\s+(\w+)", source)
    traits = re.findall(r"(?:pub\s+)?trait\s+(\w+)", source)
    if structs or enums or traits:
        lines.append("")
        lines.append("## Types")
        for s in structs:
            lines.append(f"- struct {s}")
        for e in enums:
            lines.append(f"- enum {e}")
        for t in traits:
            lines.append(f"- trait {t}")

    impls = re.findall(r"impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)", source)
    funcs = re.findall(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)", source)
    if funcs:
        lines.append("")
        lines.append("## Functions")
        for name, params in funcs[:20]:
            lines.append(f"- {name}({_shorten_params(params)})")

    return "\n".join(lines)


def _summarize_c_cpp(source: str, rel_path: str) -> str:
    """Summarize C/C++ using regex extraction."""
    lang = "C++" if rel_path.endswith((".cpp", ".hpp", ".cc")) else "C"
    lines = [f"# {rel_path}", f"Language: {lang}", f"Lines: {len(source.splitlines())}"]

    includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', source)
    if includes:
        lines.append(f"Includes: {', '.join(includes[:10])}")

    structs = re.findall(r"(?:typedef\s+)?struct\s+(\w+)", source)
    if structs:
        lines.append(f"Structs: {', '.join(set(structs))}")

    classes = re.findall(r"class\s+(\w+)", source)
    if classes:
        lines.append(f"Classes: {', '.join(set(classes))}")

    funcs = re.findall(
        r"(?:[\w*]+\s+)+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?\{",
        source,
    )
    if funcs:
        lines.append("")
        lines.append("## Functions")
        for name, params in funcs[:20]:
            if name not in ("if", "for", "while", "switch", "catch"):
                lines.append(f"- {name}({_shorten_params(params)})")

    return "\n".join(lines)


def _summarize_ruby(source: str, rel_path: str) -> str:
    """Summarize Ruby using regex extraction."""
    lines = [f"# {rel_path}", "Language: Ruby", f"Lines: {len(source.splitlines())}"]

    requires = re.findall(r"require\s+['\"]([^'\"]+)['\"]", source)
    if requires:
        lines.append(f"Requires: {', '.join(requires[:10])}")

    classes = re.findall(r"class\s+(\w+)(?:\s*<\s*(\w+))?", source)
    modules = re.findall(r"module\s+(\w+)", source)
    if classes or modules:
        lines.append("")
        lines.append("## Classes/Modules")
        for name, base in classes:
            ext = f"({base})" if base else ""
            lines.append(f"- class {name}{ext}")
        for name in modules:
            lines.append(f"- module {name}")

    methods = re.findall(r"def\s+(\w+[?!=]?)\s*(?:\(([^)]*)\))?", source)
    if methods:
        lines.append("")
        lines.append("## Methods")
        for name, params in methods[:20]:
            lines.append(f"- {name}({params})")

    return "\n".join(lines)


def _summarize_shell(source: str, rel_path: str) -> str:
    """Summarize shell scripts."""
    lines = [f"# {rel_path}", "Language: Shell", f"Lines: {len(source.splitlines())}"]

    # Shebang
    if source.startswith("#!"):
        shebang = source.split("\n")[0]
        lines.append(f"Shebang: {shebang}")

    # First comment block as description
    comment_lines = []
    for line in source.splitlines()[1:]:
        if line.startswith("#") and not line.startswith("#!"):
            comment_lines.append(line.lstrip("# ").strip())
        elif line.strip():
            break
    if comment_lines:
        lines.append(f"Purpose: {' '.join(comment_lines[:3])}")

    funcs = re.findall(r"(?:function\s+)?(\w+)\s*\(\)\s*\{", source)
    if funcs:
        lines.append("")
        lines.append("## Functions")
        for name in funcs[:15]:
            lines.append(f"- {name}()")

    return "\n".join(lines)


def _summarize_markdown(source: str, rel_path: str) -> str:
    """Summarize Markdown by extracting headers and structure."""
    lines = [f"# {rel_path}", "Type: Markdown", f"Lines: {len(source.splitlines())}"]

    # First paragraph as description
    paragraphs = re.split(r"\n\s*\n", source.strip())
    if paragraphs:
        first = paragraphs[0].strip()
        if not first.startswith("#"):
            lines.append(f"Description: {first[:200]}")

    # Extract headers hierarchy
    headers = re.findall(r"^(#{1,6})\s+(.+)$", source, re.MULTILINE)
    if headers:
        lines.append("")
        lines.append("## Structure")
        for hashes, title in headers[:20]:
            indent = "  " * (len(hashes) - 1)
            lines.append(f"{indent}- {title.strip()}")

    return "\n".join(lines)


def _summarize_config(source: str, rel_path: str) -> str:
    """Summarize YAML/TOML config files."""
    lines = [f"# {rel_path}", "Type: Config", f"Lines: {len(source.splitlines())}"]

    # Extract top-level keys
    if rel_path.endswith((".yaml", ".yml")):
        keys = re.findall(r"^(\w[\w-]*):", source, re.MULTILINE)
    else:  # TOML
        keys = re.findall(r"^\[([^\]]+)\]", source, re.MULTILINE)
        if not keys:
            keys = re.findall(r"^(\w[\w-]*)\s*=", source, re.MULTILINE)

    if keys:
        lines.append(f"Top-level keys: {', '.join(keys[:15])}")

    return "\n".join(lines)


def _summarize_json(source: str, rel_path: str) -> str:
    """Summarize JSON by extracting top-level structure."""
    import json

    lines = [f"# {rel_path}", "Type: JSON", f"Lines: {len(source.splitlines())}"]

    try:
        data = json.loads(source)
        if isinstance(data, dict):
            lines.append(f"Top-level keys: {', '.join(list(data.keys())[:15])}")
        elif isinstance(data, list):
            lines.append(f"Array with {len(data)} items")
            if data and isinstance(data[0], dict):
                lines.append(f"Item keys: {', '.join(list(data[0].keys())[:10])}")
    except json.JSONDecodeError:
        lines.append("(invalid JSON)")

    return "\n".join(lines)


def _summarize_sql(source: str, rel_path: str) -> str:
    """Summarize SQL files."""
    lines = [f"# {rel_path}", "Language: SQL", f"Lines: {len(source.splitlines())}"]

    tables_created = re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", source, re.IGNORECASE)
    if tables_created:
        lines.append(f"Creates tables: {', '.join(tables_created)}")

    views = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)", source, re.IGNORECASE)
    if views:
        lines.append(f"Creates views: {', '.join(views)}")

    procs = re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+(\w+)", source, re.IGNORECASE)
    if procs:
        lines.append(f"Procedures/Functions: {', '.join(procs)}")

    return "\n".join(lines)


def _summarize_css(source: str, rel_path: str) -> str:
    """Summarize CSS files."""
    lines = [f"# {rel_path}", "Type: CSS", f"Lines: {len(source.splitlines())}"]

    # Count selectors
    selectors = re.findall(r"^([^{/\n@][^{]*)\{", source, re.MULTILINE)
    lines.append(f"Selectors: {len(selectors)}")

    # Media queries
    media = re.findall(r"@media\s+([^{]+)", source)
    if media:
        lines.append(f"Media queries: {len(media)}")

    # CSS variables
    variables = re.findall(r"--[\w-]+", source)
    if variables:
        lines.append(f"CSS variables: {len(set(variables))}")

    return "\n".join(lines)


def _summarize_html(source: str, rel_path: str) -> str:
    """Summarize HTML files."""
    lines = [f"# {rel_path}", "Type: HTML", f"Lines: {len(source.splitlines())}"]

    title = re.search(r"<title>([^<]+)</title>", source, re.IGNORECASE)
    if title:
        lines.append(f"Title: {title.group(1).strip()}")

    scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', source, re.IGNORECASE)
    if scripts:
        lines.append(f"Scripts: {', '.join(scripts[:5])}")

    return "\n".join(lines)


def _summarize_basic(source: str, rel_path: str, suffix: str) -> str:
    """Fallback summary for files that failed specialized parsing."""
    lines = [f"# {rel_path}", f"Type: {suffix}", f"Lines: {len(source.splitlines())}"]

    # First non-empty line as hint
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#!"):
            lines.append(f"First line: {stripped[:100]}")
            break

    return "\n".join(lines)


def _shorten_params(params: str, max_len: int = 80) -> str:
    """Shorten a parameter list if it's too long."""
    params = params.strip()
    if len(params) <= max_len:
        return params
    # Count params and abbreviate
    parts = [p.strip() for p in params.split(",")]
    if len(parts) <= 3:
        return params[:max_len] + "..."
    return f"{parts[0]}, {parts[1]}, ... ({len(parts)} params)"


# Supported extensions for code summarization
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".rb", ".sh",
    ".md", ".yaml", ".yml", ".toml", ".json", ".sql", ".css", ".html",
}
