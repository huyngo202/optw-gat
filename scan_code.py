#!/usr/bin/env python3
"""
scan_code.py

Gom tất cả file .py trong một thư mục thành 1 file markdown.
"""
from pathlib import Path
import argparse
import datetime
import sys
import io
import os

DEFAULT_EXCLUDE_DIRS = {"__pycache__", ".git", "venv", "env", ".venv", "node_modules", "build", "dist"}

def find_py_files(root: Path, recursive: bool, exclude_dirs: set):
    if not root.exists():
        return []
    py_files = []
    if recursive:
        for p in root.rglob("*.py"):
            # skip if any parent directory in exclude_dirs
            if any(part in exclude_dirs for part in p.parts):
                continue
            # skip hidden files? (optional) keep them
            py_files.append(p)
    else:
        for p in root.glob("*.py"):
            if any(part in exclude_dirs for part in p.parts):
                continue
            py_files.append(p)
    # sort by relative path for stable order
    py_files.sort(key=lambda p: str(p.relative_to(root)))
    return py_files

def safe_read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            return f"# Could not read file {path}: {e}\n"

def render_file_md(path: Path, root: Path):
    rel = path.relative_to(root)
    stat = path.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds")
    header = f"## `{rel}`\n\n"
    meta = f"- **Path**: `{path}`\n- **Size**: {stat.st_size} bytes\n- **Last modified**: {mtime}\n\n"
    content = safe_read_text(path)
    # ensure code fence doesn't break if triple backticks are inside file:
    # use ```python and if file contains ``` we can use ```\u200b`` with zero-width space inside fence marker? Simpler: use triple backticks and escape by replacing any occurrence of ``` with ```\u200b
    content_sanitized = content.replace("```", "```​")  # adds ZWSP preventing fence break
    code_block = "```python\n" + content_sanitized + "\n```\n\n"
    return header + meta + code_block

def main():
    parser = argparse.ArgumentParser(description="Gom tất cả file .py thành 1 file markdown")
    parser.add_argument("-d", "--dir", default=".", help="Thư mục gốc để tìm file .py (default: current dir)")
    parser.add_argument("-o", "--output", default="all_python_code.md", help="File markdown đầu ra")
    parser.add_argument("-r", "--recursive",default=True, action="store_true", help="Đệ quy (tìm trong subfolders)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Tên các thư mục cần loại trừ (space-separated)")
    parser.add_argument("--prepend-toc", action="store_true", help="Thêm mục lục (TOC) phía trên")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    exclude_set = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude)

    py_files = find_py_files(root, args.recursive, exclude_set)
    if not py_files:
        print("Không tìm thấy file .py nào.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    parts = []
    title = f"# Gom mã nguồn Python từ `{root}`\n\nGenerated: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}\n\n"
    parts.append(title)

    if args.prepend_toc:
        parts.append("## Mục lục\n\n")
        for p in py_files:
            rel = p.relative_to(root)
            anchor = str(rel).replace("/", "-").replace(" ", "-")
            parts.append(f"- [{rel}](#{anchor})\n")
        parts.append("\n")

    for p in py_files:
        parts.append(render_file_md(p, root))

    # write atomically
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(parts)
    tmp.replace(out_path)
    print(f"Wrote {len(py_files)} files to {out_path}")

if __name__ == "__main__":
    main()
