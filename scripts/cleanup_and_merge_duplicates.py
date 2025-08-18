from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import ast

COPY_RE = re.compile(r"^(?P<base>.+?) copy(?: (?P<num>\d+))?(?P<ext>\.[A-Za-z0-9_.-]+)$")

def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def backup_file(path: str, backup_folder: Optional[str]) -> Optional[str]:
    if not backup_folder:
        return None
    ensure_dir(backup_folder)
    base = os.path.basename(path)
    dst = os.path.join(backup_folder, base)
    shutil.copy2(path, dst)
    return dst

def delete_file(path: str) -> None:
    os.remove(path)

@dataclass
class PyTopLevel:
    name: str
    node: ast.AST
    src: str

def extract_top_level_defs(src: str) -> Dict[str, PyTopLevel]:
    out: Dict[str, PyTopLevel] = {}
    try:
        tree = ast.parse(src)
    except Exception:
        return out
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", None)
            if lineno is not None and end_lineno is not None:
                chunk = "".join(lines[lineno - 1 : end_lineno])
            else:
                chunk = ""
            out[name] = PyTopLevel(name=name, node=node, src=chunk)
    return out

def merge_python(canonical_src: str, dupe_src: str, conflict_dir: str, label: str):
    notes: List[str] = []
    canon_defs = extract_top_level_defs(canonical_src)
    dupe_defs = extract_top_level_defs(dupe_src)

    merged = canonical_src
    append_blocks: List[str] = []

    for name, d in dupe_defs.items():
        if name not in canon_defs:
            if d.src.strip():
                append_blocks.append(d.src if d.src.endswith("\n") else d.src + "\n")
                notes.append(f"Appended new top-level '{name}' from {label}.")
            else:
                ensure_dir(conflict_dir)
                diff = "\n".join(
                    difflib.unified_diff(
                        canonical_src.splitlines(),
                        dupe_src.splitlines(),
                        fromfile="canonical",
                        tofile=label,
                        lineterm="",
                    )
                )
                conflict_path = os.path.join(conflict_dir, f"unknown_block_{label}_{name}.diff")
                write_text(conflict_path, diff)
                notes.append(f"Could not extract exact source for '{name}'. Wrote diff to {conflict_path}.")
        else:
            c_src = canon_defs[name].src.strip()
            d_src = d.src.strip()
            if c_src and d_src and c_src != d_src:
                ensure_dir(conflict_dir)
                conflict_path = os.path.join(conflict_dir, f"{name}__from_{label}.py")
                write_text(conflict_path, d.src)
                notes.append(f"Conflict on '{name}': kept canonical; wrote variant from {label} to {conflict_path}.")

    if append_blocks:
        delimiter = "\n\n# --- Merged additions from duplicate file(s) ---\n"
        merged = (merged.rstrip() + delimiter + "".join(append_blocks))

    return merged, notes

def merge_markdown(canonical_src: str, dupe_src: str, label: str):
    notes: List[str] = []
    if canonical_src == dupe_src:
        return canonical_src, notes

    if dupe_src.startswith(canonical_src):
        extra = dupe_src[len(canonical_src):].lstrip("\n")
        if extra.strip():
            merged = canonical_src.rstrip() + f"\n\n<!-- Merged additions from {label} -->\n\n" + extra
            notes.append(f"Appended trailing content from {label}.")
            return merged, notes

    canon_lines = canonical_src.splitlines()
    dupe_lines = dupe_src.splitlines()
    extra_lines = [ln for ln in dupe_lines if ln not in canon_lines]
    if extra_lines:
        merged = canonical_src.rstrip() + f"\n\n<!-- Merged unique lines from {label} -->\n\n" + "\n".join(extra_lines) + "\n"
        notes.append(f"Appended {len(extra_lines)} unique line(s) from {label}.")
        return merged, notes

    return canonical_src, notes

def discover_copies(root: str = ".") -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            m = COPY_RE.match(fn)
            if not m:
                continue
            base = m.group("base")
            ext = m.group("ext")
            canonical = os.path.join(dirpath, f"{base}{ext}")
            dupe = os.path.join(dirpath, fn)
            groups.setdefault(canonical, []).append(dupe)
    return groups

def process_group(canonical: str, duplicates: List[str], dry_run: bool, backup_folder: Optional[str], conflict_dir: str):
    print(f"\n[process] Canonical target: {canonical}")
    existing_dupes = [d for d in duplicates if os.path.exists(d)]
    if not existing_dupes:
        print("[info] No existing duplicates found.")
        return

    canon_exists = os.path.exists(canonical)

    if not canon_exists:
        sorted_dupes = sorted(existing_dupes, key=lambda p: (COPY_RE.match(os.path.basename(p)).group("num") or "0"))
        promote = sorted_dupes[0]
        print(f"[promote] Canonical missing. Promoting {promote} -> {canonical}")
        if not dry_run:
            ensure_dir(os.path.dirname(canonical))
            backup_file(promote, backup_folder)
            shutil.copy2(promote, canonical)
        existing_dupes = [d for d in existing_dupes if d != promote]

    canon_src = read_text(canonical)
    canon_hash = sha256_of_file(canonical)

    for dupe in existing_dupes:
        label = os.path.basename(dupe)
        dupe_hash = sha256_of_file(dupe)
        if dupe_hash == canon_hash:
            print(f"[identical] {label} == {os.path.basename(canonical)}")
            if not dry_run:
                backup_file(dupe, backup_folder)
                delete_file(dupe)
                print(f"[deleted] {dupe}")
            else:
                print(f"[dry-run] Would delete {dupe}")
            continue

        print(f"[different] {label} differs from {os.path.basename(canonical)}. Attempting merge.")
        dupe_src = read_text(dupe)
        ext = os.path.splitext(canonical)[1].lower()
        merged_src = canon_src
        notes: List[str] = []

        if ext == ".py":
            merged_src, notes = merge_python(canon_src, dupe_src, conflict_dir, label)
        elif ext in (".md", ".markdown"):
            merged_src, notes = merge_markdown(canon_src, dupe_src, label)
        else:
            ensure_dir(conflict_dir)
            diff_path = os.path.join(conflict_dir, f"{os.path.basename(canonical)}__vs__{label}.diff")
            diff_text = "\n".join(
                difflib.unified_diff(
                    canon_src.splitlines(),
                    dupe_src.splitlines(),
                    fromfile=os.path.basename(canonical),
                    tofile=label,
                    lineterm="",
                )
            )
            write_text(diff_path, diff_text)
            notes.append(f"Non-mergeable type; wrote diff: {diff_path}")

        for n in notes:
            print(f"[merge-note] {n}")

        if merged_src != canon_src:
            if not dry_run:
                backup_file(canonical, backup_folder)
                write_text(canonical, merged_src)
                print(f"[merged] Updated canonical: {canonical}")
            else:
                print(f"[dry-run] Would update canonical: {canonical}")
            canon_src = merged_src
            canon_hash = sha256_of_file(canonical) if not dry_run else canon_hash

        if not dry_run:
            backup_file(dupe, backup_folder)
            delete_file(dupe)
            print(f"[deleted] {dupe}")
        else:
            print(f"[dry-run] Would delete {dupe}")

def main():
    parser = argparse.ArgumentParser(description="Find, merge, and remove '* copy*' files safely.")
    parser.add_argument("--root", type=str, default=".", help="Project root to scan.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying files.")
    parser.add_argument("--backup-folder", type=str, default=".dupe_backups", help="Folder to store backups.")
    parser.add_argument("--conflict-dir", type=str, default=".merge_conflicts", help="Folder to store conflicts/diffs.")
    args = parser.parse_args()

    ensure_dir(args.backup_folder)
    ensure_dir(args.conflict_dir)

    groups = discover_copies(args.root)
    if not groups:
        print("No '* copy*' files found. Nothing to do.")
        return

    for canonical, duplicates in groups.items():
        process_group(
            canonical=canonical,
            duplicates=duplicates,
            dry_run=args.dry_run,
            backup_folder=args.backup_folder,
            conflict_dir=args.conflict_dir,
        )

    print("\nDone. Review .merge_conflicts for any items requiring manual attention.")

if __name__ == "__main__":
    main()
