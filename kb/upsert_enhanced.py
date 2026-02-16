"""
Enhanced Upsert Module for Resume Context Builder

Features:
- Multiple context database support
- Proper file deletion
- Time-based filtering
- Information decay integration
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import re
import hashlib
from datetime import datetime

from context_packager_data.tokenizer import get_encoding

# Import from enhanced database module
from kb.db_enhanced import (
    get_engine,
    upsert_chunks,
    get_file_index,
    upsert_file_index,
    delete_chunks_by_path,
    delete_file_index,
    fetch_chunk_ids_by_path,
    DEFAULT_CONTEXT_DB,
    list_context_databases,
    delete_context_database,
    apply_information_decay,
    get_time_range,
)
from kb.graph import build_meta_graph


def _find_prev_boundary(text: str, approx_char_idx: int, window: int = 500) -> int:
    """Find a reasonable previous boundary near approx_char_idx."""
    start = max(0, approx_char_idx - window)
    snippet = text[start:approx_char_idx]
    bl = snippet.rfind("\n\n")
    if bl != -1:
        return start + bl + 2
    for pat in (r"\n# ", r"\n## ", r"\n### "):
        m = list(re.finditer(pat, snippet))
        if m:
            return start + m[-1].start() + 1
    return approx_char_idx


def _find_next_boundary(text: str, approx_char_idx: int, window: int = 500) -> int:
    """Find a reasonable next boundary near approx_char_idx."""
    end = min(len(text), approx_char_idx + window)
    snippet = text[approx_char_idx:end]
    if approx_char_idx <= 0 or approx_char_idx >= len(text):
        return max(0, min(len(text), approx_char_idx))
    if text[approx_char_idx - 1].isspace():
        return approx_char_idx
    bl = snippet.find("\n\n")
    if bl != -1:
        return approx_char_idx + bl + 2
    for pat in (r"\n# ", r"\n## ", r"\n### "):
        m = re.search(pat, snippet)
        if m:
            return approx_char_idx + m.start() + 1
    m = re.search(r"\s", snippet)
    if m:
        pos = approx_char_idx + m.start()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos
    return approx_char_idx


def slice_text_tokens(
    text: str,
    max_tokens: int = 1500,
    overlap_tokens: int = 150,
    encoding_name: str = "o200k_base",
) -> List[str]:
    """Token-based chunking with overlap."""
    if not text:
        return []
    if max_tokens <= 0:
        return [text]
    
    enc = get_encoding(encoding_name)
    toks = enc.encode(text)
    chunks: List[str] = []
    step = max(1, max_tokens - max(0, overlap_tokens))
    
    for i in range(0, len(toks), step):
        chunk_toks = toks[i : i + max_tokens]
        if not chunk_toks:
            break
        candidate = enc.decode(chunk_toks)
        approx_start = len(enc.decode(toks[:i]))
        approx_end = len(enc.decode(toks[: i + max_tokens]))
        start_adj = 0 if i == 0 else _find_next_boundary(text, approx_start)
        end_adj = _find_prev_boundary(text, approx_end)
        
        if end_adj <= start_adj:
            start_adj = approx_start
            end_adj = max(approx_end, start_adj)
        
        start_adj = max(0, min(len(text), start_adj))
        end_adj = max(start_adj, min(len(text), end_adj))
        refined = text[start_adj:end_adj].strip()
        chunks.append(refined if refined else candidate)
    
    return chunks


def upsert_markdown_files(
    md_files: List[Path],
    *,
    max_tokens_per_chunk: Optional[int] = 1500,
    overlap_tokens: int = 150,
    encoding_name: str = "o200k_base",
    progress_cb: Optional[Callable[[int, int, Path], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    context_db: str = DEFAULT_CONTEXT_DB,
) -> int:
    """Upsert markdown files into a specific context database.
    
    Args:
        md_files: List of markdown file paths
        max_tokens_per_chunk: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        encoding_name: Tokenizer encoding name
        progress_cb: Callback for progress updates
        cancel_cb: Callback to check for cancellation
        context_db: Context database name (default: "default")
    
    Returns:
        Number of chunks upserted
    """
    engine = get_engine(context_db)
    records: List[Tuple[str, str, str]] = []
    total = len(md_files)
    
    for idx, md in enumerate(md_files):
        if cancel_cb is not None:
            try:
                if cancel_cb():
                    break
            except Exception:
                pass
        
        if progress_cb is not None:
            try:
                progress_cb(idx + 1, total, md)
            except Exception:
                pass
        
        try:
            content = md.read_text(encoding="utf-8")
            file_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            params_sig = f"t={max_tokens_per_chunk}|o={overlap_tokens}|e={encoding_name}"
            
            prev = get_file_index(engine, str(md))
            if prev and prev[0] == file_sha and prev[1] == params_sig:
                # Unchanged - skip
                continue
            else:
                # Changed or new - delete old chunks first (proper deletion)
                deleted_count = delete_chunks_by_path(engine, str(md))
                if deleted_count > 0:
                    print(f"[info] Replaced {deleted_count} old chunks for {md.name}")
            
            # Get file modification time as source date
            file_mtime = datetime.fromtimestamp(md.stat().st_mtime).isoformat()
            
            if max_tokens_per_chunk and max_tokens_per_chunk > 0:
                chunks = slice_text_tokens(
                    content,
                    max_tokens=max_tokens_per_chunk,
                    overlap_tokens=overlap_tokens,
                    encoding_name=encoding_name,
                )
            else:
                chunks = [content]
            
            for i, ch in enumerate(chunks):
                records.append((str(md), f"part{i+1}", ch))
            
            upsert_file_index(engine, str(md), file_sha, params_sig, file_date=file_mtime)
            
        except Exception as e:
            print(f"[warn] Failed to process {md}: {e}", file=sys.stderr)
            continue
    
    if records:
        upsert_chunks(engine, records)
        
        # Update Meta-Graph for new chunks
        paths = list(set(r[0] for r in records))
        all_new_ids = []
        for p in paths:
            all_new_ids.extend(fetch_chunk_ids_by_path(engine, p))
        
        if all_new_ids:
            build_meta_graph(all_new_ids)
    
    return len(records)


def delete_file_from_context(
    file_path: str,
    context_db: str = DEFAULT_CONTEXT_DB,
) -> int:
    """Completely delete a file and all its chunks from a context database.
    
    This is the fixed deletion function that properly removes:
    - All chunks for the file
    - The file index entry
    
    Args:
        file_path: Path to the file to delete
        context_db: Context database name
    
    Returns:
        Number of chunks deleted
    """
    engine = get_engine(context_db)
    
    # Delete chunks
    deleted_chunks = delete_chunks_by_path(engine, file_path)
    
    # Delete file index entry
    delete_file_index(engine, file_path)
    
    print(f"[info] Deleted {deleted_chunks} chunks and file index for {file_path}")
    
    return deleted_chunks


def list_files_in_context(
    context_db: str = DEFAULT_CONTEXT_DB,
) -> List[Tuple[str, str, str]]:
    """List all files in a context database.
    
    Returns:
        List of (path, sha256, updated_at) tuples
    """
    engine = get_engine(context_db)
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT path, sha256, updated_at FROM file_index ORDER BY updated_at DESC")
        )
        return [(r[0], r[1], r[2]) for r in rows]


# Import text at the end to avoid circular import issues
from sqlalchemy import text


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced upsert for Resume Context Builder")
    parser.add_argument("markdown_dir", help="Directory containing markdown files")
    parser.add_argument("--context", "-c", default=DEFAULT_CONTEXT_DB, 
                        help="Context database name (default: default)")
    parser.add_argument("--delete", "-d", metavar="FILE",
                        help="Delete a specific file from context")
    parser.add_argument("--list-contexts", action="store_true",
                        help="List all context databases")
    parser.add_argument("--list-files", action="store_true",
                        help="List files in the context")
    parser.add_argument("--delete-context", metavar="NAME",
                        help="Delete an entire context database")
    parser.add_argument("--decay", metavar="DAYS", type=int,
                        help="Apply information decay (remove data older than N days)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    if args.list_contexts:
        contexts = list_context_databases()
        print("Available context databases:")
        for ctx in contexts:
            marker = " *" if ctx == DEFAULT_CONTEXT_DB else ""
            print(f"  - {ctx}{marker}")
        sys.exit(0)
    
    if args.delete_context:
        if args.delete_context == DEFAULT_CONTEXT_DB:
            print(f"Error: Cannot delete default context '{DEFAULT_CONTEXT_DB}'")
            sys.exit(1)
        if delete_context_database(args.delete_context):
            print(f"Deleted context database: {args.delete_context}")
        else:
            print(f"Context not found: {args.delete_context}")
        sys.exit(0)
    
    if args.list_files:
        files = list_files_in_context(args.context)
        print(f"Files in context '{args.context}':")
        for path, sha, updated in files:
            print(f"  - {path} (updated: {updated})")
        sys.exit(0)
    
    if args.delete:
        deleted = delete_file_from_context(args.delete, args.context)
        print(f"Deleted {deleted} chunks from {args.delete}")
        sys.exit(0)
    
    if args.decay:
        engine = get_engine(args.context)
        results = apply_information_decay(engine, max_age_days=args.decay, dry_run=args.dry_run)
        print(f"Information decay ({args.decay} days):")
        print(f"  Chunks: {results['chunks']}")
        print(f"  File index: {results['file_index']}")
        print(f"  Job runs: {results['job_runs']}")
        print(f"  Feedback: {results['feedback']}")
        if args.dry_run:
            print("(Dry run - nothing was actually deleted)")
        sys.exit(0)
    
    # Default: upsert files
    root = Path(args.markdown_dir).expanduser().resolve()
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        sys.exit(1)
    
    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {root}")
        sys.exit(0)
    
    count = upsert_markdown_files(md_files, context_db=args.context)
    print(f"Upserted {count} chunk(s) from {len(md_files)} file(s) into context '{args.context}'")
