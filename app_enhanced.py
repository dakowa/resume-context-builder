"""
Resume Context Builder - Enhanced Streamlit App

New features:
- Multiple context database selection
- Time range filtering for search
- Information decay controls
- Fixed file deletion
"""

import os
import sys
import base64
import zipfile
import uuid
import time
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
from context_packager_data.tokenizer import get_encoding
import streamlit.components.v1 as components

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from hr_tools.pdf_to_md import convert_pdfs_to_markdown
from hr_tools.package_context import package_markdown_directory

# Import enhanced modules
from kb.db_enhanced import (
    list_context_databases,
    get_engine,
    apply_information_decay,
    get_decay_log,
    get_time_range,
    DEFAULT_CONTEXT_DB,
)
from kb.upsert_enhanced import (
    upsert_markdown_files,
    delete_file_from_context,
    list_files_in_context,
)
from kb.search_enhanced import HybridSearcher


st.set_page_config(
    page_title="Resume Context Builder (Enhanced)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Resume Context Builder")
st.caption("Enhanced version with multiple contexts, time filtering, and information decay")

# Initialize session state
if "context_db" not in st.session_state:
    st.session_state.context_db = DEFAULT_CONTEXT_DB

# Get available contexts
available_contexts = list_context_databases()
if not available_contexts:
    available_contexts = [DEFAULT_CONTEXT_DB]

# Sidebar with enhanced options
with st.sidebar:
    st.header("Context Database")
    
    # Context selection
    selected_context = st.selectbox(
        "Select Context",
        options=available_contexts,
        index=available_contexts.index(st.session_state.context_db) if st.session_state.context_db in available_contexts else 0,
        help="Choose which context database to use"
    )
    st.session_state.context_db = selected_context
    
    # Create new context
    with st.expander("Create New Context"):
        new_context_name = st.text_input("New context name", placeholder="e.g., project-alpha")
        if st.button("Create Context") and new_context_name:
            if new_context_name not in available_contexts:
                get_engine(new_context_name)  # Initialize
                st.success(f"Created context: {new_context_name}")
                st.rerun()
            else:
                st.error("Context already exists")
    
    st.divider()
    
    # Information Decay Controls
    st.header("Information Decay")
    with st.expander("Auto-Cleanup Settings"):
        decay_days = st.number_input(
            "Remove data older than (days)",
            min_value=1,
            max_value=3650,
            value=365,
            help="Automatically remove data older than this many days"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview Decay", use_container_width=True):
                engine = get_engine(st.session_state.context_db)
                results = apply_information_decay(engine, max_age_days=int(decay_days), dry_run=True)
                st.write("Would delete:")
                st.write(f"- Chunks: {results['chunks']}")
                st.write(f"- File entries: {results['file_index']}")
                st.write(f"- Job runs: {results['job_runs']}")
                st.write(f"- Feedback: {results['feedback']}")
        
        with col2:
            if st.button("Apply Decay", type="primary", use_container_width=True):
                engine = get_engine(st.session_state.context_db)
                results = apply_information_decay(engine, max_age_days=int(decay_days), dry_run=False)
                st.success(f"Deleted: {results['chunks']} chunks, {results['file_index']} files")
    
    st.divider()
    
    # Settings
    st.header("Settings")
    md_dir = st.text_input("Output Markdown directory", value=str(Path.home() / "context-packager-md"))
    output_file = st.text_input("Packaged context output file", value=str(Path.home() / "output" / "context.md"))
    use_hr_instructions = st.checkbox("Include HR instructions", value=True)
    instruction_file = st.text_input("Instruction file path", value="")
    header_text = st.text_input("Header text", value="HR Candidate Resume Pack")
    max_tokens = st.number_input("Max tokens per file (0 = no split)", value=120000, min_value=0, step=1000)
    encoding_name = st.selectbox("Tokenizer/encoding", options=["o200k_base", "cl100k_base"], index=0)

# Main content area
tab_ingest, tab_search, tab_manage = st.tabs(["📤 Ingest", "🔍 Search", "⚙️ Manage"])

# Tab 1: Ingest
with tab_ingest:
    st.subheader(f"Ingest into Context: `{st.session_state.context_db}`")
    
    uploaded_files = st.file_uploader(
        "Upload ZIP or supported files",
        type=None,
        accept_multiple_files=True,
        key="file_uploader"
    )
    fallback_dir_main = st.text_input("Or enter a directory path (optional)", value="")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start = st.button("Process and Package", type="primary", use_container_width=True)
        clear = st.button("Reset", use_container_width=True)
    
    if clear:
        st.session_state.pop("context_content", None)
        st.session_state.pop("out_paths", None)
        st.session_state.pop("selected_chunk", None)
        try:
            st.session_state["file_uploader"] = []
        except Exception:
            pass
    
    if start:
        effective_pdf_dir = None
        
        if uploaded_files:
            uploads_root = Path.home() / ".context-packager-uploads"
            uploads_root.mkdir(parents=True, exist_ok=True)
            temp_root = uploads_root / f"session-{uuid.uuid4().hex[:8]}"
            temp_root.mkdir(parents=True, exist_ok=True)
            combined_dir = temp_root / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            
            for f in uploaded_files:
                name = Path(f.name).name
                if name.lower().endswith(".zip"):
                    zip_path = temp_root / name
                    zip_path.write_bytes(f.read())
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(combined_dir)
                else:
                    (combined_dir / name).write_bytes(f.read())
            effective_pdf_dir = str(combined_dir.resolve())
        elif fallback_dir_main and os.path.isdir(fallback_dir_main):
            effective_pdf_dir = fallback_dir_main
        else:
            st.error("Please upload a ZIP/PDFs or provide a valid directory path.")
            st.stop()
        
        Path(md_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)
        
        with st.spinner("Converting PDFs to Markdown..."):
            generated_md = convert_pdfs_to_markdown(effective_pdf_dir, md_dir)
            st.success(f"Converted {len(generated_md)} markdown files.")
        
        # Upsert into selected context
        with st.spinner(f"Upserting into context '{st.session_state.context_db}'..."):
            count = upsert_markdown_files(
                [Path(p) for p in generated_md],
                context_db=st.session_state.context_db,
                max_tokens_per_chunk=1500,
                overlap_tokens=150,
                encoding_name=encoding_name
            )
            st.success(f"Upserted {count} chunks into context '{st.session_state.context_db}'")
        
        with st.spinner("Packaging context with Repomix..."):
            instr_path = instruction_file if (use_hr_instructions and os.path.isfile(instruction_file)) else None
            run_tag = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            base = Path(output_file)
            unique_output = base.with_name(f"{base.stem}_{run_tag}{base.suffix}")
            out_paths = package_markdown_directory(
                md_dir,
                str(unique_output),
                instruction_file=instr_path,
                header_text=header_text,
                max_tokens=max_tokens or None,
                encoding_name=encoding_name,
                predefined_md_files=[str(p) for p in generated_md],
            )
            st.success(f"Packaged {len(out_paths)} file(s)")
            
            try:
                content = Path(out_paths[0]).read_text(encoding="utf-8")
                st.session_state["context_content"] = content
            except Exception as e:
                st.error(f"Failed to read output file: {e}")
            st.session_state["out_paths"] = [str(p) for p in out_paths]
            
            # Show token counts
            enc = get_encoding(encoding_name)
            rows = []
            for p in out_paths:
                text = Path(p).read_text(encoding="utf-8")
                tok = len(enc.encode(text))
                rows.append((str(p), tok))
            st.write("Generated files (with token counts):")
            for p, tok in rows:
                st.write(f"{p} — tokens: {tok}")
        
        try:
            st.session_state["file_uploader"] = []
        except Exception:
            pass
    
    if "out_paths" in st.session_state and st.session_state["out_paths"]:
        st.subheader("Packaged Context")
        options = [f"Part {i+1}: {Path(p).name}" for i, p in enumerate(st.session_state["out_paths"])]
        idx = st.session_state.get("selected_chunk", 0)
        idx = st.selectbox("Select chunk", options=list(range(len(options))), format_func=lambda i: options[i], index=min(idx, len(options)-1))
        st.session_state["selected_chunk"] = idx
        selected_path = st.session_state["out_paths"][idx]
        selected_content = Path(selected_path).read_text(encoding="utf-8")
        
        col_dl, col_cp = st.columns(2)
        with col_dl:
            st.download_button(
                label=f"Download {Path(selected_path).name}",
                data=selected_content,
                file_name=Path(selected_path).name,
                mime="text/markdown",
                use_container_width=True,
            )
        with col_cp:
            b64 = base64.b64encode(selected_content.encode("utf-8")).decode("ascii")
            html = f"""
            <style>
            .copy-wrap {{ position: relative; width: 100%; }}
            .copy-btn {{
              all: unset; display: inline-block; width: 100%; text-align: center;
              padding: 0.6rem 1rem; border-radius: 0.25rem; background-color: #F63366;
              color: #fff; cursor: pointer; font-weight: 600;
            }}
            .copy-btn:hover {{ filter: brightness(0.95); }}
            .copy-btn.copied {{ background-color: #22c55e; }}
            </style>
            <script>
            async function copyChunk(){{
              const text = atob('{b64}');
              try {{ await navigator.clipboard.writeText(text); }} catch(e) {{
                const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
                ta.focus(); ta.select(); try {{ document.execCommand('copy'); }} catch(e2) {{}} document.body.removeChild(ta);
              }}
              const btn = document.getElementById('copy-btn');
              if (btn) {{ btn.classList.add('copied'); btn.textContent = 'Copied!'; setTimeout(()=>{{ btn.classList.remove('copied'); btn.textContent = 'Copy to clipboard'; }}, 1200); }}
            }}
            </script>
            <div class="copy-wrap">
              <button id="copy-btn" class="copy-btn" onclick="copyChunk()">Copy to clipboard</button>
            </div>
            """
            components.html(html, height=60)
        
        st.text_area("Preview", selected_content, height=400)

# Tab 2: Search
with tab_search:
    st.subheader(f"Search Context: `{st.session_state.context_db}`")
    
    # Time range filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search query", placeholder="Enter search terms...")
    with col2:
        time_range = st.selectbox(
            "Time range",
            options=["all", "1d", "1w", "1m", "3m", "6m", "1y"],
            format_func=lambda x: {
                "all": "All time",
                "1d": "Last 1 day",
                "1w": "Last 1 week",
                "1m": "Last 1 month",
                "3m": "Last 3 months",
                "6m": "Last 6 months",
                "1y": "Last 1 year"
            }.get(x, x)
        )
    
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)
    
    if st.button("Search", type="primary"):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                try:
                    searcher = HybridSearcher(st.session_state.context_db)
                    searcher.fit_from_db(time_range=time_range)
                    results = searcher.search(search_query, top_k=top_k, time_range=time_range)
                    
                    if not results:
                        st.info("No results found")
                    else:
                        st.success(f"Found {len(results)} results")
                        for i, (cid, score, path, cname, snippet, full_text) in enumerate(results, 1):
                            with st.expander(f"{i}. Score: {score:.4f} | {Path(path).name}"):
                                st.write(f"**Chunk:** {cname}")
                                st.write(f"**Path:** {path}")
                                st.write(f"**Content:**")
                                st.text(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))
                                
                                # Copy button for this result
                                b64 = base64.b64encode(full_text.encode("utf-8")).decode("ascii")
                                html = f"""
                                <script>
                                async function copyResult{i}(){{
                                  const text = atob('{b64}');
                                  try {{ await navigator.clipboard.writeText(text); }} catch(e) {{
                                    const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
                                    ta.focus(); ta.select(); try {{ document.execCommand('copy'); }} catch(e2) {{}} document.body.removeChild(ta);
                                  }}
                                  alert('Copied to clipboard!');
                                }}
                                </script>
                                <button onclick="copyResult{i}()" style="padding: 0.5rem 1rem; background: #F63366; color: white; border: none; border-radius: 0.25rem; cursor: pointer;">Copy full text</button>
                                """
                                components.html(html, height=50)
                except Exception as e:
                    st.error(f"Search failed: {e}")

# Tab 3: Manage
with tab_manage:
    st.subheader(f"Manage Context: `{st.session_state.context_db}`")
    
    # List files in context
    st.write("### Files in Context")
    files = list_files_in_context(st.session_state.context_db)
    
    if not files:
        st.info("No files in this context")
    else:
        st.write(f"Total files: {len(files)}")
        
        for path, sha, updated in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- `{path}`")
                st.caption(f"Updated: {updated}")
            with col2:
                if st.button("Delete", key=f"del_{sha}"):
                    deleted = delete_file_from_context(path, st.session_state.context_db)
                    st.success(f"Deleted {deleted} chunks")
                    st.rerun()
    
    st.divider()
    
    # Decay log
    st.write("### Decay History")
    engine = get_engine(st.session_state.context_db)
    logs = get_decay_log(engine, limit=5)
    
    if not logs:
        st.info("No decay operations recorded")
    else:
        for action, details, executed_at in logs:
            st.write(f"**{executed_at}**")
            st.write(f"Action: {action}")
            st.write(f"Details: {details}")
            st.divider()
