"""
Enhanced Search Module for Resume Context Builder

Features:
- Time range filtering (1 day to 12 months)
- Multiple context database support
- Integration with information decay
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

# Import from enhanced database module
from kb.db_enhanced import (
    get_engine,
    fetch_all_chunks,
    fetch_chunk_vectors,
    fetch_feedback_for_query,
    persist_chunk_vectors,
    get_time_range,
    DEFAULT_CONTEXT_DB,
)

try:
    from pynndescent import NNDescent
except Exception:
    NNDescent = None


class HybridSearcher:
    """Enhanced hybrid searcher with time range support."""
    
    def __init__(self, context_db: str = DEFAULT_CONTEXT_DB):
        self.context_db = context_db
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words="english",
            sublinear_tf=True,
            max_df=0.95,
            norm="l2",
        )
        self.matrix = None
        self.count_vectorizer = CountVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            stop_words="english",
            max_df=0.95,
        )
        self.count_matrix = None
        self.doc_lengths: np.ndarray | None = None
        self.avg_doc_length: float = 0.0
        self.bm25_idf: np.ndarray | None = None
        self.bm25_k1: float = 1.2
        self.bm25_b: float = 0.75
        self.ids: List[int] = []
        self.texts: List[str] = []
        self.meta: List[Tuple[str, str]] = []
        self.svd: TruncatedSVD | None = None
        self.doc_vectors_reduced: np.ndarray | None = None
        self.ann_index = None
        self.path_to_ordered_indices: Dict[str, List[int]] = {}
        self.max_pgvector_cache: int = 10000
        self.enable_ann: bool = True
        self.svd_components_cap: int = 256

    def fit(
        self, 
        docs: List[Tuple[int, str, str, str]],
        time_range: str | None = None
    ):
        """Fit the searcher on documents, optionally filtered by time range.
        
        Args:
            docs: List of (id, path, chunk_name, content) tuples
            time_range: Optional time range filter (e.g., "1d", "1w", "3m", "1y")
        """
        # Apply time range filter if specified
        if time_range:
            since, until = get_time_range(time_range)
            if since or until:
                # Filter docs by time
                engine = get_engine(self.context_db)
                # Get all chunks with their timestamps
                all_chunks = fetch_all_chunks(engine, since=since, until=until)
                allowed_ids = {c[0] for c in all_chunks}
                docs = [d for d in docs if d[0] in allowed_ids]
        
        self.ids = [d[0] for d in docs]
        self.meta = [(d[1], d[2]) for d in docs]
        self.texts = [d[3] for d in docs]
        
        if self.texts:
            self.matrix = self.vectorizer.fit_transform(self.texts)
            
            # Build BM25 structures
            try:
                self.count_matrix = self.count_vectorizer.fit_transform(self.texts)
                n_docs = self.count_matrix.shape[0]
                df = (self.count_matrix > 0).astype(np.int32).sum(axis=0)
                df = np.asarray(df).ravel()
                self.bm25_idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                self.doc_lengths = np.asarray(self.count_matrix.sum(axis=1)).ravel()
                self.avg_doc_length = float(self.doc_lengths.mean() if self.doc_lengths.size else 0.0)
            except Exception:
                self.count_matrix = None
                self.doc_lengths = None
                self.avg_doc_length = 0.0
                self.bm25_idf = None
            
            # Build path -> ordered matrix index list
            order_map: Dict[str, List[Tuple[int, int]]] = {}
            for m_idx, (path, cname) in enumerate(self.meta):
                m = re.search(r"(\d+)$", cname)
                chunk_num = int(m.group(1)) if m else (m_idx + 1)
                order_map.setdefault(path, []).append((chunk_num, m_idx))
            self.path_to_ordered_indices = {
                p: [mi for _, mi in sorted(pairs, key=lambda x: x[0])] 
                for p, pairs in order_map.items()
            }
            
            # Try to hydrate SVD vectors from pgvector
            engine = get_engine(self.context_db)
            pgvec: Dict[int, List[float]] = {}
            try:
                if engine.dialect.name == "postgresql" and len(self.ids) <= self.max_pgvector_cache:
                    pg = fetch_chunk_vectors(engine, self.ids)
                    if pg:
                        pgvec = {cid: vec for cid, vec in pg.items() if vec}
            except Exception:
                pgvec = {}
            
            # Build ANN index
            if self.enable_ann and NNDescent is not None and self.matrix.shape[0] >= 10:
                n_comp = min(
                    self.svd_components_cap,
                    max(16, min(self.matrix.shape[0] - 1, self.matrix.shape[1] - 1))
                )
                try:
                    self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    if len(pgvec) == len(self.ids):
                        try:
                            doc_vecs = np.array([pgvec.get(cid, []) for cid in self.ids], dtype=np.float32)
                            if doc_vecs.ndim != 2 or doc_vecs.shape[1] != n_comp:
                                raise ValueError("pgvector dim mismatch")
                        except Exception:
                            doc_vecs = self.svd.fit_transform(self.matrix).astype(np.float32, copy=False)
                    else:
                        doc_vecs = self.svd.fit_transform(self.matrix).astype(np.float32, copy=False)
                    
                    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    doc_vecs = doc_vecs / norms
                    self.doc_vectors_reduced = doc_vecs
                    
                    try:
                        n_nbrs = int(min(64, max(10, doc_vecs.shape[0] - 1)))
                        self.ann_index = NNDescent(doc_vecs, metric="cosine", n_neighbors=n_nbrs, random_state=42)
                    except Exception:
                        self.ann_index = None
                except Exception:
                    self.svd = None
                    self.doc_vectors_reduced = None
                    self.ann_index = None
            
            # Persist vectors to pgvector
            try:
                if engine.dialect.name == "postgresql" and self.doc_vectors_reduced is not None:
                    rows = []
                    for i, cid in enumerate(self.ids):
                        vec = self.doc_vectors_reduced[i].astype(np.float32).tolist()
                        rows.append((int(cid), vec))
                    persist_chunk_vectors(engine, rows)
            except Exception:
                pass

    def search(
        self,
        query: str,
        top_k: int = 5,
        time_range: str | None = None,
        **kwargs
    ) -> List[Tuple[int, float, str, str, str, str]]:
        """Search with optional time range filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            time_range: Optional time range ("1d", "1w", "1m", "3m", "6m", "1y", "all")
            **kwargs: Additional search parameters
        
        Returns:
            List of (id, score, path, chunk_name, snippet, full_text) tuples
        """
        # If time range specified, we need to filter the corpus
        if time_range and time_range != "all":
            engine = get_engine(self.context_db)
            since, until = get_time_range(time_range)
            filtered_docs = fetch_all_chunks(engine, since=since, until=until)
            allowed_ids = {d[0] for d in filtered_docs}
            
            # Filter internal data structures
            filtered_indices = [i for i, cid in enumerate(self.ids) if cid in allowed_ids]
            if not filtered_indices:
                return []
            
            # Create temporary filtered searcher
            filtered_ids = [self.ids[i] for i in filtered_indices]
            filtered_texts = [self.texts[i] for i in filtered_indices]
            filtered_meta = [self.meta[i] for i in filtered_indices]
            
            # Run search on filtered subset
            return self._search_subset(
                query, top_k, filtered_ids, filtered_texts, filtered_meta, **kwargs
            )
        
        # No time filtering - search full corpus
        return self._search_subset(
            query, top_k, self.ids, self.texts, self.meta, **kwargs
        )

    def _search_subset(
        self,
        query: str,
        top_k: int,
        ids: List[int],
        texts: List[str],
        meta: List[Tuple[str, str]],
        **kwargs
    ) -> List[Tuple[int, float, str, str, str, str]]:
        """Internal search implementation on a subset of documents."""
        if not texts:
            return []
        
        query = (query or "").strip()
        if not query:
            return []
        
        # Transform query
        q_vec = self.vectorizer.transform([query])
        k = min(max(1, top_k), len(ids))
        if k <= 0:
            return []
        
        # Build temporary matrix for subset
        subset_matrix = self.matrix[[self.ids.index(cid) for cid in ids]]
        
        # Compute similarities
        similarities = linear_kernel(q_vec, subset_matrix).ravel()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            cid = ids[idx]
            score = float(similarities[idx])
            path, cname = meta[idx]
            text = texts[idx]
            
            # Build snippet
            snippet = text[:400] + "..." if len(text) > 400 else text
            
            results.append((cid, score, path, cname, snippet, text))
        
        return results

    def fit_from_db(
        self,
        time_range: str | None = None,
        limit: int | None = None
    ):
        """Fit searcher directly from database with optional time filtering.
        
        Args:
            time_range: Optional time range filter
            limit: Maximum number of chunks to load
        """
        engine = get_engine(self.context_db)
        
        since, until = (None, None)
        if time_range and time_range != "all":
            since, until = get_time_range(time_range)
        
        docs = fetch_all_chunks(engine, since=since, until=until)
        
        if limit and len(docs) > limit:
            docs = docs[:limit]
        
        self.fit(docs)


def search_with_time_filter(
    query: str,
    time_range: str = "all",
    top_k: int = 5,
    context_db: str = DEFAULT_CONTEXT_DB
) -> List[Tuple[int, float, str, str, str, str]]:
    """Convenience function for searching with time filter.
    
    Args:
        query: Search query
        time_range: Time range ("1d", "1w", "1m", "3m", "6m", "1y", "all")
        top_k: Number of results
        context_db: Context database name
    
    Returns:
        Search results
    """
    searcher = HybridSearcher(context_db)
    searcher.fit_from_db(time_range=time_range)
    return searcher.search(query, top_k=top_k, time_range=time_range)


if __name__ == "__main__":
    # CLI interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced search for Resume Context Builder")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--context", "-c", default=DEFAULT_CONTEXT_DB,
                        help="Context database name")
    parser.add_argument("--time-range", "-t", default="all",
                        choices=["1d", "1w", "1m", "3m", "6m", "1y", "all"],
                        help="Time range filter")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of results")
    
    args = parser.parse_args()
    
    results = search_with_time_filter(
        args.query,
        time_range=args.time_range,
        top_k=args.top_k,
        context_db=args.context
    )
    
    print(f"Search results for '{args.query}' (time range: {args.time_range}):")
    print("-" * 60)
    for cid, score, path, cname, snippet, _ in results:
        print(f"\nScore: {score:.4f} | Chunk: {cname}")
        print(f"Path: {path}")
        print(f"Snippet: {snippet[:200]}...")
        print("-" * 60)
