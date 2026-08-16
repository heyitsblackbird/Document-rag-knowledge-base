'''
Prometheus metrics for the RAG pipeline: retrieval, reranking, and generation stages.
'''

from prometheus_client import Counter, Histogram

# Latency buckets tuned for sub-second retrieval/rerank steps and slower LLM calls.
FAST_STAGE_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
LLM_STAGE_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34)

bm25_latency_seconds = Histogram(
    "rag_bm25_latency_seconds", "BM25 keyword search latency", buckets=FAST_STAGE_BUCKETS
)
vector_search_latency_seconds = Histogram(
    "rag_vector_search_latency_seconds", "Vector similarity search latency", buckets=FAST_STAGE_BUCKETS
)
rerank_latency_seconds = Histogram(
    "rag_rerank_latency_seconds", "Cross-encoder reranking latency", buckets=FAST_STAGE_BUCKETS
)
generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds", "LLM answer generation latency", buckets=LLM_STAGE_BUCKETS
)

vector_distance = Histogram(
    "rag_vector_distance",
    "Distance score of the top-ranked vector search result (lower = more similar)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0),
)
rerank_score = Histogram(
    "rag_rerank_score",
    "Cross-encoder relevance score of retained chunks after reranking",
    buckets=(-5, -2, -1, 0, 1, 2, 3, 5, 8),
)

chunks_retrieved_total = Counter(
    "rag_chunks_retrieved_total", "Chunks passed to the LLM as context"
)
chunks_cited_total = Counter(
    "rag_chunks_cited_total", "Chunks that were actually cited (e.g. [Source N]) in the generated answer"
)

generation_errors_total = Counter(
    "rag_generation_errors_total", "LLM generation failures", ["reason"]
)
queries_total = Counter(
    "rag_queries_total", "Total /query requests by outcome", ["outcome"]
)
