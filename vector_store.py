
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from process_document import process_pdf
from answer_generator import generate_answer


class VectorStore:
    """Store document chunks and retrieve relevant passages."""

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.reranker = CrossEncoder(reranker_name)

        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        """Create a normalized FAISS index from document chunks."""

        if not chunks:
            raise ValueError("No text chunks were provided.")

        # Keep only chunks containing text.
        self.chunks = [
            chunk
            for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not self.chunks:
            raise ValueError("No non-empty text chunks were provided.")

        texts = [
            chunk["text"].strip()
            for chunk in self.chunks
        ]

        # Generate normalized embeddings for cosine-similarity search.
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) >= 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid numeric values."
            )

        # Inner product on normalized vectors approximates cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        print(
            f"Vector index built with {len(self.chunks)} chunks."
        )

    def search(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        candidate_k: int | None = None,
        debug: bool = False,
    ) -> list[dict]:
        """
        Retrieve candidates using FAISS, rerank them using a cross-encoder,
        and combine both rankings using Reciprocal Rank Fusion (RRF).
        """

        if self.index is None:
            raise RuntimeError(
                "Build the vector store before searching."
            )

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if (
            similarity_threshold is not None
            and not -1.0 <= similarity_threshold <= 1.0
        ):
            raise ValueError(
                "similarity_threshold must be between -1 and 1."
            )

        if not self.chunks:
            return []

        # Create the question embedding.
        query_embedding = self.model.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        # Retrieve a broad candidate set for reranking.
        if candidate_k is None:
            candidate_k = max(30, top_k * 10)

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero."
            )

        candidate_k = min(candidate_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        candidates = []

        # Preserve the original FAISS ranking.
        for faiss_rank, (score, chunk_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if chunk_index < 0:
                continue

            score = float(score)

            # Apply a threshold only when explicitly requested.
            if (
                similarity_threshold is not None
                and score < similarity_threshold
            ):
                continue

            chunk = self.chunks[chunk_index]

            candidates.append({
                "chunk_id": chunk.get(
                    "chunk_id",
                    chunk_index,
                ),
                "document": chunk.get("document"),
                "text": chunk["text"],
                "page": chunk.get("page"),
                "score": score,
                "faiss_rank": faiss_rank,
            })

        if debug:
            print("\nTop FAISS candidates:")

            for item in candidates[:10]:
                print(
                    f"Page: {item['page']} | "
                    f"FAISS rank: {item['faiss_rank']} | "
                    f"Score: {item['score']:.4f} | "
                    f"Text: {item['text'][:180]}"
                )

        if not candidates:
            return []

        # Rerank candidates using the cross-encoder.
        pairs = [
            (question.strip(), item["text"])
            for item in candidates
        ]

        rerank_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for item, rerank_score in zip(
            candidates,
            rerank_scores,
        ):
            item["rerank_score"] = float(rerank_score)

        # Rank candidates by cross-encoder score.
        reranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: candidates[index]["rerank_score"],
            reverse=True,
        )

        for rerank_rank, index in enumerate(
            reranked_indices,
            start=1,
        ):
            candidates[index]["rerank_rank"] = rerank_rank

        # Combine FAISS and cross-encoder ranks using RRF.
        # RRF combines ranks rather than comparing incompatible score scales.
        rrf_constant = 60

        for item in candidates:
            item["hybrid_score"] = (
                1.0 / (
                    rrf_constant + item["faiss_rank"]
                )
                + 1.0 / (
                    rrf_constant + item["rerank_rank"]
                )
            )

        # Highest combined score appears first.
        candidates.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        results = candidates[:top_k]

        if debug:
            print("\nFinal hybrid-ranked results:")

            for item in results:
                print(
                    f"Page: {item['page']} | "
                    f"FAISS rank: {item['faiss_rank']} | "
                    f"FAISS score: {item['score']:.4f} | "
                    f"Rerank rank: {item['rerank_rank']} | "
                    f"Rerank score: {item['rerank_score']:.4f} | "
                    f"Hybrid score: {item['hybrid_score']:.6f} | "
                    f"Text: {item['text'][:180]}"
                )

        return results


if __name__ == "__main__":
    pdf_path = input(
        "Enter the path to your PDF: "
    ).strip().strip('"')

    try:
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(
                "The specified PDF does not exist."
            )

        print("\nProcessing PDF...")
        chunks = process_pdf(pdf_path)

        print("\nBuilding vector search index...")
        store = VectorStore()
        store.build(chunks)

        print("\nDocument ready for questions!")

        while True:
            question = input(
                "\nAsk a question (or type 'exit' to quit): "
            ).strip()

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            try:
                results = store.search(
                    question,
                    top_k=10,
                    candidate_k=54,
                    debug=True,
                )

                if not results:
                    print(
                        "No candidate passages were retrieved."
                    )
                    continue

                context = "\n\n".join(
                    (
                        f"Source page: {result['page']}\n"
                        f"{result['text']}"
                    )
                    for result in results
                )

                print(
                    "\nGenerating an answer with Gemini..."
                )

                answer = generate_answer(
                    question,
                    context,
                )

                print("\nAnswer:")
                print(answer)

                print("\nRetrieved sources:")

                for number, result in enumerate(
                    results,
                    start=1,
                ):
                    print(
                        f"\nSource {number} | "
                        f"Page: {result['page']} | "
                        f"FAISS score: {result['score']:.4f} | "
                        f"Rerank score: "
                        f"{result['rerank_score']:.4f} | "
                        f"Hybrid score: "
                        f"{result['hybrid_score']:.6f}"
                    )

                    print(result["text"])

            except Exception as error:
                print(
                    f"\nCould not process your question: {error}"
                )

    except Exception as error:
        print(f"\nError: {error}")
