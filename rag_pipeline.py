
from pathlib import Path
import re

from process_document import process_pdf
from vector_store import VectorStore
from answer_generator import generate_answer


class RAGPipeline:
    """Connect PDF processing, retrieval, and Gemini answers."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.document_loaded = False
        self.document_name = None

    def load_document(self, pdf_path: str) -> int:
        """Process a PDF and build its vector search index."""

        path = Path(pdf_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        chunks = process_pdf(str(path))

        if not chunks:
            raise ValueError(
                "No readable text was found in the PDF."
            )

        self.vector_store.build(chunks)

        self.document_name = path.name
        self.document_loaded = True

        return len(chunks)

    def _get_section_chunks(
        self,
        section_heading: str,
        next_section_heading: str,
    ) -> list[dict]:
        """
        Retrieve chunks from a numbered section of the loaded document.

        The section starts at its heading and ends before the next
        section heading. This avoids relying only on semantic ranking
        for questions that require a complete list.
        """

        chunks = self.vector_store.chunks

        start_index = None
        end_index = len(chunks)

        start_pattern = re.compile(
            re.escape(section_heading),
            re.IGNORECASE,
        )

        end_pattern = re.compile(
            re.escape(next_section_heading),
            re.IGNORECASE,
        )

        for index, chunk in enumerate(chunks):
            text = chunk.get("text", "")

            if start_pattern.search(text):
                start_index = index
                break

        if start_index is None:
            return []

        for index in range(start_index + 1, len(chunks)):
            text = chunks[index].get("text", "")

            if end_pattern.search(text):
                end_index = index
                break

        section_chunks = chunks[start_index:end_index]

        results = []

        for chunk in section_chunks:
            text = chunk.get("text", "").strip()

            if not text:
                continue

            results.append({
                "chunk_id": chunk.get("chunk_id"),
                "document": chunk.get("document"),
                "text": text,
                "page": chunk.get("page"),
                # These are section-retrieved passages, not scored
                # results from FAISS or the cross-encoder.
                "score": None,
                "rerank_score": None,
            })

        return results

    def _is_limitations_question(self, question: str) -> bool:
        """Identify questions about the research limitations section."""

        return bool(
            re.search(r"\blimitations?\b", question, re.IGNORECASE)
        )

    def ask(self, question: str) -> dict:
        """Answer a question and return supporting source passages."""

        if not self.document_loaded:
            raise RuntimeError(
                "Load a PDF before asking questions."
            )

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        # For limitation questions, retrieve the actual section
        # so that all its subsections can be included.
        if self._is_limitations_question(question):
            results = self._get_section_chunks(
                section_heading="VIII. LIMITATIONS OF THE RESEARCH",
                next_section_heading="IX. FUTURE SCOPE",
            )

            if results:
                print(
                    "\nSection-aware retrieval: "
                    f"found {len(results)} limitation-section chunks."
                )
            else:
                print(
                    "\nLimitations section not found. "
                    "Falling back to semantic retrieval."
                )

        else:
            results = []

        # Fall back to normal semantic search when the requested
        # section is not found, or for all other questions.
        if not results:
            results = self.vector_store.search(
                question,
                top_k=4,
                candidate_k=54,
                similarity_threshold=None,
                debug=True,
            )

        if not results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the document to answer this question."
                ),
                "sources": [],
            }

        # Include the source document and page with each passage.
        context = "\n\n".join(
            (
                f"Source document: "
                f"{result.get('document') or self.document_name}\n"
                f"Source page: "
                f"{result.get('page', 'Unknown')}\n"
                f"Passage: {result['text']}"
            )
            for result in results
        )

        answer = generate_answer(question, context)

        sources = []

        for result in results:
            similarity_score = result.get("score")
            rerank_score = result.get("rerank_score")

            sources.append({
                "document": (
                    result.get("document") or self.document_name
                ),
                "page": result.get("page"),
                "passage": result["text"],
                "similarity_score": (
                    round(float(similarity_score), 4)
                    if similarity_score is not None
                    else None
                ),
                "rerank_score": (
                    round(float(rerank_score), 4)
                    if rerank_score is not None
                    else None
                ),
            })

        return {
            "answer": answer,
            "sources": sources[:4],
        }
