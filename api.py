
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from rag_pipeline import RAGPipeline


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description=(
        "Ask questions about PDF documents using RAG and Gemini."
    ),
    version="1.0.0",
)

pipeline = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the loaded document.",
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class SearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to search for in the loaded document.",
    )

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


@app.get("/")
def home():
    """Serve the frontend homepage."""

    index_path = Path(__file__).parent / "static" / "index.html"

    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend file static/index.html was not found.",
        )

    return FileResponse(index_path)


@app.get("/health")
def health():
    """Check API status and document loading state."""

    return {
        "status": "healthy",
        "document_loaded": pipeline.document_loaded,
        "document_name": pipeline.document_name,
    }


@app.post("/load-document")
async def load_document(file: UploadFile = File(...)):
    """Receive an uploaded PDF and build its search index."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        chunk_count = pipeline.load_document(str(temp_path))
        pipeline.document_name = Path(file.filename).name

        # Store the original filename in every indexed chunk.
        for chunk in pipeline.vector_store.chunks:
            chunk["document"] = pipeline.document_name

        return {
            "message": "Document loaded successfully.",
            "document": pipeline.document_name,
            "chunks_created": chunk_count,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Error while loading document: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not process the uploaded PDF.",
        ) from error

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Generate an answer and return supporting page citations."""

    if not pipeline.document_loaded:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF document first.",
        )

    try:
        result = pipeline.ask(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Error while generating answer: {error}")

        error_message = str(error)

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. Please try again "
                    "after your quota resets."
                ),
            ) from error

        if (
            "503" in error_message
            or "UNAVAILABLE" in error_message
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gemini is temporarily busy. Please try again later."
                ),
            ) from error

        raise HTTPException(
            status_code=500,
            detail="Could not generate an answer. Please try again.",
        ) from error


@app.post("/search")
def search_document(request: SearchRequest):
    """Retrieve relevant passages with their PDF page numbers."""

    if not pipeline.document_loaded:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF document first.",
        )

    try:
        results = pipeline.vector_store.search(
            request.question,
            top_k=3,
            similarity_threshold=0.30,
        )

        return {
            "question": request.question,
            "sources": [
                {
                    "document": pipeline.document_name,
                    "page": result["page"],
                    "passage": result["text"],
                    "similarity_score": round(
                        float(result["score"]), 4
                    ),
                }
                for result in results
            ],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Error while searching document: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not search the document. Please try again.",
        ) from error
