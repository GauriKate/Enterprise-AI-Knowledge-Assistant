
from rag_pipeline import RAGPipeline


def main():
    pdf_path = input(
        "Enter the path to your PDF: "
    ).strip().strip('"')

    pipeline = RAGPipeline()

    try:
        print("\nLoading document...")
        chunk_count = pipeline.load_document(pdf_path)

        print(f"Document loaded successfully: {chunk_count} chunks.")

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
                answer = pipeline.ask(question)
                print("\nAnswer:")
                print(answer)

            except Exception as error:
                print(f"Could not answer the question: {error}")

    except Exception as error:
        print(f"Could not load the document: {error}")


if __name__ == "__main__":
    main()
