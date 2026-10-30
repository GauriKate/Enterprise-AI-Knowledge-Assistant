
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY is missing from .env")
        return

    client = genai.Client(api_key=api_key)

    # Step 1: List available Gemini models
    print("Checking available Gemini models...\n")

    try:
        for model in client.models.list():
            print(model.name)

    except Exception as error:
        print("Could not list models:")
        print(error)

    # Step 2: Test Gemini answer generation
    print("\nTesting Gemini answer generation...")

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Reply with: Gemini is working!"
        )

        if response.text:
            print("Gemini response:", response.text)
        else:
            print("Gemini returned an empty response.")

    except Exception as error:
        print("Gemini generation failed:")
        print(error)


if __name__ == "__main__":
    main()
