import sys
import os
import base64
from io import BytesIO
from PIL import Image
from unstructured.partition.auto import partition
import ollama

def extract_and_ask_images(filepath, question):
    print(f"Partitioning: {filepath}")
    
    elements = partition(
        filename=filepath,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )

    # Collect all base64 images
    images_b64 = []
    for el in elements:
        b64 = getattr(el.metadata, "image_base64", None)
        if b64:
            images_b64.append(b64)

    print(f"Found {len(images_b64)} image(s) in PDF.")

    if not images_b64:
        print("No images found. Check that poppler and tesseract are installed.")
        return

    # Save images locally so you can verify what's being sent
    for i, b64 in enumerate(images_b64):
        img = Image.open(BytesIO(base64.b64decode(b64)))
        img.save(f"extracted_image_{i+1}.png")
        print(f"Saved extracted_image_{i+1}.png")

    # Send all images + question to the vision LLM
    print(f"\nAsking vision model: '{question}'")

    response = ollama.chat(
        model="my-qwen3vl:latest",
        messages=[
            {
                "role": "user",
                "content": question,
                "images": images_b64  # ollama accepts raw base64 strings here
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python class.py <pdf_path> "<question>"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    answer = extract_and_ask_images(pdf_path, question)
    print("\n--- Response ---")
    print(answer)