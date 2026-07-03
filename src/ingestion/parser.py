# src/ingestion/parser.py
import os
import base64
from groq import Groq
from llama_parse import LlamaParse
from src.config import Config

class MultimodalParser:
    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.parser = LlamaParse(
            api_key=Config.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            num_workers=4,
            verbose=True
        )

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image_with_groq(self, image_path: str) -> str:
        """Uses Groq's high-speed vision model to create a summary of layout artifacts."""
        try:
            base64_image = self._encode_image_to_base64(image_path)
            response = self.groq_client.chat.completions.create(
                model=Config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this technical image/diagram from a document. Provide a comprehensive summary explaining its architectural components, dataflows, metrics, or tables depicted. Output only the analysis, without introductory chat elements."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Failed to analyze image {image_path} with Groq Vision: {str(e)}")
            return ""

    def process_document(self, file_name: str) -> str:
        """Parses document layout and stitches text content with image visual summaries."""
        file_path = os.path.join(Config.DATA_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")

        print(f"🚀 Kicking off Layout-Aware Parsing for {file_name}...")
        # LlamaParse handles extraction and dumps images found inline to a specified directory
        documents = self.parser.load_data(file_path, extra_info={"target_dir": Config.IMAGE_OUTPUT_DIR})
        
        full_markdown_text = "\n\n".join([doc.text for doc in documents])
        
        # Look for extracted images to process and weave back into metadata context
        image_summaries = []
        for img_file in os.listdir(Config.IMAGE_OUTPUT_DIR):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(Config.IMAGE_OUTPUT_DIR, img_file)
                print(f"📸 Found extracted artifact: {img_file}. Summarizing via Vision LLM...")
                summary = self.analyze_image_with_groq(img_path)
                image_summaries.append(f"\n\n### [Visual Element Summary: {img_file}]\n{summary}")
                # Clean up image after parsing to conserve local disk space
                os.remove(img_path)

        # Merge extracted text and visual synthesis blocks
        enriched_corpus = full_markdown_text + "\n\n" + "\n\n".join(image_summaries)
        return enriched_corpus