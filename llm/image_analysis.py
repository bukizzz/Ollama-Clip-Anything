# image_analysis.py
"""
Provides centralized functions for analyzing images with LLMs.
"""

from PIL import Image
import ollama
from core.llm_models import image_to_base64 # Import the utility function
from core.config import IMAGE_RECOGNITION_MODEL # Import the new image recognition model from config

def describe_image(image: Image.Image, prompt: str) -> str:
    """
    Uses an LLM to generate a description of an image.
    In a real implementation, this would involve loading the model
    and performing inference.
    """
    print(f"Querying multi-modal LLM with text: {prompt} and image data.")
    
    # Convert PIL Image to base64 string
    base64_image = image_to_base64(image)
    
    # Define the messages for the Ollama chat API
    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': [base64_image] # Pass the base64 image here
        }
    ]
    
    try:
        # Use the IMAGE_RECOGNITION_MODEL from config
        response = ollama.chat(model=IMAGE_RECOGNITION_MODEL, messages=messages) # Removed stop parameter
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            raise ValueError(f"Unexpected response format from LLM: {response}")
    except Exception as e:
        print(f"❌ Error querying multi-modal LLM for image description: {e}")
        # Fallback to a generic response if LLM call fails
        return "Could not generate image description due to an error."
