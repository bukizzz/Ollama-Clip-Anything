# image_analysis.py
"""
Provides centralized functions for analyzing images with LLMs.
"""


from llm.llm_interaction import robust_llm_json_extraction
from pydantic import BaseModel, Field

class ImageAnalysisResult(BaseModel):
    scene_description: str = Field(description="A concise description of the scene in the image.")
    content_type: str = Field(description="The type of content in the scene (e.g., discussion, demo, action, static).")
    hook_potential: int = Field(description="The hook potential of the scene (e.g., 1-10).", ge=0, le=10)

def describe_image(image_path: str, prompt: str) -> ImageAnalysisResult:
    """
    Uses an LLM to generate a structured description and analysis of an image.
    """
    print(f"Querying multi-modal LLM with text: {prompt} and image data.")
    
    system_prompt = """
    You are an expert image analysis AI. Your task is to describe the provided image concisely,
    identify the content type, and assess its hook potential. Focus on people and their clothing, features.

    You MUST output ONLY a JSON object that adheres to the ImageAnalysisResult Pydantic schema.
    DO NOT include any other text, explanations, or markdown outside of the JSON block.
    DO NOT include any ```json``` markdown blocks!!!!

    Example output:
    
{
  "scene_description": "Man with a black hat wearing a red checkered shirt speaking into the microphone.",
  "content_type": "educational/entertainment/promotional/tutorial/discussion/other",
  "hook_potential": 7
}

    """
    
    # The actual prompt for the LLM will be very direct, relying on the system_prompt and schema
    # to guide the content generation.
    llm_user_prompt = "Generate the image analysis JSON for the provided image."
    
    try:
        response_model = robust_llm_json_extraction(
            system_prompt=system_prompt,
            user_prompt=llm_user_prompt,
            output_schema=ImageAnalysisResult,
            image_path=image_path,
            max_attempts=10
        )
        return response_model
    except Exception as e:
        print(f"❌ Error querying multi-modal LLM for image description: {e}")
        # Fallback to a generic response if LLM call fails
        return ImageAnalysisResult(
            scene_description="Could not generate image description due to an error.",
            content_type="unknown",
            hook_potential=0
        )

import concurrent.futures # Import concurrent.futures
import os # Import os

def describe_images_batch(image_paths_with_prompts: list[tuple[str, str]]) -> list[ImageAnalysisResult]:
    """
    Performs batch LLM inference for multiple images concurrently.
    Leverages ThreadPoolExecutor to parallelize calls to describe_image.
    """
    results = []
    # Determine max workers based on available CPU cores or a reasonable default
    max_workers = os.cpu_count() or 4 # Use number of CPU cores, or default to 4

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_image = {executor.submit(describe_image, image_path, prompt): (image_path, prompt)
                           for image_path, prompt in image_paths_with_prompts}
        
        for future in concurrent.futures.as_completed(future_to_image):
            image_path, prompt = future_to_image[future]
            try:
                analysis_result = future.result()
                results.append(analysis_result)
            except Exception as e:
                print(f"❌ Error in batch LLM analysis for image {image_path} (prompt: '{prompt[:50]}...'): {e}. Appending empty analysis.")
                results.append(ImageAnalysisResult(
                    scene_description="Could not generate image description due to an error.",
                    content_type="unknown",
                    hook_potential=0
                ))
    
    # Sort results by the original order of image_paths_with_prompts
    # This assumes image_paths_with_prompts has unique image_paths or a consistent order
    # For simplicity, we'll just return the results in the order they complete for now.
    # If strict order is needed, a more complex mapping would be required.
    return results
