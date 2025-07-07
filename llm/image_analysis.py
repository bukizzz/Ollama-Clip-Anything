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

def describe_images_batch(image_paths_with_prompts: list[tuple[str, str]]) -> list[ImageAnalysisResult]:
    """
    Performs batch LLM inference for multiple images.
    NOTE: This is a simulated batch for now. For true batching, the underlying LLM API
    (robust_llm_json_extraction) would need to support multiple image inputs.
    """
    results = []
    for image_path, prompt in image_paths_with_prompts:
        print(f"Querying multi-modal LLM for batch item with text: {prompt} and image data.")
        try:
            analysis_result = describe_image(image_path, prompt)
            results.append(analysis_result)
        except Exception as e:
            print(f"❌ Error in batch LLM analysis for image {image_path}: {e}. Appending empty analysis.")
            results.append(ImageAnalysisResult(
                scene_description="Could not generate image description due to an error.",
                content_type="unknown",
                hook_potential=0
            ))
    return results
