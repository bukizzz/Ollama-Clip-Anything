from agents.base_agent import Agent
from typing import Dict, Any
import os
from PIL import Image
from llm.image_analysis import describe_image

class BrollAnalysisAgent(Agent):
    """Agent responsible for analyzing B-roll assets and generating descriptions."""

    def __init__(self):
        super().__init__("BrollAnalysisAgent")
        self.b_roll_assets_dir = "b_roll_assets" # Defined in CFC.md

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("BrollAnalysisAgent: Scanning and analyzing B-roll assets...")

        b_roll_data = []
        if not os.path.exists(self.b_roll_assets_dir):
            print(f"B-roll assets directory '{self.b_roll_assets_dir}' not found. Skipping B-roll analysis.")
            context["b_roll_data"] = b_roll_data
            return context

        for filename in os.listdir(self.b_roll_assets_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.b_roll_assets_dir, filename)
                try:
                    with Image.open(image_path) as img:
                        # Get description using the centralized function
                        description_prompt = "Describe this image in detail, focusing on key objects, subjects, and the overall scene."
                        image_description = describe_image(img, description_prompt)
                        
                        b_roll_data.append({
                            "path": image_path,
                            "description": image_description
                        })
                        print(f"Analyzed B-roll asset: {filename}")
                except Exception as e:
                    print(f"Error processing B-roll image {filename}: {e}")
        
        context["b_roll_data"] = b_roll_data
        print(f"✅ B-roll analysis complete. Found {len(b_roll_data)} assets.")
        return context
