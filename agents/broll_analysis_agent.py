from agents.base_agent import Agent
from typing import Dict, Any, List, Optional
import os
import json
import hashlib
from datetime import datetime
from PIL import Image
from llm import llm_interaction
from core.state_manager import set_stage_status
from llm.image_analysis import describe_image, ImageAnalysisResult # Import ImageAnalysisResult
from llm.llm_interaction import SuggestionsResponse # Import the new model
from core.cache_manager import cache_manager # Import cache_manager
from core.resource_manager import resource_manager # Import resource_manager

class BrollAnalysisAgent(Agent):
    def __init__(self, config, state_manager):
        super().__init__("BrollAnalysisAgent")
        self.config = config
        self.state_manager = state_manager
        self.b_roll_assets_dir = config.get('b_roll_assets_dir')
        self.asset_database_path = os.path.join(self.b_roll_assets_dir, 'asset_database.json')
        # Use standard LLM model for text-based suggestions
        self.llm_model = config.get('llm_model', 'llama3.1:8b')
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_name = self.name
        print(f"\nExecuting stage: {stage_name}")

        # --- Pre-flight Check ---
        # If B-roll suggestions already exist in the context, skip this stage.
        if context.get('b_roll_suggestions') and context.get('pipeline_stages', {}).get(stage_name) == 'complete':
            print(f"✅ Skipping {stage_name}: B-roll analysis already complete.")
            return context

        # Retrieve clips from the correct location in the hierarchical context
        clips = context.get('current_analysis', {}).get('clips', [])
        audio_rhythm = context.get('audio_rhythm_data', {}) # This might also need to be checked for its correct location

        if not clips:
            print("🎬 No clips selected, skipping B-roll analysis.")
            context['b_roll_suggestions'] = []
            set_stage_status('broll_analysis', 'skipped', {'reason': 'No clips'})
            return context

        print("🎬 Starting B-roll analysis...")
        set_stage_status('broll_analysis', 'running')

        processed_video_path = context.get('processed_video_path')
        if processed_video_path is None:
            self.log_error("Processed video path is missing from context. Cannot generate B-roll cache key.")
            set_stage_status('broll_analysis', 'failed', {'reason': 'Missing processed_video_path'})
            return context

        cache_key = f"broll_analysis_{os.path.basename(processed_video_path)}"
        cached_results = cache_manager.get(cache_key, level="disk")

        if cached_results:
            print("⏩ Skipping B-roll analysis. Loaded from cache.")
            context.update(cached_results)
            set_stage_status('broll_analysis', 'complete', {'loaded_from_cache': True})
            return context

        # Discover and catalog assets
        b_roll_assets = self._discover_and_catalog_assets()
        
        if not b_roll_assets:
            self.log_warning("⚠️ No B-roll assets found")
            context['b_roll_suggestions'] = []
            set_stage_status('broll_analysis', 'complete', {'num_suggestions': 0})
            return context

        # Generate suggestions for clips
        suggestions = []
        for clip in clips:
            # Ensure clip is a dictionary before processing
            if not isinstance(clip, dict):
                self.log_warning(f"Skipping non-dictionary clip in B-roll analysis: {clip}")
                continue
            suggestions.extend(self._generate_suggestions_for_clip(clip, b_roll_assets, audio_rhythm))
        
        context['b_roll_suggestions'] = suggestions
        print(f"✅ B-roll analysis complete - {len(suggestions)} suggestions generated")
        set_stage_status('broll_analysis', 'complete', {'num_suggestions': len(suggestions)})
        
        # Cache the results before returning
        cache_manager.set(cache_key, {
            'b_roll_suggestions': suggestions
        }, level="disk")

        return context

    def _discover_and_catalog_assets(self) -> List[Dict[str, Any]]:
        """Discover and catalog all B-roll assets with comprehensive metadata."""
        print("🔍 Discovering B-roll assets...")
        
        if not os.path.exists(self.b_roll_assets_dir):
            print(f"❌ B-roll directory not found: {self.b_roll_assets_dir}")
            return []

        # Load existing database
        asset_database = self._load_asset_database()
        
        # Scan directory for image files
        current_files = self._scan_image_files()
        print(f"📁 Found {len(current_files)} image files in directory")
        
        # Find new files that need processing
        processed_files = set(asset_database.keys())
        new_files = [f for f in current_files if f not in processed_files]
        
        if new_files:
            print(f"🆕 Processing {len(new_files)} new files...")
            for file_path in new_files:
                print(f"⚙️ Processing: {os.path.basename(file_path)}")
                asset_metadata = self._process_single_asset(file_path)
                if asset_metadata:
                    asset_database[file_path] = asset_metadata
        else:
            print("✅ All files already processed")
        
        # Remove entries for files that no longer exist
        removed_files = [f for f in processed_files if f not in current_files]
        if removed_files:
            print(f"🗑️ Removing {len(removed_files)} deleted files from database")
            for file_path in removed_files:
                del asset_database[file_path]
        
        # Save updated database
        self._save_asset_database(asset_database)
        
        # Convert to list format for suggestions
        assets_list = []
        for file_path, metadata in asset_database.items():
            assets_list.append({
                'path': file_path,
                'filename': os.path.basename(file_path),
                'description': metadata.get('description', 'No description'),
                'resolution': metadata.get('resolution', 'Unknown'),
                'file_size': metadata.get('file_size', 0),
                'format': metadata.get('format', 'Unknown'),
                'colors': metadata.get('colors', []),
                'tags': metadata.get('tags', [])
            })
        
        print(f"📊 Asset discovery complete - {len(assets_list)} assets available")
        return assets_list

    def _load_asset_database(self) -> Dict[str, Dict[str, Any]]:
        """Load existing asset database from JSON file."""
        if os.path.exists(self.asset_database_path):
            try:
                with open(self.asset_database_path, 'r', encoding='utf-8') as f:
                    database = json.load(f)
                print(f"📋 Loaded existing database with {len(database)} entries")
                return database
            except Exception as e:
                print(f"⚠️ Error loading database, starting fresh: {e}")
                return {}
        else:
            print("📋 No existing database found, creating new one")
            return {}

    def _save_asset_database(self, database: Dict[str, Dict[str, Any]]) -> None:
        """Save asset database to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.asset_database_path), exist_ok=True)
            with open(self.asset_database_path, 'w', encoding='utf-8') as f:
                json.dump(database, f, indent=2, ensure_ascii=False)
            print(f"💾 Database saved with {len(database)} entries")
        except Exception as e:
            print(f"❌ Error saving database: {e}")

    def _scan_image_files(self) -> List[str]:
        """Scan directory for supported image files."""
        image_files = []
        
        for root, dirs, files in os.walk(self.b_roll_assets_dir):
            # Skip the database file itself
            if 'asset_database.json' in files:
                files.remove('asset_database.json')
                
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in self.supported_formats):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        return sorted(image_files)

    def _process_single_asset(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single asset and extract comprehensive metadata."""
        try:
            # Basic file info
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            modified_time = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            
            # Image technical info
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                # Extract dominant colors (simplified approach)
                colors = self._extract_dominant_colors(img)
            
            # Generate file hash for integrity checking
            file_hash = self._generate_file_hash(file_path)
            
            # Generate description using LLM
            description = self._generate_asset_description(file_path)
            
            # Extract simple tags from description
            tags = self._extract_tags_from_description(description)
            
            metadata = {
                'file_hash': file_hash,
                'file_size': file_size,
                'modified_time': modified_time,
                'resolution': f"{width}x{height}",
                'width': width,
                'height': height,
                'format': format_name,
                'color_mode': mode,
                'colors': colors,
                'description': description,
                'tags': tags,
                'processed_time': datetime.now().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
            return None

    def _extract_dominant_colors(self, img: Image.Image, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image using simple quantization."""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image for faster processing
            img = img.resize((150, 150))
            
            # Use quantize to find dominant colors
            quantized = img.quantize(colors=num_colors)
            palette = quantized.getpalette()
            
            # Convert palette to hex colors
            colors = []
            for i in range(num_colors):
                r, g, b = palette[i*3:(i+1)*3]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                colors.append(hex_color)
            
            return colors
            
        except Exception as e:
            print(f"⚠️ Error extracting colors: {e}")
            return []

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for integrity checking."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"⚠️ Error generating hash: {e}")
            return ""

    def _generate_asset_description(self, file_path: str) -> str:
        """Generate description of the asset using vision model."""
        try:
            prompt = """Describe this image in 1-2 sentences. Focus on:
- Main subject or objects
- Setting/environment
- Colors and mood
- Any text or notable details

Keep it concise and useful for video editing context."""

            print(f"🧠 Generating description for {os.path.basename(file_path)}")
            
            # Use existing image analysis infrastructure
            image_analysis_result: ImageAnalysisResult = describe_image(file_path, prompt)
            
            # ImageAnalysisResult is guaranteed to have scene_description, even if empty or error message
            description = image_analysis_result.scene_description
            
            # Clean up the response
            if len(description) > 200:
                description = description[:200] + "..."
            
            return description
            
        except Exception as e:
            print(f"⚠️ Error generating description: {e}")
            resource_manager.unload_all_models() # Unload models on error
            return f"Image file: {os.path.basename(file_path)} (Error: {e})" # Include error for better debugging

    def _extract_tags_from_description(self, description: str) -> List[str]:
        """Extract simple tags from description using keyword matching."""
        # Simple keyword-based tag extraction
        tag_keywords = {
            'people': ['person', 'people', 'man', 'woman', 'child', 'group'],
            'nature': ['tree', 'forest', 'mountain', 'sky', 'water', 'ocean', 'beach', 'landscape'],
            'city': ['building', 'street', 'city', 'urban', 'road', 'traffic'],
            'technology': ['computer', 'phone', 'screen', 'device', 'tech'],
            'food': ['food', 'meal', 'restaurant', 'cooking', 'kitchen'],
            'business': ['office', 'meeting', 'presentation', 'corporate', 'professional'],
            'indoor': ['indoor', 'inside', 'room', 'interior'],
            'outdoor': ['outdoor', 'outside', 'exterior'],
            'colorful': ['colorful', 'bright', 'vibrant'],
            'dark': ['dark', 'night', 'shadow'],
            'abstract': ['abstract', 'pattern', 'texture', 'geometric']
        }
        
        description_lower = description.lower()
        tags = []
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                tags.append(tag)
        
        return tags

    def _generate_suggestions_for_clip(self, clip: Dict[str, Any], b_roll_assets: List[Dict[str, Any]], audio_rhythm: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate B-roll suggestions for a specific clip."""
        clip_description_value = clip.get('clip_description')
        clip_text = str(clip_description_value) if clip_description_value is not None else ''
        
        # Create a simplified asset list for the LLM
        asset_summaries = []
        for asset in b_roll_assets:
            summary = {
                'filename': asset['filename'],
                'description': asset['description'],
                'tags': asset['tags'][:5],  # Limit tags
                'resolution': asset['resolution']
            }
            asset_summaries.append(summary)

        prompt = f"""Based on the clip text: "{clip_text}"

Suggest 2-3 relevant B-roll images from the available assets. Consider the content and mood.

Available assets (filename, description, tags):
{json.dumps(asset_summaries[:20], indent=2)}

For each suggestion, provide:
1. The exact filename
2. When to show it (start_time in seconds from clip start)
3. Duration to show (in seconds)
4. Brief reason why it fits

Format as simple lines like:
filename.jpg | 5.0 | 3.0 | Shows relevant scene
filename2.jpg | 10.0 | 2.5 | Matches the mood

Keep suggestions practical and relevant."""

        print(f"🎯 Generating suggestions for: {clip_text[:50]}...")
        
        try:
            system_prompt_suggestions = """
You are an AI assistant that provides B-roll suggestions.
You MUST respond with ONLY the suggestions formatted as specified. No other text or markdown.
"""
            response_model = llm_interaction.robust_llm_json_extraction(
                system_prompt=system_prompt_suggestions,
                user_prompt=prompt,
                output_schema=SuggestionsResponse, # Use the new schema
                raw_output_mode=True # Get raw text response
            )
            response = response_model.suggestions_text # Access the raw text from the Pydantic model
            suggestions = self._parse_suggestions_from_response(response, b_roll_assets)
            return suggestions
            
        except Exception as e:
            self.log_error(f"Failed to generate suggestions: {e}")
            return []

    def _parse_suggestions_from_response(self, response: str, b_roll_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse LLM response into structured suggestions."""
        suggestions = []
        
        # Create filename to asset mapping
        filename_to_asset = {asset['filename']: asset for asset in b_roll_assets}
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue
                
            try:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    filename = parts[0]
                    start_time = float(parts[1])
                    duration = float(parts[2])
                    reason = parts[3]
                    
                    # Find the matching asset
                    asset = filename_to_asset.get(filename) # Use .get() to safely retrieve asset
                    if asset: # Check if asset was found
                        suggestion = {
                            'b_roll_path': asset.get('path'), # Use .get() for safety
                            'filename': filename,
                            'start_time': start_time,
                            'duration': duration,
                            'reason': reason,
                            'resolution': asset.get('resolution'), # Use .get() for safety
                            'description': asset.get('description') # Use .get() for safety
                        }
                        suggestions.append(suggestion)
                        
            except (ValueError, IndexError) as e: # Catch specific errors
                self.log_warning(f"Malformed suggestion line skipped: '{line}' - Error: {e}")
                continue
            except Exception as e: # Catch any other unexpected errors
                self.log_error(f"Unexpected error parsing suggestion line: '{line}' - Error: {e}")
                continue
        
        return suggestions
