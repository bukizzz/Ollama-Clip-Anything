import random
import math
import spacy
import cv2
from typing import List, Dict, Tuple
from moviepy.editor import TextClip, VideoClip # MoviePy is needed for TextClip and VideoClip
from core.config import SUBTITLE_FONT_FAMILIES

# Initialize models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class SubtitleGenerator:
    """Advanced subtitle generation with animations and styling"""
    
    def __init__(self):
        self.font_families = SUBTITLE_FONT_FAMILIES
        self.animation_styles = ['fade', 'slide', 'zoom', 'bounce', 'typewriter']
        
    def analyze_text_sentiment(self, text: str) -> str:
        """Analyze text sentiment for styling"""
        if nlp is None:
            return 'neutral'
        
        # Simple sentiment analysis based on token sentiment
        # We don't need to assign doc to a variable if it's not used directly.
        # The nlp(text) call itself performs the necessary processing.
        nlp(text)
        positive_words = ['great', 'amazing', 'awesome', 'fantastic', 'excellent', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def get_subtitle_style(self, sentiment: str, emphasis: bool = False) -> Dict:
        """Get styling based on sentiment and emphasis"""
        base_style = {
            'fontsize': 45 if emphasis else 35,
            'color': 'white',
            'stroke_color': 'black',
            'stroke_width': 2,
            'method': 'caption'
        }
        
        if sentiment == 'positive':
            base_style.update({
                'color': '#00FF7F',  # Spring green
                'stroke_color': '#006400'  # Dark green
            })
        elif sentiment == 'negative':
            base_style.update({
                'color': '#FF6B6B',  # Light red
                'stroke_color': '#8B0000'  # Dark red
            })
        elif emphasis:
            base_style.update({
                'color': '#FFD700',  # Gold
                'stroke_color': '#FF8C00'  # Dark orange
            })
        
        return base_style
    
    def create_word_by_word_subtitles(self, transcript: List[Dict], video_duration: float, video_size: Tuple[int, int]) -> List[VideoClip]:
        """Create animated word-by-word subtitles"""
        subtitle_clips = []
        w, h = video_size
        
        for segment in transcript:
            if 'words' not in segment:
                continue
            
            words = segment['words']
            segment_sentiment = self.analyze_text_sentiment(segment.get('text', ''))
            
            for i, word_data in enumerate(words):
                if isinstance(word_data, dict):
                    word = word_data.get('word', '')
                    start_time = float(word_data.get('start', 0))
                    end_time = float(word_data.get('end', start_time + 0.5))
                else:
                    # Handle simple string format
                    word = str(word_data)
                    start_time = segment.get('start', 0) + i * 0.3
                    end_time = start_time + 0.5
                
                if not word.strip():
                    continue
                
                duration = end_time - start_time
                if duration <= 0:
                    duration = 0.5
                
                # Determine if word should be emphasized
                emphasis = word.upper() == word and len(word) > 2
                
                # Get styling
                style = self.get_subtitle_style(segment_sentiment, emphasis)
                
                # Create base text clip
                txt_clip = TextClip(
                    word.strip(),
                    font=random.choice(self.font_families),
                    **style
                ).set_duration(duration).set_start(start_time)
                
                # Position at bottom center
                txt_clip = txt_clip.set_position(('center', h * 0.85))
                
                # Add animation
                animation_style = random.choice(self.animation_styles)
                txt_clip = self.apply_text_animation(txt_clip, animation_style, duration)
                
                subtitle_clips.append(txt_clip)
        
        return subtitle_clips
    
    def apply_text_animation(self, text_clip: TextClip, animation_style: str, duration: float) -> VideoClip:
        """Apply various animations to text clips"""
        if animation_style == 'fade':
            return text_clip.fadein(0.2).fadeout(0.2)
        
        elif animation_style == 'slide':
            w, h = text_clip.size
            return text_clip.set_position(lambda t: ('center', max(100, 200 - t * 100)))
        
        elif animation_style == 'zoom':
            def zoom_effect(get_frame, t):
                frame = get_frame(t).copy()
                if t < 0.3:
                    scale = 0.5 + (t / 0.3) * 0.5
                    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
                    frame = cv2.resize(frame, new_size)
                return frame
            return text_clip.fl(zoom_effect, apply_to=['mask'])
        
        elif animation_style == 'bounce':
            def bounce_effect(t):
                bounce_height = 20 * abs(math.sin(t * math.pi * 4))
                return ('center', text_clip.pos(0)[1] - bounce_height)
            return text_clip.set_position(bounce_effect)
        
        elif animation_style == 'typewriter':
            # This would require more complex implementation
            return text_clip.fadein(0.1)
        
        return text_clip
