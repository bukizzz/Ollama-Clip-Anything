import cv2
import numpy as np
from typing import Tuple, Optional
from moviepy.editor import VideoClip, concatenate_videoclips # MoviePy is needed for VideoClip and concatenate_videoclips

class VideoEffects:
    """Advanced video effects and transitions"""
    
    @staticmethod
    def apply_zoom_effect(clip: VideoClip, zoom_factor: float = 1.2, focus_point: Optional[Tuple[int, int]] = None) -> VideoClip:
        """Apply zoom effect with optional focus point"""
        w, h = clip.size
        
        if focus_point is None:
            focus_point = (w // 2, h // 2)
        
        def zoom_frame(get_frame, t):
            frame = get_frame(t).copy()
            zoom = 1 + (zoom_factor - 1) * min(t / clip.duration, 1)
            
            # Calculate new dimensions
            new_w, new_h = int(w * zoom), int(h * zoom)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Calculate crop to maintain original size
            crop_x = max(0, (new_w - w) // 2)
            crop_y = max(0, (new_h - h) // 2)
            
            # Adjust crop based on focus point
            focus_x, focus_y = focus_point
            crop_x = max(0, min(crop_x + int((focus_x - w//2) * (zoom - 1)), new_w - w))
            crop_y = max(0, min(crop_y + int((focus_y - h//2) * (zoom - 1)), new_h - h))
            
            cropped = resized[crop_y:crop_y + h, crop_x:crop_x + w]
            
            # Ensure we have the right dimensions
            if cropped.shape[:2] != (h, w):
                cropped = cv2.resize(cropped, (w, h))
                
            return cropped
        
        return clip.fl(zoom_frame)
    
    @staticmethod
    def apply_scene_transition(clip1: VideoClip, clip2: VideoClip, transition_duration: float = 1.0, transition_type: str = 'crossfade') -> VideoClip:
        """Apply transitions between scenes"""
        if transition_type == 'crossfade':
            clip1_out = clip1.fadeout(transition_duration)
            clip2_in = clip2.fadein(transition_duration).set_start(clip1.duration - transition_duration)
            return concatenate_videoclips([clip1_out, clip2_in])
        
        elif transition_type == 'slide':
            # Implement slide transition
            w, h = clip1.size
            
            def slide_effect(get_frame, t):
                if t < clip1.duration - transition_duration:
                    return clip1.get_frame(t)
                elif t > clip1.duration:
                    return clip2.get_frame(t - clip1.duration)
                else:
                    # During transition
                    progress = (t - (clip1.duration - transition_duration)) / transition_duration
                    frame1 = clip1.get_frame(clip1.duration - 0.01)
                    frame2 = clip2.get_frame(0)
                    
                    # Slide effect
                    offset = int(w * progress)
                    result = np.zeros_like(frame1)
                    result[:, :w-offset] = frame1[:, offset:]
                    result[:, w-offset:] = frame2[:, :offset]
                    
                    return result
            
            total_duration = clip1.duration + clip2.duration - transition_duration
            return VideoClip(slide_effect, duration=total_duration)
        
        # Default to crossfade
        return VideoEffects.apply_scene_transition(clip1, clip2, transition_duration, 'crossfade')

    @staticmethod
    def apply_color_grading(clip: VideoClip, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0) -> VideoClip:
        """Apply color grading effects (brightness, contrast, saturation)"""
        def color_grade_frame(get_frame, t):
            frame = get_frame(t).copy()

            # Apply brightness and contrast
            # new_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            # A more robust way to apply brightness and contrast
            new_frame = frame * contrast + brightness
            new_frame = np.clip(new_frame, 0, 255).astype(np.uint8)

            # Apply saturation
            if saturation != 1.0:
                hsv_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
                hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] * saturation, 0, 255)
                new_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            
            return new_frame
        
        return clip.fl(color_grade_frame)

    from moviepy.editor import TextClip

    @staticmethod
    def add_text_overlay(clip: VideoClip, text: str, duration: float, font_size: int = 50, color: str = 'white', position: Tuple[int, int] = ('center', 'center')) -> VideoClip:
        """Adds a text overlay to a clip."""
        txt_clip = TextClip(text, fontsize=font_size, color=color, font='Arial-Bold')
        txt_clip = txt_clip.set_pos(position).set_duration(duration)
        return concatenate_videoclips([clip, txt_clip.set_start(clip.duration - duration)], method="compose")

    @staticmethod
    def apply_simple_animation(clip: VideoClip, animation_type: str = 'fade_in', duration: float = 1.0) -> VideoClip:
        """Applies simple animations to a clip."""
        if animation_type == 'fade_in':
            return clip.fadein(duration)
        elif animation_type == 'fade_out':
            return clip.fadeout(duration)
        else:
            print(f"Unknown animation type: {animation_type}. Skipping animation.")
            return clip
