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
