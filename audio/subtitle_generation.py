from core.config import config
from typing import List, Dict, Optional

def format_time(seconds):
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    return f"{h:01d}:{m:02d}:{s:02d}.{ms:02d}"

def create_ass_file(
    timestamps: List[Dict], 
    output_path: str, 
    time_offset: int = 0, 
    video_height: int = 1080, 
    layout: str = 'default',
    speaker_colors: Optional[Dict[str, str]] = None,
    is_vertical: bool = False
):
    """Creates a professional ASS subtitle file with advanced text synchronization."""
    with open(output_path, "w") as f:
        f.write("[Script Info]\nTitle: Advanced Subtitles\nScriptType: v4.00+\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")

        font_name = config.get('subtitle_font_families')[0] if config.get('subtitle_font_families') else 'Arial'
        font_size = int(config.get('subtitle_font_size') * 1.5) if is_vertical else config.get('subtitle_font_size')

        def hex_to_bgr(hex_color):
            return f"&H00{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}"

        primary_color = hex_to_bgr(config.get('subtitle_font_color'))
        outline_color = hex_to_bgr(config.get('subtitle_outline_color'))

        base_style = f"{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,2,1,5,10,10,20,1"
        f.write(f"Style: Default,{base_style}\n")
        if speaker_colors:
            for speaker, color in speaker_colors.items():
                f.write(f"Style: {speaker},{font_name},{font_size},{hex_to_bgr(color)},&H000000FF,{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,2,1,5,10,10,20,1\n")

        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        last_word_end = 0
        for segment in timestamps:
            speaker = segment.get('speaker', 'Default')
            style = speaker if speaker_colors and speaker in speaker_colors else 'Default'
            
            for word in segment.get('words', []):
                start, end = word['start'] - time_offset, word['end'] - time_offset
                text = word['text'].strip()
                if start >= end:
                    continue

                # Handle pauses
                if last_word_end > 0 and start - last_word_end > 0.5: # Pause > 0.5s
                    # Insert a non-breaking space or similar pause indicator if needed
                    pass

                # Optimize display duration
                if (end - start) < 0.2:
                    end = start + 0.2 # Min duration for readability

                # Positioning and animation
                pos_y = video_height * (0.8 if not is_vertical else 0.7)
                tags = f"{{\an5\pos(960, {pos_y})}}" # Centered, adjustable Y
                if config.get('subtitle_animation.word_by_word_timing_enabled'):
                    tags += "\fad(150,150)"
                if config.get('subtitle_animation.emphasis_effects_enabled') and word.get('emphasis'):
                    tags += "\\fscx120\\fscy120\\c&H00FFFF&"

                f.write(f"Dialogue: 0,{format_time(start)},{format_time(end)},{style},,0,0,0,,{tags}{text}\n")
                last_word_end = end
