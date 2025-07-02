from core.config import SUBTITLE_FONT_SIZE, SUBTITLE_FONT_COLOR, SUBTITLE_OUTLINE_COLOR, SUBTITLE_FONT_FAMILIES, SUBTITLE_BACKGROUND_COLOR

def format_time(seconds):
    """Converts seconds to ASS time format (H:MM:SS.ss)"""
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int((seconds % 60))
    ms = int((seconds - int(seconds)) * 100)
    return f"{h:01d}:{m:02d}:{s:02d}.{ms:02d}"

def create_ass_file(timestamps, output_path="subtitles.ass", time_offset=0, video_height: int = 1080, split_screen_mode: bool = False, word_by_word_timing: bool = True, emphasis_effects: bool = False, speaker_color_coding: bool = False):
    """
    Creates an ASS subtitle file with word-by-word highlighting.
    """
    with open(output_path, "w") as f:
        # Write ASS header
        f.write("[Script Info]\n")
        print(f"DEBUG: ASS file created at: {output_path}")
        f.write("Title: Animated Subtitles\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write("\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        # Define a style for the single word, with background and rounded borders
        # Using the first font from SUBTITLE_FONT_FAMILIES
        font_name = SUBTITLE_FONT_FAMILIES[0] if SUBTITLE_FONT_FAMILIES else 'Arial'
        
        # ASS uses BGR for colors, so convert RGB (from config) to BGR
        def rgb_to_bgr_ass(hex_color):
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"

        primary_color_ass = rgb_to_bgr_ass(SUBTITLE_FONT_COLOR)
        outline_color_ass = rgb_to_bgr_ass(SUBTITLE_OUTLINE_COLOR)
        background_color_ass = rgb_to_bgr_ass(SUBTITLE_BACKGROUND_COLOR)

        # Bold (1 for bold, 0 for not bold)
        bold_setting = 1 

        if split_screen_mode:
            margin_v = int(video_height * 0.5 - SUBTITLE_FONT_SIZE / 2) # Approximately middle
            alignment_tag = "\\an5" # Middle-center
        else:
            margin_v = int(video_height * 0.30)
            alignment_tag = "\\an2" # Bottom-center

        # Add the BorderStyle and Outline/Shadow values from config
        # BorderStyle 1 is for outline+shadow. Outline and Shadow values are in pixels.
        # For rounded corners, we can use the \bord and \shad tags, and rely on the player's rendering.
        # A common way to achieve rounded corners in ASS is with drawing commands, but that's more complex.
        # For simplicity, we'll use the BackColour and rely on the player's default rendering.
        # If SUBTITLE_BORDER_RADIUS is to be used, it would typically involve drawing commands or a custom renderer.
        # For now, we'll include it in the style definition as a placeholder for future implementation.
        f.write(f"Style: SingleWord,{font_name},{SUBTITLE_FONT_SIZE},{primary_color_ass},&H000000FF,{outline_color_ass},{background_color_ass},{bold_setting},0,0,0,100,100,0,0,1,2,1,2,10,10,{margin_v},1\n")
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        # Write each word as a separate event
        for segment in timestamps:
            if 'words' in segment and segment['words']:
                for word in segment['words']:
                    word_start = word['start'] - time_offset
                    word_end = word['end'] - time_offset
                    word_text = word['text'].strip()

                    if word_start >= word_end: # Skip if invalid time range
                        continue

                    # Use the SingleWord style for each word
                    # \bord<size> for border, \shad<size> for shadow
                    # \blur<size> for blur (can simulate rounded corners with background)
                    # For rounded corners, ASS doesn't have a direct tag. It's usually done via
                    # the rendering engine or by drawing shapes. For a simple background,
                    # we can rely on the player's rendering of the BackColour.
                    # Some players might support  for rounded rectangles, but it's not standard.
                    # We'll use the BackColour and rely on the player's default rendering.
                    
                    # To simulate rounded corners with a solid background, we might need to
                    # use drawing tags, which is more complex. For now, we'll set the BackColour
                    # and assume the player handles it.
                    
                    # For a solid background, we set the BackColour.
                    # For rounded borders, ASS doesn't have a direct tag. This is usually
                    # handled by the renderer or by drawing shapes. We'll set the BackColour
                    # and rely on the player's default rendering.
                    
                    # The \an5 tag centers the text.
                    # The \1c&H00RRGGBB& for primary color, \3c&H00RRGGBB& for outline, \4c&H00RRGGBB& for shadow, \b1 for bold
                    # We've defined the style, so we just need to apply it.
                    
                    # To implement SUBTITLE_BORDER_RADIUS, we would typically use drawing commands like {\p1}m 0 0 l 100 0 100 100 0 100{\p0}
                    # However, this is significantly more complex and requires calculating the bounding box for each word.
                    # For a simpler approach that leverages the existing style, we can't directly apply rounded corners
                    # without a custom renderer or more advanced ASS drawing.
                    # For now, I will add a comment indicating where it would be implemented if drawing commands were used.
                    # The current implementation relies on the player's default rendering of the BackColour.
                    
                    # Placeholder for rounded corners using drawing commands (requires complex bounding box calculations):
                    # f.write(f"Dialogue: 0,{format_time(word_start)},{format_time(word_end)},SingleWord,,0,0,0,,{{\p1}}m {x1} {y1} l {x2} {y1} {x2} {y2} {x1} {y2}{{\p0}}{alignment_tag}{word_text}\n")
                    
                    # For now, we will just use the existing style and rely on the player's rendering.
                    # The SUBTITLE_BORDER_RADIUS is passed to the config, but its direct visual effect
                    # is dependent on the ASS renderer's capabilities.
                    f.write(f"Dialogue: 0,{format_time(word_start)},{format_time(word_end)},SingleWord,,0,0,0,,{alignment_tag}{word_text}\n")
