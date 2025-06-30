

from core.config import SUBTITLE_FONT_SIZE, SUBTITLE_FONT_COLOR, SUBTITLE_OUTLINE_COLOR, SUBTITLE_SHADOW_COLOR

def format_time(seconds):
    """Converts seconds to ASS time format (H:MM:SS.ss)"""
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int((seconds % 60))
    ms = int((seconds - int(seconds)) * 100)
    return f"{h:01d}:{m:02d}:{s:02d}.{ms:02d}"

def create_ass_file(timestamps, output_path="subtitles.ass", time_offset=0):
    """
    Creates an ASS subtitle file with word-by-word highlighting.
    """
    with open(output_path, "w") as f:
        # Write ASS header
        f.write("[Script Info]\n")
        f.write("Title: Animated Subtitles\n")
        f.write("ScriptType: v4.00+\n")
        f.write("WrapStyle: 0\n")
        f.write("ScaledBorderAndShadow: yes\n")
        f.write("\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # Define a style for the highlighted word and the rest of the line
        f.write(f"Style: Default,Arial,{SUBTITLE_FONT_SIZE},&H00{SUBTITLE_FONT_COLOR},&H000000FF,&H00{SUBTITLE_OUTLINE_COLOR},&H00{SUBTITLE_SHADOW_COLOR},0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1\n")
        f.write("\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        # Write each word as a separate event
        for segment in timestamps:
            for word in segment['words']:
                start_time = format_time(word['start'] - time_offset)
                end_time = format_time(word['end'] - time_offset)
                text = word['text'].strip()
                f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
