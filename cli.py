import typer
from typing_extensions import Annotated
from typing import Optional
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

from main import main as run_main_pipeline # Import the main pipeline function

app = typer.Typer()

@app.command()
def run(
    video_path: Annotated[Optional[str], typer.Option("--video-path", help="Path to a local MP4 video file.")] = None,
    youtube_url: Annotated[Optional[str], typer.Option("--youtube-url", help="URL of a YouTube video to download.")] = None,
    youtube_quality: Annotated[Optional[int], typer.Option("--youtube-quality", help="Desired YouTube video quality option (e.g., 0, 1, 2...).")] = None,
    user_prompt: Annotated[Optional[str], typer.Option(help="Optional: A specific prompt for the LLM to guide clip selection.")] = None,
    retry: Annotated[bool, typer.Option(help="Automatically resume from a previous failed session.")] = False,
    nocache: Annotated[bool, typer.Option(help="Force a fresh start, deleting any existing state and temporary files.")] = False,
):
    """
    Runs the main video processing pipeline.
    """
    typer.echo("Starting video processing pipeline...")
    # Pass arguments as a dictionary to the main pipeline function
    args = {
        "video_path": video_path,
        "youtube_url": youtube_url,
        "youtube_quality": youtube_quality,
        "user_prompt": user_prompt,
        "retry": retry,
        "nocache": nocache,
    }
    run_main_pipeline(args)

@app.command()
def config():
    """
    Manages the application's configuration. (Not yet implemented)
    """
    typer.echo("Configuration management (not yet implemented).")

@app.command()
def tools():
    """
    Accesses utility scripts, such as downloading models. (Not yet implemented)
    """
    typer.echo("Tools management (not yet implemented).")

@app.command()
def state():
    """
    Manages the application's state, such as deleting the state file. (Not yet implemented)
    """
    typer.echo("State management (not yet implemented).")

if __name__ == "__main__":
    app()
