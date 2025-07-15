# Product Context: 60-Second Clips Generator

## Why this project exists:
In today's fast-paced digital landscape, short-form video content dominates. Content creators, marketers, and businesses constantly need engaging, high-quality clips for platforms like TikTok, YouTube Shorts, and Instagram Reels. Manually identifying and editing these clips from longer videos is time-consuming and requires specialized skills. This project aims to automate and optimize this process, making viral content creation accessible and efficient.

## Problems it solves:
- **Time-consuming manual editing:** Automates the laborious process of reviewing long videos and manually cutting clips.
- **Subjectivity in content selection:** Leverages AI to objectively identify engaging moments based on multimodal analysis, reducing reliance on human intuition alone.
- **Lack of virality optimization:** Integrates insights from engagement, sentiment, and hook analysis to increase the likelihood of clips going viral.
- **Technical barriers:** Simplifies complex video processing and AI model integration into a streamlined pipeline.
- **Scalability:** Enables processing of multiple videos efficiently, supporting high-volume content production.

## How it should work:
1. **Input:** User provides a video file path or YouTube URL.
2. **Preprocessing:** Video and audio are extracted and pre-processed (e.g., transcription, smart frame extraction).
3. **Analysis:** Comprehensive multimodal analysis is performed, including:
    - **Audio:** Transcription, speaker diarization, sentiment analysis, rhythm detection.
    - **Visual:** Scene detection, facial expression analysis, gesture recognition, visual complexity, layout detection, object tracking (conceptual).
    - **LLM-driven:** Storyboarding, content alignment, hook identification, viral potential scoring, B-roll suggestions.
4. **Selection:** LLMs intelligently select optimal 60-90 second clips based on analysis insights and user prompts.
5. **Production:** Selected clips are enhanced with dynamic editing, music synchronization, and animated subtitles.
6. **Output:** High-quality, ready-to-share short video clips are generated.

## User experience goals:
- **Simplicity:** Easy to use with minimal configuration, even for non-technical users.
- **Efficiency:** Rapid generation of clips, significantly faster than manual methods.
- **Quality:** Produced clips should be professional-grade and highly engaging.
- **Customization:** Allow for user prompts to guide the AI's content selection.
- **Transparency:** Provide clear logs and reports on the processing steps and outcomes.
- **Reliability:** Robust error handling and resume capabilities to ensure smooth operation.
