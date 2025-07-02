#Video Processing Pipeline Evolution Instructions

## Phase 1: Foundation Refactoring (Based on v4.0.9.2)

### Core Infrastructure Updates
- Update `core/config.yaml` to include all new pipeline configuration sections:
  - `engagement_analysis`: thresholds for facial expressions, gesture detection, energy levels
  - `audio_rhythm`: tempo detection parameters, beat matching sensitivity
  - `layout_detection`: multi-person detection thresholds, screen share identification
  - `subtitle_animation`: word-by-word timing, emphasis effects, speaker color coding
  - `music_integration`: mood detection, tempo matching, beat synchronization
  - `intro_narration`: voice cloning settings, tone matching, duration limits (5 seconds max)
  - `qwen_vision`: frame extraction rate, resolution settings, temporal encoding parameters
- Extend `core/config.py` to load all new configuration sections from YAML
- Add new dependencies to `requirements.txt`:
  - `librosa>=0.10.0` (for advanced audio rhythm analysis)
  - `scikit-learn>=1.3.0` (for engagement metric clustering)
  - `opencv-contrib-python>=4.8.0` (for advanced face analysis)
  - `sentence-transformers>=2.2.0` (for content semantic analysis)
  - `transformers>=4.30.0` (for multimodal LLM integration)
  - `torch-audio>=2.0.0` (for voice cloning capabilities)
  - `qwen-vl>=2.5.0` (for advanced video understanding)

### State Management Enhancement
- Extend `core/state_manager.py` to handle new pipeline stages:
  - `audio_rhythm_analysis_complete`
  - `engagement_analysis_complete`
  - `layout_detection_complete`
  - `multimodal_analysis_complete`
  - `speaker_tracking_complete`
  - `intro_narration_generated`
  - `qwen_vision_analysis_complete`
  - `frame_feature_extraction_complete`
- Add checkpoint recovery for each new stage
- Implement rollback capability for failed stages

### GPU Memory Management
- Enhance `core/gpu_manager.py` to handle multiple model types:
  - Face analysis models (MediaPipe, custom emotion detection)
  - Audio processing models (Whisper, rhythm detection)
  - Multimodal LLM models (MiniCPM, ImageBind)
  - Voice cloning models (TTS library)
  - Qwen2.5-VL vision models (dynamic resolution processing)
- Implement model loading queue with priority system
- Add automatic model unloading when memory threshold reached

## Phase 2: Enhanced Audio Processing Pipeline

### Audio Rhythm Analysis Agent
- Create `agents/audio_rhythm_agent.py`:
  - Extract audio tempo using `librosa.beat.tempo`
  - Detect beat positions with `librosa.beat.beat_track`
  - Analyze speech rhythm patterns (pause detection, emphasis detection)
  - Generate rhythm map for dynamic editing synchronization
  - Save rhythm data to state for clip processing
- Integrate rhythm analysis into existing `AudioTranscriptionAgent`
- Modify `audio/audio_processing.py` to include rhythm extraction functions

### Enhanced Audio Analysis Agent
- Create `agents/audio_analysis_agent.py`:
  - Perform speaker diarization using `pyannote.audio`
  - Extract audio sentiment using `transformers` emotion models
  - Detect speech energy levels and vocal emphasis
  - Identify audio themes using semantic analysis
  - Map speakers to timestamp ranges for visual alignment
- Update existing audio processing to include speaker tracking

### Intro Narration Agent
- Create `agents/intro_narration_agent.py`:
  - Analyze selected clips to identify key themes and hooks
  - Generate compelling intro narration using LLM (5 seconds max)
  - Ensure non-clickbait approach with honest content framing
  - Match narration tone to original content mood
  - Integrate with voice cloning for speaker consistency
  - Save narration audio files to temp directory

## Phase 3: Advanced Video Analysis Pipeline

### Frame Preprocessing Agent
- Create `agents/frame_preprocessing_agent.py`:
  - Decompose video into individual frames at configurable frame rate (1 fps default)
  - Implement metadata alignment with accurate timestamp preservation
  - Optimize frame extraction for processing efficiency
  - Prepare frames in format compatible with Qwen2.5-VL requirements
  - Generate frame-to-timestamp mapping for temporal context

### Qwen2.5-VL Integration Agent
- Create `agents/qwen_vision_agent.py`:
  - Integrate Qwen2.5-VL for advanced video understanding and object localization
  - Process frames using dynamic resolution processing and absolute time encoding
  - Extract structured features: bounding boxes, object classifications, temporal event markers
  - Generate comprehensive visual scene analysis with long-video comprehension
  - Output structured data in JSON format for downstream processing
- Leverage Qwen2.5-VL's visual recognition capabilities for enhanced content analysis

### Enhanced Video Analysis Agent
- Extend `agents/video_analysis_agent.py`:
  - Add facial expression analysis using `opencv-contrib-python`
  - Implement gesture recognition using MediaPipe Holistic
  - Calculate visual complexity scores per frame
  - Detect energy levels based on movement and facial expressions
  - Extract frame-level engagement metrics
  - Integrate with Qwen2.5-VL feature extraction for comprehensive analysis
- Integrate with existing face and object tracking
- Output structured engagement data for LLM selection

### Engagement Analysis Agent  
- Create `agents/engagement_analysis_agent.py`:
  - Score frames based on facial expressions (surprise, excitement, laughter)
  - Detect gesture emphasis (pointing, hand movements, body language)
  - Identify hook moments (surprising reactions, strong statements)
  - Calculate viral potential scores using engagement metrics
  - Rank video segments by engagement potential
- Integrate engagement scores into clip selection criteria

### Layout Detection Agent
- Create `agents/layout_detection_agent.py`:
  - Detect number of visible faces per frame
  - Identify screen sharing vs. camera-only content
  - Classify layout types: single person, multi-person, presentation mode
  - Detect speaker transitions and layout changes
  - Map optimal layout configurations per video segment
- Store layout recommendations in analysis results

### Enhanced Storyboarding Agent
- Extend `agents/storyboarding_agent.py`:
  - Integrate multimodal LLM analysis at scene boundaries only
  - Analyze visual context using ImageBind or MiniCPM models  
  - Identify content types: discussion, presentation, demo, reaction
  - Detect hook potential and viral moments
  - Generate scene-level content descriptions
  - Incorporate Qwen2.5-VL structured data for enhanced scene understanding
- Optimize for efficiency by analyzing only scene change frames

## Phase 4: Intelligent Content Integration

### Content Alignment Agent
- Extend `agents/content_alignment_agent.py`:
  - Align transcript with visual features at frame level
  - Map speakers to face tracking IDs
  - Sync audio rhythm data with visual engagement metrics
  - Correlate scene changes with content topic shifts
  - Generate comprehensive content-visual alignment map
  - Integrate Qwen2.5-VL temporal markers with audio transcription
- Integrate all analysis results into unified content structure

### Speaker Tracking Agent
- Create `agents/speaker_tracking_agent.py`:
  - Maintain consistent speaker IDs across video duration
  - Map speaker audio segments to face tracking results
  - Handle speaker overlap and multi-person conversations
  - Generate speaker transition timestamps
  - Create speaker visual profiles for layout optimization
- Integrate with existing face tracking infrastructure

### Hook Identification Agent  
- Create `agents/hook_identification_agent.py`:
  - Identify compelling opening moments using engagement metrics
  - Detect surprising statements or reactions
  - Score viral potential based on facial expressions and audio emphasis
  - Find quotable moments with high social media potential
  - Prioritize content with strong narrative hooks
- Feed hook data into LLM selection for optimal clip starts

## Phase 5: Advanced Clip Selection

### LLM Video Director Integration
- Create `agents/llm_video_director_agent.py`:
  - Integrate DirectorLLM or equivalent for orchestrating video content
  - Process structured data from Qwen2.5-VL and other analysis agents
  - Analyze sequence of events and object interactions for logical cut points
  - Ensure narrative coherence and emphasize significant events
  - Generate intelligent cut decisions based on comprehensive scene analysis
- Feed structured visual and audio data for informed editing decisions

### Improved LLM Selection Agent
- Enhance `agents/llm_selection_agent.py`:
  - Integrate transcript + visual features + engagement metrics + scene boundaries
  - Use multimodal analysis results for content-aware selection
  - Prioritize clips with high engagement scores and hook potential
  - Consider layout requirements for different content types
  - Select clips optimized for viral sharing and viewer retention
  - Incorporate LLM video director recommendations for optimal cut placement
- Update LLM prompts to include all new data types including Qwen2.5-VL outputs

### Viral Potential Agent
- Create `agents/viral_potential_agent.py`:
  - Score clips based on engagement metrics, hook potential, quotability
  - Analyze emotional impact using facial expression and audio sentiment
  - Identify shareable moments with broad audience appeal
  - Rank clips by social media virality potential
  - Generate viral optimization recommendations
- Integrate viral scores into final clip selection

## Phase 6: Dynamic Editing and Effects

### Enhanced Clip Processor
- Extend `video/video_editing.py`:
  - Implement rhythm-based dynamic editing using audio beat data
  - Add beat-synced zoom in/out effects based on audio tempo
  - Create smooth layout transitions for multi-person content
  - Integrate speaker-aware visual effects and focusing
  - Apply engagement-optimized cuts at high-energy moments
  - Execute cuts determined by LLM video director for enhanced storytelling
- Maintain compatibility with existing clip generation

### Dynamic Editing Agent
- Create `agents/dynamic_editing_agent.py`:
  - Generate editing decisions based on audio rhythm and visual engagement
  - Calculate optimal cut points using beat alignment and content flow
  - Apply dynamic effects: rhythm-synced zoom, beat-matched transitions
  - Optimize clip pacing for maximum viewer retention
  - Ensure smooth visual flow between different content segments
  - Implement LLM video director recommendations with quality assurance
- Integrate with existing video editing pipeline

### Music Sync Agent
- Create `agents/music_sync_agent.py`:
  - Select background music based on content mood and genre
  - Match music tempo to video rhythm and speaking pace  
  - Synchronize music beats with visual cuts and transitions
  - Adjust music volume levels to complement speech audio
  - Generate seamless audio mixing for professional output
- Integrate with audio processing and clip generation

### Layout Optimization Agent
- Create `agents/layout_optimization_agent.py`:
  - Apply optimal layouts based on content type and speaker count
  - Implement multi-person conversation layouts: split screen, grid, focus
  - Handle screen share + speaker combinations with optimal positioning
  - Create dynamic speaker focus with smooth transitions
  - Generate professional broadcast-quality visual compositions
- Integrate with frame processing and video editing

### Subtitle Animation Agent
- Create `agents/subtitle_animation_agent.py`:
  - Generate word-by-word animated subtitles using precise timing
  - Apply dynamic text effects: scale-in, fade-in, slide-up animations
  - Implement emphasis styling for key words and phrases
  - Use speaker color coding for multi-person conversations
  - Optimize subtitle positioning for different layout types
- Enhance existing subtitle generation with animation capabilities

## Phase 7: Advanced Visual Effects and Layout Management

### Enhanced Frame Processor
- Extend `video/frame_processor.py`:
  - Implement intelligent layout switching based on content type
  - Add smooth morphing transitions between layout configurations
  - Apply active speaker highlighting with subtle visual effects
  - Generate speaker labels that appear/disappear with speech activity
  - Ensure voice-visual synchronization for subtitle placement
- Maintain compatibility with existing dynamic framing

### Advanced Layout System
- Create `video/layout_manager.py`:
  - Define layout templates: single speaker, multi-person, presentation, screen share
  - Implement smooth animated transitions between layout modes
  - Handle content-aware layout adaptation (screen share detection)
  - Apply engagement-driven focus (zoom on animated speakers)
  - Generate seamless morphing effects between different compositions
- Integrate with frame processing and video editing

### Professional Text Synchronization
- Enhance `audio/subtitle_generation.py`:
  - Implement precise word-level timing with speech pattern analysis
  - Handle natural speech pauses with intelligent spacing
  - Manage multiple speaker subtitle positioning to avoid conflicts
  - Optimize subtitle display duration for comfortable reading
  - Apply mobile-optimized font sizes and positioning for vertical content
- Maintain compatibility with existing subtitle workflow

## Phase 8: Advanced Audio Features

### Voice Cloning Integration
- Enhance `audio/voice_cloning.py`:
  - Implement full TTS library integration for speaker voice cloning
  - Generate intro narrations using original speaker voices
  - Ensure voice quality matches original audio characteristics
  - Handle voice consistency across different generated content
  - Optimize for real-time processing and memory efficiency
- Integrate with intro narration generation

### Advanced Audio Processing
- Extend `audio/audio_processing.py`:
  - Implement advanced voice separation using Demucs models
  - Add audio enhancement and noise reduction capabilities
  - Apply dynamic audio mixing for intro narration and background music
  - Optimize audio levels and EQ for different content types
  - Generate professional-quality audio output
- Maintain compatibility with existing transcription workflow

## Phase 9: B-roll and Content Enhancement  

### B-roll Integration Agent
- Enhance `agents/broll_analysis_agent.py`:
  - Focus B-roll analysis on selected clips only (not entire video)
  - Generate contextual B-roll suggestions based on clip content
  - Integrate B-roll timing with audio rhythm and visual cuts
  - Apply smooth B-roll transitions synchronized with beat patterns
  - Optimize B-roll selection for engagement and content relevance
- Integrate with existing B-roll asset management

### Content Enhancement Pipeline
- Create final integration layer combining all agents:
  - Coordinate between all processing agents for optimal resource usage
  - Ensure smooth data flow between analysis and production stages
  - Implement quality checks at each stage with automatic retry capability
  - Generate comprehensive processing reports with metrics and recommendations
  - Optimize overall pipeline performance and memory usage

## Phase 10: Production Quality Output

### Final Quality Assurance
- Implement comprehensive quality checks:
  - Verify audio-video synchronization across all clips
  - Validate subtitle timing and positioning accuracy
  - Check layout transitions for smoothness and professionalism
  - Ensure consistent visual quality and effects application
  - Validate engagement optimization and viral potential scores
  - Review LLM video director cut decisions for narrative coherence

### Performance Optimization
- Optimize entire pipeline for production use:
  - Implement parallel processing where possible
  - Optimize GPU memory usage across all model types
  - Add progress tracking and ETA estimation for long videos
  - Implement intelligent caching for repeated processing
  - Generate detailed performance metrics and bottleneck analysis

### Documentation and Testing
- Create comprehensive documentation for new pipeline:
  - Document all new configuration options and their effects
  - Provide examples of optimal settings for different content types
  - Create troubleshooting guide for common issues
  - Generate performance benchmarks for different hardware configurations
  - Implement automated testing for all pipeline stages

## Implementation Priority Order

1. **Phase 1**: Foundation (Core infrastructure, dependencies, state management)
2. **Phase 2**: Audio Pipeline (Rhythm analysis, speaker tracking, intro narration)  
3. **Phase 3**: Video Analysis (Frame preprocessing, Qwen2.5-VL integration, engagement metrics, layout detection, enhanced storyboarding)
4. **Phase 4**: Content Integration (Alignment, speaker tracking, hook identification)
5. **Phase 5**: Clip Selection (LLM video director integration, enhanced LLM selection, viral potential scoring)
6. **Phase 6**: Dynamic Editing (Rhythm-based editing, music sync, layout optimization)
7. **Phase 7**: Visual Effects (Advanced layouts, subtitle animation, professional transitions)
8. **Phase 8**: Audio Features (Voice cloning, advanced processing)
9. **Phase 9**: B-roll Integration (Targeted B-roll, content enhancement)
10. **Phase 10**: Production Quality (QA, optimization, documentation)

## Critical Success Factors

- Maintain backward compatibility with existing v4.0.9.2 functionality
- Implement comprehensive error handling and recovery for all new components
- Optimize GPU memory usage to handle multiple AI models NEVER CONCURRENTLY!
- Remove cli.py and keep the user menus we already have built into main.py and others.
- Maintain processing speed while adding advanced analysis capabilities
- Generate production-quality output that rivals professional video editing software
- Ensure seamless integration between Qwen2.5-VL vision analysis and LLM video director for coherent editing decisions
