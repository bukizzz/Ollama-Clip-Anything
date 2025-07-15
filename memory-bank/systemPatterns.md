# System Patterns: 60-Second Clips Generator

## System Architecture:
The system employs a modular, agent-based pipeline architecture. Each "Agent" is a self-contained unit responsible for a specific task in the video processing workflow. This design promotes:
- **Modularity:** Agents can be developed, tested, and maintained independently.
- **Extensibility:** New agents or analysis techniques can be easily integrated into the pipeline.
- **Reusability:** Agents can be reordered or reused in different workflows.
- **Resilience:** Failure in one agent can be isolated, and the pipeline can potentially resume from that point.

The `main.py` orchestrates the execution of these agents via an `AgentManager`, passing a shared `context` dictionary that accumulates data throughout the pipeline. The `config.yaml` explicitly defines the `enabled_agents` and their execution order, as well as `parallel_groups` for performance optimization.

## Key Technical Decisions:
- **Agent-Based Design:** Central to the architecture, allowing for clear separation of concerns and a sequential processing flow.
- **Shared Context Dictionary:** All data and intermediate results are passed between agents via a mutable `context` dictionary. This acts as the central data store for the current processing run.
- **State Management (`StateManager`):** Persists the `context` to disk, enabling resume functionality and preventing loss of progress in case of interruptions.
- **Caching (`CacheManager`):** Utilizes disk-based caching for computationally intensive steps (e.g., frame extraction, multimodal analysis) to speed up subsequent runs with the same input.
- **Resource Management (`ResourceManager`):** Manages the loading and unloading of large models (e.g., LLMs, PyTorch models) to optimize GPU/CPU memory usage, especially important for resource-constrained environments.
- **LLM Integration:** Extensive use of Large Language Models (LLMs) for intelligent tasks like clip selection, storyboarding, content alignment, hook identification, and viral potential assessment. A robust JSON extraction mechanism with retry and self-correction is implemented for reliable LLM interaction.
- **Multimodal Analysis:** Combines audio and visual analysis techniques to provide a holistic understanding of the video content.
- **Pydantic for Data Validation:** Ensures strict schema validation for LLM outputs and internal data structures, improving data integrity and reducing errors.
- **FFmpeg for Video Operations:** Leverages FFmpeg for robust video and audio processing tasks (extraction, cutting, merging, adding effects).

## Design Patterns in Use:
- **Pipeline Pattern:** The entire system operates as a sequential pipeline of agents.
- **Chain of Responsibility (Implicit):** Agents process the context and pass it to the next agent in the chain.
- **Singleton (Implicit):** `StateManager`, `CacheManager`, `ResourceManager`, and `Monitor` appear to be managed as singletons or effectively singletons within the application's lifecycle, ensuring a single point of control for their respective functionalities.
- **Strategy Pattern:** Different LLM models or analysis techniques can be swapped out (e.g., `llm_selection_agent` can use different LLM models based on configuration).
- **Observer Pattern (Implicit):** The `Monitor` class acts as an observer, collecting metrics throughout the pipeline.

## Component Relationships:
- `main.py`: Orchestrates the pipeline, initializes `StateManager`, `AgentManager`, and `Monitor`.
- `core/config.py`: Centralized configuration management, including agent enablement and parallelization.
- `core/state_manager.py`: Manages persistent state (`context`).
- `core/cache_manager.py`: Manages disk-based caching.
- `core/resource_manager.py`: Manages model loading/unloading.
- `core/monitoring.py`: Collects performance metrics.
- `core/utils.py`: Provides utility functions (e.g., system checks, video info).
- `llm/`: Contains modules for LLM interaction, image analysis, and prompt utilities.
    - `llm_interaction.py`: Handles communication with LLMs, including robust JSON extraction and model switching.
    - `image_analysis.py`: Specific LLM calls for image description and analysis.
    - `prompt_utils.py`: Tools for building adaptive prompts within token limits.
- `agents/`: Directory containing all individual agent implementations.
    - `base_agent.py`: Abstract base class for all agents, defining the `execute` method.
    - Each specific agent (e.g., `VideoInputAgent`, `StoryboardingAgent`, `LLMSelectionAgent`, `VideoProductionAgent`) implements its unique logic, operating on the shared `context`.
- `video/`: Contains modules for video processing tasks (e.g., `frame_processor.py`, `scene_detection.py`, `video_editing.py`).
- `audio/`: Contains modules for audio processing tasks (e.g., `audio_processing.py`, `subtitle_generation.py`, `voice_cloning.py`).

## Critical Implementation Paths:
The `config.yaml` defines the `enabled_agents` and their order, which dictates the primary flow. Parallel execution groups are also defined for performance.
1. **Video Ingestion & Preprocessing:** `VideoInputAgent` -> `FramePreprocessingAgent` (can run in parallel with `AudioIntelligenceAgent`).
2. **Audio Analysis:** `AudioIntelligenceAgent` (can run in parallel with `FramePreprocessingAgent`).
3. **Core Analysis & Storyboarding:** `MultimodalAnalysisAgent` (can run in parallel with `StoryboardingAgent`) -> `LayoutSpeakerAgent`.
4. **Intelligent Clip Selection & Content Direction:** `LLMSelectionAgent` (can run in parallel with `ContentDirectorAgent`).
5. **Post-Selection Analysis:** `HookIdentificationAgent` -> `ViralPotentialAgent` (can run in parallel with `DynamicEditingAgent` and `MusicSyncAgent`).
6. **Dynamic Editing & Music Sync:** `DynamicEditingAgent` (can run in parallel with `ViralPotentialAgent` and `MusicSyncAgent`) -> `MusicSyncAgent` (can run in parallel with `ViralPotentialAgent` and `DynamicEditingAgent`).
7. **Layout Optimization & Subtitles:** `LayoutOptimizationAgent` -> `SubtitleAnimationAgent`.
8. **Final Production & Summary:** `VideoProductionAgent` -> `ResultsSummaryAgent`.
