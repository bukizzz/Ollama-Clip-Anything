# USER INPUTTED TASK
This document is a structured prompt for GEMINI that has full access to the project files and implementation. You are tasked with inspecting and documenting the structure and functionality of the current codebase. Your output will serve as foundational context for integrating **ImageBind** and restructuring the system into a **single-model-at-a-time intertwined MoE pipeline**, constrained by limited GPU memory.

You are not being asked to implement anything. Your role is purely to provide information about the current state of the codebase. Do not speculate about future changes. All responses must be derived from the actual content of the code.

To start create a ANSWERS.md file and use it to provide answers. Format:

# Number of stage - Stage Title
- Question
- Answer

- Question
- Answer

Below you will find 17 stages you need to complete. You are to read one stage at a time, assess its requirements, analyze its questions, read the code and answer the questions for that stage only! Once all of the conditions are satisfied and all questions are answered in detail with no ambiguity, you can progress to the next stage.

Answer each question point with as much specific detail as possible, referencing filenames, function names, class names, and line numbers where applicable. Use fenced code blocks (` ```python `) for all code excerpts.

---

## 1. GENERAL PROJECT OVERVIEW

* What is the overall purpose of this project?
* What are the major functional components or modules?
* What is the high-level execution flow?
* What are the main entry points or scripts used to launch the pipeline?
* Provide a directory and subdirectory hierarchy tree with brief descriptions of the function of each folder and major file.

---

## 2. MODEL USAGE AND ARCHITECTURE

* Which LLMs or vision models are currently used in the pipeline?
* In what format are they loaded (e.g., GGUF, transformers, safetensors, etc)?
* Where are they defined, loaded, and invoked?
* Are any models run concurrently? If so, which ones?
* What are the memory usage patterns (VRAM vs CPU) of these models?
* Is any form of offloading or model unloading implemented currently?
* Is there any inference caching or checkpointing logic?

---

## 3. DATA FLOW AND CONTEXT HANDLING

* Describe how context is passed between different components or stages.
* Is the context stored in memory, written to file, or streamed?
* What data structures or formats (JSON, dict, pickle, etc) are used for context transfer?
* Where in the code does this context management occur?

---

## 4. CURRENT PIPELINE ORDER

* What is the exact sequence of execution steps?
* Which agents or models are called, in what order?
* Are there conditional branches or dynamic decisions in the execution order?

---

## 5. INTERFACES BETWEEN STAGES

* How are outputs from one stage consumed by the next?
* Are these interfaces modular and reusable?
* Do any agents share memory or operate in separate processes?

---

## 6. ASYNCHRONY AND PARALLELISM

* Is there any use of asynchronous execution or multiprocessing?
* Are any models run in background threads or subprocesses?
* Are there GPU blocking or queuing mechanisms implemented?

---

## 7. LOGGING AND METRICS

* Is logging implemented?
* Where and how is execution progress recorded?
* Are GPU/CPU usage or performance metrics tracked?

---

## 8. IMAGE AND VIDEO MODULES

* What vision/image/video modules are implemented?
* Is there any preprocessing pipeline for images/videos (resizing, encoding, batching)?
* Where are vision models called, and how are inputs formatted?
* Are there any embeddings generated from images currently?

---

## 9. EXISTING MULTIMODAL LOGIC

* Is there any current usage of multimodal fusion (text + image, etc)?
* If so, how is it handled and where is it implemented?
* Are embeddings from different modalities combined or compared?

---

## 10. DEPENDENCIES AND COMPATIBILITY

* List all pinned versions of major dependencies (torch, torchvision, transformers, etc).
* What framework constraints (e.g., PyTorch version) are critical and why?
* Is any dependency known to conflict with PyTorch 2.3.0 or ImageBind?

---

## 11. RESOURCE MANAGEMENT AND CONSTRAINTS

* What measures are in place to manage memory usage?
* Are models explicitly deleted or moved between devices?
* Is torch.cuda.empty\_cache() or gc.collect() used at any point?

---

## 12. EXISTING EXTENSION POINTS

* Are there plugin hooks, abstract base classes, or dynamic dispatch patterns that allow easy model replacement?
* If yes, where are they defined and how are they used?

---

## 13. INTEGRATION BARRIERS

* Which parts of the codebase are most tightly coupled to specific models?
* Are there hard-coded model assumptions that would block swapping in ImageBind or any other MoE-style agent?

---

## 14. EXECUTION CONTROL AND CONFIGS

* Are there config files or CLI arguments that control pipeline execution?
* Where are these parsed and used?
* Is there any scheduling logic that determines when or how agents/models are activated?

---

## 15. TEMPORARY STORAGE AND FILE I/O

* Are intermediate results written to disk or stored in memory?
* What temporary files or cache directories are created during execution?

---

## 16. FAILURES AND ERROR HANDLING

* How does the system handle inference errors or GPU memory overflows?
* Are failed stages retried, skipped, or do they abort the whole pipeline?

---

## 17. CODE STRUCTURE QUALITY

* Is the project modular and readable?
* Are functions/classes logically separated or is code monolithic?
* Any known anti-patterns, poor abstractions, or technical debt?

---

## OUTPUT FORMAT

Answer each question point with as much specific detail as possible, referencing filenames, function names, class names, and line numbers where applicable. Use fenced code blocks (` ```python `) for all code excerpts.

Your response will be used to plan an intertwined single-model-at-a-time MoE system with optional ImageBind integration. Accuracy and completeness are critical.

