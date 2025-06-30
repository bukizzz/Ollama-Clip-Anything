### FAILURE RECOVERY AND RESUMPTION LOGIC

## PIPELINE DEFINITION

# Define Pipeline Stages

- Identify and document all discrete stages in the clip generation pipeline:
	- Video downloading (if input is URL)
	- Transcription
	- Segmentation (LLM-based clip identification)
	- Clip extraction (video cutting)
	- Subtitle generation (future stage)
	- VFX application (future stage)
	- B-roll integration (future stage)
	- Final clip rendering

## STATE MANAGEMENT

# State Tracking System

- State File: `.temp_clips.json` (JSON format)

- Data to Track:
	- input_source: Path/URL of original video
	- current_stage: Last completed stage (e.g., "transcription")
	- completed_segments: List of already-processed clip timecodes
	- temp_files: Critical intermediate files (e.g., `transcript.json`, `segments.json`)
	- failure_point: Stage where failure occurred
	- error_log: Relevant error messages
	- segment_queue: Pending segments for processing

# Checkpoint Implementation

- After each pipeline stage:
	- Validate stage output
	- Update `.temp_clips.json`:
		- Set `current_stage` to completed stage
		- Record output file paths in `temp_files`
		- Clear `failure_point` and `error_log`
	- Persist file immediately after update

- Segment-level tracking:
	- For clip extraction stage:
		- Track individual segment completion in `completed_segments`
		- Maintain pending segments in `segment_queue`

## FAILURE CONTROL

# Failure Handling

- Global exception handler:
	- Catch all exceptions
	- Write to state file:
		- `failure_point`: Current stage name
		- `error_log`: Exception details
	- Preserve temporary files listed in `temp_files`

- Mid-segment failures:
	- Track last processed segment
	- Ensure no mid-sentence cuts by:
		- Validating segment boundaries against transcript
		- Rolling back to previous valid endpoint

## RECOVERY EXECUTION

# Resume Mechanism

- Startup workflow:

IF .temp_clips.json EXISTS:
IF --nocache flag:
Delete state + temp files → Fresh start
ELIF --retry flag:
Resume automatically
ELSE:
Prompt: "Previous session failed. Resume? [y/n]"
YES → Resume
NO → Delete state + temp files → Fresh start
ELSE:
Fresh start


- Resumption process:
	- Load `.temp_clips.json`
	- Verify existence of `temp_files`
	- Skip stages before `current_stage`
	- For clip extraction:
		- Process only segments in `segment_queue`
		- Omit `completed_segments`

## INTERFACE DESIGN

# CLI Arguments

- Add to argument parser:
	- `--retry`:
		- Auto-resume if state exists
		- Log warning if no state found
	- `--nocache`:
		- Force-delete `.temp_clips.json` + associated temp files
		- Disables resume functionality

## TEMPORARY DATA CONTROL

# Temporary File Management

- Designate temp directory: `./temp_processing/`

- File tracking:
	- Stage outputs saved to temp directory
	- Paths recorded in `temp_files` (relative paths)

- Cleanup protocol:
	- Successful completion → Delete all temp files + state
	- User declines resume → Delete all temp files + state
	- `--nocache` → Preemptive deletion

## INTEGRITY PROTECTION

# Validation Safeguards

- State file integrity:
	- Schema validation on load
	- Checksum verification for critical files

- Segment validation:
	- Ensure segments align with transcript
	- Prevent mid-word cuts by:
		- Adjusting boundaries to nearest punctuation
		- Flagging segments crossing sentence boundaries

- Cross-stage checks:
	- Verify temp files match pipeline stage requirements
	- Re-run prerequisite stages if critical files corrupted

## EXECUTION ROADMAP

# Implementation Sequence

- Add state file management utilities:
	- `create_state_file()`
	- `load_state_file()`
	- `update_state_file()`
	- `delete_state_file()`

- Instrument pipeline stages with checkpointing

- Implement global exception handler

- Add CLI argument parsing

- Build resume workflow controller

- Integrate temp file validation

- Add boundary validation for segments

## LONG-TERM MAINTAINABILITY

# Future-Proofing

- State file versioning: Include `version` field

- Modular stage design: Ensure new stages can:
	- Report completion status
	- Accept inputs from previous state
	- Output verifiable results

- Error code system: Standardize failure reasons for:
	- Automatic recovery attempts
	- User-friendly error reporting

## SECURITY ENFORCEMENT

# Security Considerations

- State file sanitization:
	- Avoid storing sensitive data
	- Validate paths before deletion

- User prompts:
	- Explicit confirmation for destructive actions
	- Clear warnings about file deletions

## QUALITY ASSURANCE

# Testing Strategy

- Simulated failures:
	- Inject failures at each stage
	- Verify correct state persistence

- Resume validation:
	- Confirm skipped stages don't reprocess
	- Verify segment queue handling

- Edge cases:
	- Corrupted state files
	- Missing temp files during resume
	- Concurrent execution prevention

## DESIGN PRINCIPLES

# Key Principles

- Atomic Operations: Stages must fully succeed or fail completely

- Idempotence: Resuming shouldn't produce duplicate outputs

- User Control: Always prioritize CLI flags over automated decisions

- Validation First: Never resume without verifying prerequisites