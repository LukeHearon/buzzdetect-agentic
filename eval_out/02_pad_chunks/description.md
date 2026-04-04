Zero-pad chunks in the streamer so every chunk (including the last of each file) is exactly `chunklength_samples` long. This ensures all chunks hit the precompiled XLA kernel size rather than falling back to the slower non-XLA path.

`frames_complete` is computed from the pre-padding sample count and stored on `AssignChunk`. The writer slices `results[:frames_complete]` before formatting, so padded output is never written.
