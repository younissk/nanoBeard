"""Rejection-sampling support: prompt set + N-samples-per-prompt generation.

Generation targets a llama.cpp GGUF build via `llama-server`'s raw
`/completion` endpoint (NOT `/chat`) — the GGUF carries no chat template, and
the SFT format is a plain transcript defined in `nanobeard.sft_data`.
"""
