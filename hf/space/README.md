---
title: NanoBeard Playground
emoji: ☠️
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
short_description: A small playground to play around with the nanoBeard LLM
---

# nanoBeard playground

Chat playground for [younissk/nanoBeard](https://huggingface.co/younissk/nanoBeard) —
a tiny pirate-themed GPT (~14M params) trained on piratized TinyStories and
SFT-tuned on a pirate version of Dolly-15k.

The model wasn't trained for multi-turn dialogue; each message is sent as a
fresh `### Instruction:` prompt. Outputs are pirate-flavored nonsense.

Source: https://github.com/younissk/pirate_llm
