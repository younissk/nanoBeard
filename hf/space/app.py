"""Gradio chat playground for nanoBeard. Multi-version aware via MODEL_REGISTRY."""

import json
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tokenizers import Tokenizer

from nanobeard.config import Config
from nanobeard.models import MODEL_REGISTRY, build_model

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 20

# Pick which HF model repo to serve. Comma-separated list shows dropdown.
DEFAULT_REPOS = ",".join(s.hf_repo for s in MODEL_REGISTRY.values())
REPO_LIST = [
    r.strip() for r in os.environ.get("NANOBEARD_REPOS", DEFAULT_REPOS).split(",") if r.strip()
]


def _load(repo: str):
    with open(hf_hub_download(repo, "config.json")) as f:
        cfg_dict = json.load(f)
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    model = build_model(cfg).eval()
    load_model(model, hf_hub_download(repo, "model.safetensors"))
    tok = Tokenizer.from_file(hf_hub_download(repo, "pirate_bpe.json"))
    eos = tok.token_to_id("<|endoftext|>")
    display = cfg_dict.get("display_name", f"nanoBeard {cfg_dict.get('codename', '?')}")
    return cfg, model, tok, eos, display


# Eager-load all configured repos so swapping is instant.
LOADED = {repo: _load(repo) for repo in REPO_LIST}
DEFAULT_REPO = REPO_LIST[0]


@torch.no_grad()
def respond(message: str, history, repo_choice: str):
    cfg, model, tok, eos, _ = LOADED[repo_choice]
    prompt = PROMPT_TEMPLATE.format(instruction=message.strip())
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
    prompt_len = ids.size(1)

    for _ in range(MAX_NEW_TOKENS):
        logits, _ = model(ids[:, -cfg.block_size :])
        logits = logits[:, -1] / TEMPERATURE
        v, _ = torch.topk(logits, min(TOP_K, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

        if eos is not None and next_id.item() == eos:
            break

        yield tok.decode(ids[0, prompt_len:].tolist())

    yield tok.decode(ids[0, prompt_len:].tolist())


pirate_theme = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="red",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Pixelify Sans"), "monospace"],
    font_mono=[gr.themes.GoogleFont("Pixelify Sans"), "monospace"],
).set(
    body_background_fill="#1a0f08",
    body_background_fill_dark="#1a0f08",
    background_fill_primary="#2b1810",
    background_fill_secondary="#3d2418",
    block_background_fill="#2b1810",
    block_border_color="#d4a017",
    block_border_width="0px",
    block_label_text_color="#f4c430",
    block_title_text_color="#f4c430",
    body_text_color="#f4e4bc",
    button_primary_background_fill="#d4a017",
    button_primary_text_color="#1a0f08",
    button_primary_background_fill_hover="#f4c430",
    input_background_fill="#1a0f08",
    input_background_fill_focus="#1a0f08",
    input_border_color="#d4a017",
    input_text_size="*text_lg",
)


PIXEL_CSS = """
html, body, gradio-app, .gradio-container, .main, .contain {
    height: 100vh !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.gradio-container { font-size: 18px !important; }
footer { display: none !important; }

[data-testid="chatbot"] *,
[data-testid="chatbot"] *::before,
[data-testid="chatbot"] *::after,
.chatbot *, [class*="chatbot"] * {
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #f4e4bc !important;
}
[class*="user-message"], [class*="user-bubble"], [class*="user"] [class*="message"] {
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="chatbot"] *::before,
[data-testid="chatbot"] *::after { display: none !important; }

[data-testid="chatbot"] {
    border: 2px solid #d4a017 !important;
    background: #2b1810 !important;
}

[data-testid="chatbot"] img {
    background: #ffffff !important;
    border: 2px solid #d4a017 !important;
    border-radius: 0 !important;
    image-rendering: pixelated;
    display: inline-block !important;
}

textarea, input[type="text"] {
    color: #f4e4bc !important;
    background: #1a0f08 !important;
    caret-color: #f4c430 !important;
    border: 2px solid #d4a017 !important;
    border-radius: 0 !important;
}
textarea::placeholder, input::placeholder {
    color: #d4a017 !important;
    opacity: 1 !important;
}

.examples button, [class*="example"] button {
    color: #f4e4bc !important;
    background: #1a0f08 !important;
    border: 2px solid #d4a017 !important;
    border-radius: 0 !important;
}
.examples button *, [class*="example"] button * {
    border: 0 !important;
    background: transparent !important;
    color: #f4e4bc !important;
}

button { border-radius: 0 !important; }
h1, h2, h3 { color: #f4c430 !important; text-shadow: 2px 2px 0 #000; }
"""

chatbot = gr.Chatbot(
    label="nanoBeard",
    avatar_images=("assets/nanoBeard.png", "assets/nanoBeard.png"),
    show_label=False,
    height="80vh",
)

textbox = gr.Textbox(
    placeholder="Speak yer mind, sailor…",
    label="Sailor",
    container=True,
    scale=1,
)

model_dropdown = gr.Dropdown(
    choices=[(LOADED[r][4], r) for r in REPO_LIST],
    value=DEFAULT_REPO,
    label="Model",
    visible=len(REPO_LIST) > 1,
)

with gr.Blocks(title="nanoBeard") as demo:
    gr.ChatInterface(
        respond,
        chatbot=chatbot,
        textbox=textbox,
        additional_inputs=[model_dropdown],
        examples=[
            ["Tell me a tale of buried treasure."],
            ["What does a pirate eat for breakfast?"],
            ["Write a short pirate poem."],
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(theme=pirate_theme, css=PIXEL_CSS)
