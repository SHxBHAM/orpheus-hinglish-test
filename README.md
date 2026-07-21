# Orpheus 3B — Hinglish listening test

Goal: hear whether **Orpheus 3B** beats **VoxCPM2** on expressiveness while handling
Hinglish, before committing to a fine-tune. This renders PW-style lecture lines
(Hinglish + English + emotion tags), reports latency, and builds an HTML player.

> **Quick start on a GPU box (E2E L40S etc.):** follow **[E2E_STEPS.md](E2E_STEPS.md)** —
> clone this repo, `export HF_TOKEN=...`, `pip install -U transformers accelerate snac numpy`,
> `python infer.py`. `infer.py` is the validated transformers+SNAC path (no vLLM).
> The rest of this file documents the alternative vLLM/`orpheus-speech` runner (`run_orpheus.py`).

Orpheus is a Llama-3B speech-LLM → SNAC codec, Apache 2.0, ~8 GB VRAM, inline emotion
tags (`<laugh> <sigh> <chuckle> <yawn> ...`), 24 kHz. Same architecture family as
**Veena**, and your existing fine-tune/LoRA workflow transfers directly.

## What's here
- `test_sentences.json` — 12 curated lines: 7 Hinglish (lecture/physics/bio), 2 English, 3 emotion-tag.
- `run_orpheus.py` — renders them, writes 24 kHz WAVs to `outputs/`, reports RTF, builds `outputs/index.html`.
- `setup_runpod.sh` — one-time env setup on a GPU pod.
- `requirements.txt`.

## Why RunPod (not the Mac)
A 3B model needs a GPU (~8 GB+ VRAM). Your Mac has no CUDA, so run this on a RunPod
pod — an **RTX 4090** or **A100-40GB** is plenty. (Hosted APIs like Baseten/Replicate
serve only the *English base* Orpheus, so for Hinglish you must self-host the fine-tune.)

## Run it

1. Spin up a RunPod **GPU pod** (PyTorch 2.x / CUDA 12.1 image, 4090 or A100).
2. Upload this `orpheus_test/` folder (or `git clone` your repo).
3. Set up + render:
   ```bash
   cd orpheus_test
   bash setup_runpod.sh
   source .venv/bin/activate

   # Hinglish on the Hindi fine-tune (the main event):
   python run_orpheus.py --model SachinTelecmi/Orpheus-tts-hi --subset hinglish --voice tara

   # English + emotion tags on the base fine-tune (expressiveness ceiling):
   python run_orpheus.py --model canopylabs/orpheus-3b-0.1-ft --subset emotion,english --voice tara \
       --out outputs_base
   ```
4. Download `outputs/` (and `outputs_base/`) and open `index.html` — one `<audio>`
   player per line, with gen time / RTF. A/B each against your VoxCPM2 render of the
   same sentence.

## Knobs
- `--voice` base voices: `tara leah jess leo dan mia zac zoe` (`tara` is usually best).
  For the single-speaker Hindi fine-tune, start with `tara`; if timbre sounds off,
  check the model card / discussions for the trained speaker name and pass that.
- `--subset hinglish,english,emotion` (or `all`) to pick categories.
- `--temperature 0.4 --repetition-penalty 1.1` — Orpheus **requires** rep-penalty ≥ 1.1
  for stable output; nudge temperature up for livelier (faster) delivery.

## Candidate Hinglish checkpoints to compare
- `SachinTelecmi/Orpheus-tts-hi` — Hindi/Hinglish + Devanagari, Apache 2.0 (default here).
- `canopylabs/orpheus-3b-0.1-ft` — English base; best for the emotion-tag stress test.
- `lex-au/Orpheus-3b-Hindi-FT` — GGUF (llama.cpp/LM Studio runtime, not this vLLM script).

## If you like what you hear → next step: fine-tune your PW voices
You already have the corpus + LoRA pipeline. Orpheus fine-tunes the same way you did
VoxCPM2/IndicF5:
1. Format your Hinglish clips as `(text, audio)` pairs; encode audio to SNAC tokens.
2. LoRA or full-FT the Llama-3B backbone on the paired token sequences (single A100).
3. Register each PW voice under its own speaker name (e.g. `smoll`, `priya`) so
   `--voice smoll` selects it — mirrors your VoxCPM voice-per-folder setup.
Canopy Labs publishes a fine-tune notebook/recipe in the official repo; point it at
your dataset and reuse your existing training infra.

## Troubleshooting
- **OOM on 4090:** lower `--max-model-len` (e.g. 1024) or use the A100.
- **`vllm` import/version errors:** the version is pinned to `0.7.3` in a clean venv on
  purpose — don't let the base image's vLLM leak in; always `source .venv/bin/activate`.
- **401 pulling weights:** `huggingface-cli login` (SNAC + some checkpoints are gated).
- **Robotic/looping audio:** raise `--repetition-penalty` to 1.15 and lower `--temperature`.
