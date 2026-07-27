# Run Orpheus 3B on your E2E L40S — manual steps

You'll copy 2 files to the L40S, run one script, and pull the audio back.
Everything below is copy-paste. Placeholders: `<USER>` (e.g. `root`/`ubuntu`),
`<IP>`, `<PORT>` (drop `-P/-p <PORT>` if it's the default 22).

---

## 0. (Recommended) Accept the Hindi model gate — 1 click
Log in to HF as **Shxbhxm21** and click **"Agree and access repository"** on:
- https://huggingface.co/canopylabs/3b-hi-ft-research_release

This is the same one-click auto-gate you already did for the base model. It unlocks
the official Hindi voice **ऋतिका**. *(If you skip this, open `infer.py` and switch the
Hindi job to the ungated community model — see the commented lines in `JOBS`.)*

---

## 1. Clone + set up + run  (SSH into the L40S, then run there)
```bash
ssh -p <PORT> <USER>@<IP>          # connect however you normally do

git clone https://github.com/SHxBHAM/orpheus-hinglish-test.git
cd orpheus-hinglish-test

# Isolated venv (Debian blocks system-wide pip: PEP 668). Do NOT use
# --system-site-packages: if the box's system torch was built for a different CUDA
# than the driver, it silently falls back to CPU or throws libcupti symbol errors.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install a torch that matches YOUR driver's CUDA (see `nvidia-smi`, top-right):
#   CUDA 12.8 driver -> cu128   |   12.4 -> cu124   |   12.1 -> cu121
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('cuda', torch.cuda.is_available())"   # MUST print: cuda True

pip install transformers accelerate snac numpy

# HF token — needed to download the gated base model.
# Copy the value from your Mac's  NewerTTS/.env  (the HF_TOKEN=... line):
export HF_TOKEN='hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# go
python infer.py
```
Expected: it downloads SNAC + the models (first run only), then prints one line per
clip, e.g. `base__emo_02: 7.2s in 12.7s (RTF 1.77) rms=0.048`. On an L40S expect
RTF well under the 4090's ~1.8. Output lands in `outputs/` (WAVs + `index.html` +
`results.json`).

## 2. Listen to the audio
- **On the L40S** (if a port is reachable): `cd outputs && python -m http.server 8000`
  then open `http://<IP>:8000/index.html`.
- **Or pull it to your Mac:**
  ```bash
  scp -P <PORT> -r <USER>@<IP>:~/orpheus-hinglish-test/outputs ./outputs && open outputs/index.html
  ```

---

## 3. (New) Add Veena — the native-Hinglish reference
`maya-research/Veena` is a Llama+SNAC TTS **built for Hindi/English code-mixing** —
the fallback if Orpheus's code-mix stays weak. It's **ungated** (no HF gate to click)
and runs in **bf16 (~7GB)** on the L40S — no bitsandbytes/4-bit needed. Same venv:
```bash
# in the same activated .venv, after infer.py has run:
python infer_veena.py
```
It renders the SAME sentences (voices `kavya` + `agastya` by default; edit `VOICES`
at the top to add `maitri`/`vinaya`), writes `veena_*.wav` into `outputs/`, and
builds **`outputs/compare.html`** — every model's take on each sentence stacked for a
direct A/B. Pull it the same way:
```bash
scp -P <PORT> -r <USER>@<IP>:~/orpheus-hinglish-test/outputs ./outputs && open outputs/compare.html
```

## What you're testing
- **base** (`canopylabs/orpheus-3b-0.1-ft`, voice `tara`): English + emotion tags
  (`<sigh> <laugh> <yawn>`) — Orpheus's expressiveness ceiling.
- **hindi_official** (`canopylabs/3b-hi-ft-research_release`, voice `ऋतिका`):
  the Hinglish/Devanagari lecture lines — the Orpheus code-mix result (came out bad).
- **veena_kavya / veena_agastya** (`maya-research/Veena`): the same lecture lines from
  a natively-Hinglish model — the code-mix quality ceiling to compare against.

## Troubleshooting
- **`error: externally-managed-environment`** (Debian/PEP 668) → you skipped the venv.
  Run the `python3 -m venv --system-site-packages .venv && source .venv/bin/activate`
  lines above first, or append `--break-system-packages` to the `pip install`.
- **`device=cpu` / `cuda=False` / "NVIDIA driver too old"** OR
  **`ImportError: ... libtorch_cpu.so: undefined symbol ... libcupti.so.12`** → the
  torch in the venv doesn't match the box's driver/CUDA libs (usually from reusing a
  `--system-site-packages` torch). Fix = clean isolated venv + driver-matched torch:
  `rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (swap `cu128` for your driver's CUDA), then re-check
  `python -c "import torch; print(torch.cuda.is_available())"` before reinstalling the rest.
- **`401 / gated repo`** on the base model → `HF_TOKEN` not exported, or wrong account.
- **`401 / gated`** on the Hindi model → you skipped step 0; either do it, or switch
  `JOBS` in `infer.py` to a commented ungated community model.
- **Hindi voice reads a name aloud / sounds off** → the voice token differs for that
  checkpoint. In `infer.py` set the Hindi job's voice to `None` (raw text) and rerun.
- **`NO AUDIO` / silent clips** → lower `temperature` to 0.4 and raise
  `repetition_penalty` to 1.15 (top of `infer.py` / the `generate()` call).
- **Slow** → it's plain HF `generate` (no vLLM). Fine for this ~12-clip test; if you
  productionize, switch to vLLM/`orpheus-speech` for ~5-10x throughput.
