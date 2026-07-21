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

# venv that REUSES the system torch/CUDA (Debian blocks system-wide pip: PEP 668).
# --system-site-packages = no giant torch reinstall, guaranteed CUDA match.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# HF token — needed to download the gated base model.
# Copy the value from your Mac's  NewerTTS/.env  (the HF_TOKEN=... line):
export HF_TOKEN='hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# deps — do NOT add numpy/torch here (reuse the system ones; upgrading numpy can
# break the system torch's pin).
pip install transformers accelerate snac

# sanity check — must print 'cuda True' before continuing
python -c "import torch,transformers,snac; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

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

## What you're testing
- **base** (`canopylabs/orpheus-3b-0.1-ft`, voice `tara`): English + emotion tags
  (`<sigh> <laugh> <yawn>`) — Orpheus's expressiveness ceiling.
- **hindi_official** (`canopylabs/3b-hi-ft-research_release`, voice `ऋतिका`):
  the Hinglish/Devanagari lecture lines — the real comparison vs VoxCPM2.

## Troubleshooting
- **`error: externally-managed-environment`** (Debian/PEP 668) → you skipped the venv.
  Run the `python3 -m venv --system-site-packages .venv && source .venv/bin/activate`
  lines above first, or append `--break-system-packages` to the `pip install`.
- **`401 / gated repo`** on the base model → `HF_TOKEN` not exported, or wrong account.
- **`401 / gated`** on the Hindi model → you skipped step 0; either do it, or switch
  `JOBS` in `infer.py` to a commented ungated community model.
- **Hindi voice reads a name aloud / sounds off** → the voice token differs for that
  checkpoint. In `infer.py` set the Hindi job's voice to `None` (raw text) and rerun.
- **`NO AUDIO` / silent clips** → lower `temperature` to 0.4 and raise
  `repetition_penalty` to 1.15 (top of `infer.py` / the `generate()` call).
- **Slow** → it's plain HF `generate` (no vLLM). Fine for this ~12-clip test; if you
  productionize, switch to vLLM/`orpheus-speech` for ~5-10x throughput.
