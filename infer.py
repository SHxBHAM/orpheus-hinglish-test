#!/usr/bin/env python3
"""
Orpheus 3B inference via transformers + SNAC (no vLLM). Validated working.
Renders test_sentences.json through 1+ Orpheus checkpoints -> 24 kHz WAVs in
./outputs, plus results.json and an index.html player.

Run on a GPU box (your E2E L40S):
    export HF_TOKEN=hf_...        # needed for the gated base model
    pip install -U transformers accelerate snac numpy    # torch already present
    python infer.py
"""
import json, time, wave, traceback
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
SR = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1800

# Orpheus special token ids (shared base vocab across base + fine-tunes)
SOH, EOT, EOH = 128259, 128009, 128260      # start-of-human, end-of-text, end-of-human
SOS, EOS = 128257, 128258                    # start / end of speech
AUDIO_BASE = 128266
CODEBOOK = 4096
AUDIO_MAX = AUDIO_BASE + 7 * CODEBOOK

# ============================================================================
# JOBS: (model_id, [categories], voice_or_None, tag)
#   categories pick sentences from test_sentences.json: hinglish|english|emotion
# ----------------------------------------------------------------------------
JOBS = [
    # Base English model (expressive + emotion tags). Auto-gate already accepted.
    ("canopylabs/orpheus-3b-0.1-ft", ["emotion", "english"], "tara", "base"),

    # --- Hinglish half: RECOMMENDED official model (voice = ऋतिका). ---
    # Requires a 1-click "Agree and access" at:
    #   https://huggingface.co/canopylabs/3b-hi-ft-research_release   (same as base)
    ("canopylabs/3b-hi-ft-research_release", ["hinglish", "english"], "ऋतिका", "hindi_official"),

    # --- OR ungated community model (no gate; voice undocumented). ---
    # If you skip the official gate, comment the line above and uncomment ONE below.
    # ("Shekharmeena/Orpheus_TTS_Hindi_MultiSpeaker", ["hinglish", "english"], None, "hindi_community"),
    # ("Aaryan39/orpheus-3b-hindi-ft-merged-4voices", ["hinglish", "english"], None, "hindi_community"),
]
# ============================================================================

print(f"device={DEVICE} torch={torch.__version__} cuda={torch.cuda.is_available()}")
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

sentences = json.loads((HERE / "test_sentences.json").read_text(encoding="utf-8"))
print(f"loaded {len(sentences)} sentences")

print("loading SNAC codec (hubertsiuzdak/snac_24khz)...")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(DEVICE)


def build_prompt_ids(tok, text, voice):
    p = text if voice is None else f"{voice}: {text}"
    ids = tok(p, return_tensors="pt").input_ids
    start = torch.tensor([[SOH]], dtype=torch.int64)
    end = torch.tensor([[EOT, EOH]], dtype=torch.int64)
    return torch.cat([start, ids, end], dim=1)


def decode_audio(generated_ids):
    row = generated_ids
    sos_pos = (row == SOS).nonzero(as_tuple=True)[0]
    if len(sos_pos) > 0:
        row = row[sos_pos[-1].item() + 1:]
    row = row[(row >= AUDIO_BASE) & (row < AUDIO_MAX)]
    codes = (row - AUDIO_BASE).tolist()
    n = (len(codes) // 7) * 7
    codes = codes[:n]
    if n == 0:
        return None
    l1, l2, l3 = [], [], []
    for i in range(n // 7):
        b = 7 * i
        l1.append(codes[b])
        l2.append(codes[b + 1] - CODEBOOK)
        l3.append(codes[b + 2] - 2 * CODEBOOK)
        l3.append(codes[b + 3] - 3 * CODEBOOK)
        l2.append(codes[b + 4] - 4 * CODEBOOK)
        l3.append(codes[b + 5] - 5 * CODEBOOK)
        l3.append(codes[b + 6] - 6 * CODEBOOK)
    layers = [torch.tensor(l1).unsqueeze(0).to(DEVICE),
              torch.tensor(l2).unsqueeze(0).to(DEVICE),
              torch.tensor(l3).unsqueeze(0).to(DEVICE)]
    for L in layers:
        if L.numel() == 0 or L.min() < 0 or L.max() >= CODEBOOK:
            return None
    with torch.inference_mode():
        audio = snac.decode(layers)
    return audio.squeeze().detach().cpu().float().numpy()


def write_wav(path, wav):
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


results = []
for model_id, cats, voice, tag in JOBS:
    todo = [s for s in sentences if s["category"] in cats]
    print(f"\n=== {tag}: {model_id}  ({len(todo)} sentences, voice={voice}) ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16).to(DEVICE).eval()
    except Exception as e:
        print(f"!! could not load {model_id}: {e}")
        for s in todo:
            results.append({**s, "tag": tag, "model": model_id, "ok": False,
                            "error": f"model load failed: {e}", "wav": None})
        continue

    for i, s in enumerate(todo, 1):
        rid = f"{tag}__{s['id']}"; wav_name = f"{rid}.wav"
        rec = {**s, "tag": tag, "model": model_id, "wav": wav_name, "ok": False}
        try:
            ids = build_prompt_ids(tok, s["text"], voice).to(DEVICE)
            attn = torch.ones_like(ids)
            t0 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=attn,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                    eos_token_id=EOS, pad_token_id=EOS)
            gen_s = time.monotonic() - t0
            wav = decode_audio(out[0])
            if wav is None or wav.size == 0:
                rec.update(error="no audio tokens produced")
                print(f"  [{i}/{len(todo)}] {rid}: NO AUDIO")
            else:
                write_wav(OUT / wav_name, wav)
                dur = wav.size / SR
                rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
                rec.update(ok=True, gen_s=round(gen_s, 2), audio_s=round(dur, 2),
                           rtf=round(gen_s / dur, 2) if dur else None,
                           rms=round(rms, 4), silent=bool(rms < 1e-3))
                print(f"  [{i}/{len(todo)}] {rid}: {dur:.1f}s in {gen_s:.1f}s "
                      f"(RTF {rec['rtf']}) rms={rms:.3f}{' SILENT!' if rec['silent'] else ''}")
        except Exception as e:
            rec.update(error=str(e))
            print(f"  [{i}/{len(todo)}] {rid}: ERROR {e}"); traceback.print_exc()
        results.append(rec)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

(OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- index.html player ----
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

by_tag = {}
for r in results:
    by_tag.setdefault(r["tag"], []).append(r)
sections = []
for tag, rows in by_tag.items():
    cards = []
    for r in rows:
        if r.get("ok"):
            stat = (f"gen {r.get('gen_s')}s · audio {r.get('audio_s')}s · RTF {r.get('rtf')}"
                    + (" · ⚠SILENT" if r.get("silent") else ""))
            player = f'<audio controls preload="none" src="{esc(r["wav"])}"></audio>'
        else:
            stat = f'<span style="color:#dc2626">FAILED: {esc(r.get("error",""))}</span>'; player = ""
        cards.append(f'<div class="card"><div class="meta"><span class="badge {esc(r["category"])}">'
                     f'{esc(r["category"])}</span> <code>{esc(r["id"])}</code>'
                     f'<span class="stat">{stat}</span></div><div class="text">{esc(r["text"])}</div>'
                     f'<div class="note">{esc(r.get("note",""))}</div>{player}</div>')
    sections.append(f'<h2>{esc(tag)} — <code>{esc(rows[0]["model"])}</code></h2>' + "".join(cards))
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Orpheus 3B — Hinglish test</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:840px;margin:2rem auto;padding:0 1rem}}
h2{{font-size:1rem;margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.3rem}}
.card{{border:1px solid #e3e3e3;border-radius:10px;padding:1rem;margin:.75rem 0}}
.meta{{display:flex;gap:.5rem;align-items:center;font-size:.8rem;color:#666;margin-bottom:.5rem;flex-wrap:wrap}}
.badge{{padding:.1rem .5rem;border-radius:99px;color:#fff;font-weight:600}}
.hinglish{{background:#c2410c}}.english{{background:#2563eb}}.emotion{{background:#7c3aed}}
.stat{{margin-left:auto}}.text{{font-size:1.05rem;line-height:1.5}}.note{{color:#888;font-size:.8rem;margin:.25rem 0 .5rem}}
audio{{width:100%}}code{{background:#f3f3f3;padding:.05rem .3rem;border-radius:4px}}
</style></head><body><h1>Orpheus 3B — listening test</h1>{''.join(sections)}</body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")
ok = sum(1 for r in results if r.get("ok"))
print(f"\nDONE: {ok}/{len(results)} clips -> {OUT}")
