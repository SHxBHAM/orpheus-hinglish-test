#!/usr/bin/env python3
"""
Veena (maya-research/Veena) inference via transformers + SNAC — the native-Hinglish
reference to compare against Orpheus. Same SNAC codec + audio-token layout as
infer.py (7-token interleave, base offset 128266), so only the *front-end* differs:
Veena wraps the prompt as [HUMAN] <spk_VOICE> text [/HUMAN] [AI] [SPEECH] and uses
its own AI control tokens.

Renders the SAME test_sentences.json into ./outputs (as veena_<voice>__<id>.wav),
writes veena_results.json, then rebuilds compare.html which stacks EVERY model's
take on each sentence (Orpheus base / hindi_official / Veena voices) side by side.

Run on the L40S box (bf16, ~7GB — no 4-bit needed):
    export HF_TOKEN=hf_...            # Veena is ungated, but harmless to set
    pip install -U transformers accelerate snac numpy
    python infer_veena.py
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

# ---- Veena control token ids (fixed; from the official model card) ------------
SOS, EOS = 128257, 128258            # start / end of speech
SOH, EOH = 128259, 128260            # start / end of human
SOA, EOA = 128261, 128262            # start / end of AI   (Orpheus lacks these)
AUDIO_BASE = 128266                  # same base offset as Orpheus
CODEBOOK = 4096
AUDIO_MAX = AUDIO_BASE + 7 * CODEBOOK

# ============================================================================
# CONFIG
# ----------------------------------------------------------------------------
MODEL_ID = "maya-research/Veena"     # canonical; "veena-tts" just redirects here
# Voices: kavya (warm), agastya (male/depth), maitri (neutral), vinaya (energy).
# Two by default to cover a female-warm + male-depth teacher; add the others to sweep.
VOICES = ["kavya", "agastya"]        # e.g. ["kavya", "agastya", "maitri", "vinaya"]
CATEGORIES = ["hinglish", "english"] # Veena has no <sigh>/<laugh> tags -> skip emotion
USE_4BIT = False                     # L40S has room for bf16; set True only if VRAM-tight
# Official cap is 700; raised so long lecture lines aren't truncated. Generation
# still stops early on EOS/EOA, so this is only a runaway safety ceiling.
MAX_TOKENS_CEIL = 2048
# ============================================================================

print(f"device={DEVICE} torch={torch.__version__} cuda={torch.cuda.is_available()}")
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

sentences = json.loads((HERE / "test_sentences.json").read_text(encoding="utf-8"))
todo = [s for s in sentences if s["category"] in CATEGORIES]
print(f"loaded {len(sentences)} sentences; {len(todo)} in {CATEGORIES}")

print("loading SNAC codec (hubertsiuzdak/snac_24khz)...")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(DEVICE)

print(f"loading {MODEL_ID} (4bit={USE_4BIT})...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
load_kw = dict(trust_remote_code=True, device_map="auto")
if USE_4BIT:
    from transformers import BitsAndBytesConfig
    load_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
else:
    load_kw["dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kw).eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS


def build_prompt_ids(text, voice):
    # Veena format: [SOH] <spk_voice> text [EOH] [SOA] [SOS]  (no BOS -> add_special_tokens=False)
    body = tok.encode(f"<spk_{voice}> {text}", add_special_tokens=False)
    seq = [SOH, *body, EOH, SOA, SOS]
    return torch.tensor([seq], dtype=torch.int64)


def decode_audio(generated_ids):
    # identical layout to infer.py: keep only audio tokens, 7 per SNAC frame
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
for voice in VOICES:
    tag = f"veena_{voice}"
    print(f"\n=== {tag}: {MODEL_ID}  ({len(todo)} sentences) ===")
    for i, s in enumerate(todo, 1):
        rid = f"{tag}__{s['id']}"; wav_name = f"{rid}.wav"
        rec = {**s, "tag": tag, "model": MODEL_ID, "voice": voice,
               "wav": wav_name, "ok": False}
        try:
            ids = build_prompt_ids(s["text"], voice).to(DEVICE)
            attn = torch.ones_like(ids)
            max_new = min(int(len(s["text"]) * 1.3) * 7 + 21, MAX_TOKENS_CEIL)
            t0 = time.monotonic()
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=attn,
                    max_new_tokens=max_new, do_sample=True,
                    temperature=0.4, top_p=0.9, repetition_penalty=1.05,
                    eos_token_id=[EOS, EOA], pad_token_id=PAD)
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

(OUT / "veena_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- compare.html: every model's take stacked per sentence --------------------
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# merge Orpheus results.json (if present) with this Veena run
orpheus = []
op = OUT / "results.json"
if op.exists():
    try:
        orpheus = json.loads(op.read_text(encoding="utf-8"))
    except Exception:
        orpheus = []
allr = orpheus + results

order = [s["id"] for s in sentences]          # keep test_sentences ordering
by_id = {}
for r in allr:
    by_id.setdefault(r["id"], []).append(r)

def player(r):
    if r.get("ok"):
        stat = (f"gen {r.get('gen_s')}s · audio {r.get('audio_s')}s · RTF {r.get('rtf')}"
                + (" · ⚠SILENT" if r.get("silent") else ""))
        au = f'<audio controls preload="none" src="{esc(r["wav"])}"></audio>'
    else:
        stat = f'<span style="color:#dc2626">FAILED: {esc(r.get("error",""))}</span>'; au = ""
    label = esc(r.get("tag", "")) + (f" · {esc(r['voice'])}" if r.get("voice") else "")
    return (f'<div class="rend"><div class="rlabel">{label} '
            f'<code>{esc(r.get("model",""))}</code></div>'
            f'<div class="rstat">{stat}</div>{au}</div>')

blocks = []
for sid in order:
    rows = by_id.get(sid, [])
    if not rows:
        continue
    meta = rows[0]
    blocks.append(
        f'<div class="q"><div class="qhead"><span class="badge {esc(meta["category"])}">'
        f'{esc(meta["category"])}</span> <code>{esc(sid)}</code>'
        f'<span class="note">{esc(meta.get("note",""))}</span></div>'
        f'<div class="text">{esc(meta["text"])}</div>'
        f'<div class="rends">{"".join(player(r) for r in rows)}</div></div>')

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Orpheus vs Veena — Hinglish A/B</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.3rem}}
.q{{border:1px solid #e3e3e3;border-radius:12px;padding:1rem;margin:1rem 0}}
.qhead{{display:flex;gap:.5rem;align-items:center;font-size:.8rem;color:#666;flex-wrap:wrap}}
.badge{{padding:.1rem .5rem;border-radius:99px;color:#fff;font-weight:600}}
.hinglish{{background:#c2410c}}.english{{background:#2563eb}}.emotion{{background:#7c3aed}}
.note{{color:#999}}.text{{font-size:1.1rem;line-height:1.5;margin:.5rem 0 .75rem}}
.rends{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.6rem}}
.rend{{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:.5rem}}
.rlabel{{font-size:.78rem;font-weight:600;color:#333;margin-bottom:.2rem}}
.rstat{{font-size:.72rem;color:#777;margin-bottom:.35rem}}
audio{{width:100%}}code{{background:#f0f0f0;padding:.05rem .3rem;border-radius:4px;font-size:.85em}}
</style></head><body><h1>Orpheus vs Veena — per-sentence A/B</h1>
<p style="color:#666;font-size:.9rem">Each card = one sentence, with every model's rendition stacked for direct comparison.</p>
{''.join(blocks)}</body></html>"""
(OUT / "compare.html").write_text(html, encoding="utf-8")

ok = sum(1 for r in results if r.get("ok"))
print(f"\nDONE: {ok}/{len(results)} Veena clips -> {OUT}")
print(f"open {OUT/'compare.html'} for the Orpheus-vs-Veena A/B")
