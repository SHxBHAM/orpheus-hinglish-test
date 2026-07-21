#!/usr/bin/env python3
"""
Orpheus 3B TTS test harness (run on a GPU box, e.g. RunPod).

Renders a set of Hinglish / English / emotion-tag sentences through an Orpheus
checkpoint, writes 24 kHz WAVs, reports real-time factor (RTF), and builds an
HTML player so you can A/B against VoxCPM2 by ear.

Typical usage on RunPod:
    # English + emotion tags on the base fine-tune:
    python run_orpheus.py --model canopylabs/orpheus-3b-0.1-ft --subset emotion,english --voice tara

    # Hinglish on the Hindi fine-tune:
    python run_orpheus.py --model SachinTelecmi/Orpheus-tts-hi --subset hinglish --voice tara

Then download the --out folder and open index.html locally.
"""
import argparse
import json
import os
import time
import wave
from pathlib import Path

SAMPLE_RATE = 24000


def load_sentences(path, subset):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if subset and subset.strip().lower() != "all":
        wanted = {s.strip().lower() for s in subset.split(",") if s.strip()}
        items = [it for it in items if it["category"].lower() in wanted]
    return items


def synth_to_wav(model, text, voice, out_path, params):
    """Generate speech and stream it to a WAV file. Returns audio duration (s)."""
    # generate_speech yields raw 16-bit PCM byte chunks (SNAC-decoded).
    def _generate():
        # Pass sampling params if this build of orpheus-speech accepts them,
        # otherwise fall back to the minimal signature.
        try:
            return model.generate_speech(prompt=text, voice=voice, **params)
        except TypeError:
            return model.generate_speech(prompt=text, voice=voice)

    total_bytes = 0
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        for chunk in _generate():
            if not chunk:
                continue
            wf.writeframes(chunk)
            total_bytes += len(chunk)
    return total_bytes / 2 / SAMPLE_RATE  # bytes -> samples -> seconds


def write_player(out_dir, rows, model_name, voice):
    """Emit a self-contained index.html with an <audio> player per sentence."""
    def esc(s):
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    cards = []
    for r in rows:
        badge = r["category"]
        rtf = f"{r['rtf']:.2f}" if r.get("rtf") is not None else "—"
        dur = f"{r['audio_s']:.1f}s" if r.get("audio_s") is not None else "—"
        gen = f"{r['gen_s']:.1f}s" if r.get("gen_s") is not None else "—"
        status = "" if r.get("ok") else '<span class="err">FAILED</span> '
        cards.append(f"""
    <div class="card">
      <div class="meta"><span class="badge {esc(badge)}">{esc(badge)}</span>
        <code>{esc(r['id'])}</code>
        <span class="stat">gen {gen} · audio {dur} · RTF {rtf}</span></div>
      <div class="text">{status}{esc(r['text'])}</div>
      <div class="note">{esc(r.get('note',''))}</div>
      <audio controls preload="none" src="{esc(r['wav'])}"></audio>
    </div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Orpheus test — {esc(model_name)}</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
  h1{{font-size:1.25rem}} .sub{{color:#666;margin-bottom:1.5rem}}
  .card{{border:1px solid #e3e3e3;border-radius:10px;padding:1rem;margin:.75rem 0}}
  .meta{{display:flex;gap:.5rem;align-items:center;font-size:.8rem;color:#666;margin-bottom:.5rem;flex-wrap:wrap}}
  .badge{{padding:.1rem .5rem;border-radius:99px;color:#fff;font-weight:600}}
  .hinglish{{background:#c2410c}} .english{{background:#2563eb}} .emotion{{background:#7c3aed}}
  .stat{{margin-left:auto;font-variant-numeric:tabular-nums}}
  .text{{font-size:1.05rem;line-height:1.5;margin:.25rem 0}}
  .note{{color:#888;font-size:.8rem;margin-bottom:.5rem}}
  .err{{color:#dc2626;font-weight:700}}
  audio{{width:100%}} code{{background:#f3f3f3;padding:.05rem .3rem;border-radius:4px}}
</style></head><body>
<h1>Orpheus 3B — listening test</h1>
<div class="sub">model: <code>{esc(model_name)}</code> · voice: <code>{esc(str(voice))}</code></div>
{''.join(cards)}
</body></html>"""
    (Path(out_dir) / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SachinTelecmi/Orpheus-tts-hi",
                    help="HF model id. Base English: canopylabs/orpheus-3b-0.1-ft")
    ap.add_argument("--voice", default="tara",
                    help="Speaker name (base voices: tara/leah/jess/leo/dan/mia/zac/zoe). "
                         "For single-speaker Hindi fine-tunes try 'tara' first; if it sounds "
                         "wrong, check the model card for the trained speaker name.")
    ap.add_argument("--sentences", default=str(Path(__file__).parent / "test_sentences.json"))
    ap.add_argument("--subset", default="all",
                    help="Comma list of categories: hinglish,english,emotion or 'all'")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs"))
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help=">=1.1 is required for stable Orpheus generations")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(args.sentences, args.subset)
    if not sentences:
        raise SystemExit(f"No sentences matched subset={args.subset!r}")

    print(f"Loading {args.model} (this pulls weights on first run)...")
    from orpheus_tts import OrpheusModel  # imported here so --help works without GPU deps
    model = OrpheusModel(model_name=args.model, max_model_len=args.max_model_len)

    params = dict(temperature=args.temperature, top_p=args.top_p,
                  repetition_penalty=args.repetition_penalty)

    rows = []
    for i, s in enumerate(sentences, 1):
        wav_name = f"{s['id']}.wav"
        wav_path = out_dir / wav_name
        print(f"[{i}/{len(sentences)}] {s['id']} ({s['category']}): {s['text'][:60]}...")
        row = {**s, "wav": wav_name, "ok": False,
               "gen_s": None, "audio_s": None, "rtf": None}
        try:
            t0 = time.monotonic()
            audio_s = synth_to_wav(model, s["text"], args.voice, wav_path, params)
            gen_s = time.monotonic() - t0
            row.update(ok=True, gen_s=gen_s, audio_s=audio_s,
                       rtf=(gen_s / audio_s if audio_s else None))
            print(f"      -> {audio_s:.1f}s audio in {gen_s:.1f}s  (RTF {row['rtf']:.2f})")
        except Exception as e:  # keep going so one bad line doesn't kill the run
            print(f"      !! FAILED: {e}")
        rows.append(row)

    write_player(out_dir, rows, args.model, args.voice)
    ok = sum(1 for r in rows if r["ok"])
    print(f"\nDone: {ok}/{len(rows)} rendered. Open {out_dir/'index.html'} to listen.")


if __name__ == "__main__":
    main()
