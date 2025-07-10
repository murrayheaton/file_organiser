#!/usr/bin/env python3
"""
Smart file organiser – v1
Run with:  python organise.py
"""

# 0 ─── Imports & setup ───────────────────────────────────────────────────────
from __future__ import annotations
import os, re, shutil, pathlib, json, logging
from typing import Final
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT"),
)
print("🔑", client.api_key[:10]+"…", "Project:", client.project)
MODEL: Final = "gpt-3.5-turbo-0125"

# 1 ─── Config – edit these paths if you like ────────────────────────────────
import argparse, os, pathlib

def parse_args():
    p = argparse.ArgumentParser(description="LLM‑powered file organiser")
    p.add_argument("--src", help="Source directory with incoming files")
    p.add_argument("--dst", help="Destination directory for renamed files")
    return p.parse_args()

args = parse_args()

#  Order of precedence:  CLI flag  >  environment variable  >  default
SOURCE_DIR = pathlib.Path("/Users/murrayheaton/Desktop/DriveDumpTest")
DEST_DIR   = pathlib.Path("/Users/murrayheaton/Desktop/DumpDest")
LOG_FILE    : Final = "organise.log"

# --- Debug: print working and source directories
import os
print(f"👀 Working directory: {os.getcwd()}")
print(f"👀 Source directory resolved to: {SOURCE_DIR}")

# 2 ─── House‑keeping – create folders & logging ─────────────────────────────
DEST_DIR.mkdir(parents=True, exist_ok=True)
if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Source path does not exist: {SOURCE_DIR}")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
print(f"✅ Logs → {LOG_FILE}")

# 3 ─── Instrument regex map (deterministic pre‑check) ───────────────────────
INSTRUMENT_MAP: Final[dict[str, str]] = {
    # pattern                  token
    r"\b(alto|eb\s*sax|ebsax)\b":          "Eb",
    r"\b(tpt|trumpet|tenor|clarinet)\b":   "Bb",
    r"\b(flute|piano|keys?|violin)\b":     "Concert",
    r"\b(trombone|tuba|bass)\b":           "BassClef",
    r"\b(chords?|gtr|guitar|rhythm)\b":    "Chords",
}

# 4 ─── Helper: add “ (1) ”, “ (2) ” … if needed ─────────────────────────────
def unique_path(dst: pathlib.Path) -> pathlib.Path:
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    n = 1
    while True:
        candidate = dst.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
        n += 1

# 5 ─── Load .env and create OpenAI client ───────────────────────────────────

# 6 ─── System prompt from the previous answer ───────────────────────────────
SYSTEM_PROMPT = """
You are a deterministic file‑renaming micro‑agent.

█  INPUT
Single user message, JSON encoded:
  { "filename": <str>, "instrument_hint": <str|null> }

█  OUTPUT (one line, no quotes)
Either
  • the new filename (no path) that obeys ALL rules, or
  • the string SKIP (if the skip rules apply).

█  DESTINATION PATTERN
Charts (.pdf .onsongarchive) → <SongTitle>_<Instrument>_Chart.<ext>
Audio  (.wav .mp3)           → <SongTitle>_<FileType>.<ext>

█  SONGTITLE
• Take the longest “natural language” chunk (usually before first ‘-’ or ‘_’).
• Remove spaces, punctuation, emoji, **and underscores**.
• Preserve the remaining capital letters and digits.
  » “Best Of My Love_Lil Boo Thang” → **BestOfMyLoveLilBooThang**

█  INSTRUMENT (charts only)
Canonical tokens:  Eb  Bb  Concert  BassClef  Chords  Lyrics
Mapping  (case‑insensitive substrings):
  Alto | Eb Sax | Eb‑Sax                   → Eb
  Bari | Baritone | Bari‑Sax | BaritoneSax → Eb      ← NEW
  Tenor | Trumpet | Tpt | Bb‑Sax           → Bb
  Trombone | Bass Tbn | Tuba               → BassClef
  Guitar | Gtr | Rhythm | Piano | Keys     → Chords
  Any .onsongarchive file                  → Lyrics
Default if no clue & no hint               → **Chords**  ← CHANGED

█  FILETYPE (audio only)
Canonical tokens:  Ableton  SPL  Cues  Original
Detection order (case‑insensitive):
    1. filename contains “cue” | “cues” | “ableton”  → Cues   ← unified
    2. contains “spl” | “sp”                         → SPL
    3. otherwise                                     → Original
Audio files NEVER carry an <Instrument> field.

█  SKIP RULES
Charts:  never skipped for part‑name words (Tenor, etc.)
Audio :  skip if filename (any case) contains  
         “Instrumental” | “Soprano” | “Alto” | “Tenor” |  
         “Acapella” | “A cappella” | “Everyone” | “All Parts”
All files: skip if the word TENOR or SOPRANO appears in FULL CAPS  
           (legacy rule for another band).

█  SELF‑CHECK (before replying)
Return SKIP or match one of:
  ^[A-Za-z0-9]+_(Eb|Bb|Concert|BassClef|Chords|Lyrics)_Chart\\.(pdf|onsongarchive)$
  ^[A-Za-z0-9]+_(Ableton|SPL|Cues|Original)\\.(wav|mp3)$

█  EXAMPLES (new cases)
Input                                       → Output
{"filename":"Gimme Gimme SP (with Cues).mp3"}       → GimmeGimme_Cues.mp3
{"filename":"Stand By Me - SPL (Instrumental).mp3"} → SKIP
{"filename":"Best Of MyLove_Lil Boo Thang - SP.mp3"}→ BestOfMyLoveLilBooThang_SPL.mp3
{"filename":"Toxic.pdf"}                            → Toxic_Chords_Chart.pdf
{"filename":"Toxic - ALL PARTS.mp3"}                → SKIP
{"filename":"Toxic_Tenor.pdf"}                      → Toxic_Bb_Chart.pdf
{"filename":"t_s My Life - Bari.pdf"}               → ItsMyLife_Eb_Chart.pdf
{"filename":"Best Of My Love_Lil Boo Thang.pdf"}    → BestOfMyLove_Chords_Chart.pdf

█  NORMALISATION GUIDANCE
If multiple variants of a SongTitle exist (e.g. *BestOfMyLoveBooThang* vs *BestOfBooThang*),  
always choose the **shortest variant already present in the destination set**.
Return that canonicalised title in future outputs.
"""

# 7 ─── Function: ask the model for one filename ─────────────────────────────
def propose_new_name(src_name: str, instrument_hint: str | None) -> str:
    # Build the user prompt
    user_msg = {
        "role": "user",
        "content": json.dumps(
            {"filename": src_name, "instrument_hint": instrument_hint},
            separators=(",", ":"),
        )
    }
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        user_msg
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=chat_messages,  # type: ignore[arg-type]
        temperature=0,
        max_tokens=30,
    )
    raw_content = resp.choices[0].message.content
    answer = raw_content.strip() if raw_content is not None else ""
    logging.info("LLM %s → %s", src_name, answer)
    return answer

# 8 ─── Validation: check the model obeyed the spec ──────────────────────────
CHART_RE = re.compile(r"^[A-Za-z0-9]+_(Eb|Bb|Concert|BassClef|Chords|Lyrics)_Chart\.(pdf|onsongarchive)$")
AUDIO_RE = re.compile(r"^[A-Za-z0-9]+_(Ableton|SPL|Cues|Original)\.(wav|mp3)$")

def valid_filename(name: str) -> bool:
    return bool(CHART_RE.fullmatch(name) or AUDIO_RE.fullmatch(name) or name=="SKIP")

# 9 ─── Core loop ─────────────────────────────────────────────────────────────
def organise_folder() -> None:
    for src in SOURCE_DIR.glob("**/*"):
        print(f"🔍 Scanning path: {src}")
        if not src.is_file():
            continue

        # 9.1 – skip early if extension not of interest
        ext = src.suffix.lower()
        if ext not in {".pdf", ".onsongarchive", ".mp3", ".wav"}:
            print(f"⏭️  {src.name} (unsupported type)")
            continue

        # 9.2 – skip TENOR / SOPRANO refs
        if re.search(r"\b(TENOR|SOPRANO)\b", src.name):
            print(f"🚫 {src.name} (vocal reference – skipped)")
            continue

        # 9.3 – heuristic instrument hint
        instrument_hint: str | None = None
        if ext in {".pdf", ".onsongarchive"}:
            for pat, token in INSTRUMENT_MAP.items():
                if ext in {".mp3", ".wav"} and re.search(
                        r"(Instrumental|Soprano|Alto|Tenor|Acapella|A[\\s]?cappella|Everyone|All[ _]?Parts)",
                        src.name, flags=re.I):
                    print(f"🚫 {src.name} (audio reference – skipped)")
                    continue
                # legacy full‑caps rule
                if re.search(r"\b(TENOR|SOPRANO)\b", src.name):
                    print(f"🚫 {src.name} (legacy vocal ref – skipped)")
                    continue
                    instrument_hint = token
                    break

        # 9.4 – ask the LLM
        new_name = propose_new_name(src.name, instrument_hint)

        canonical_map = {}

        def canonicalise(fname: str) -> str:
            title, *tail = fname.split("_", 1)   # split only at first underscore
            canon = shortest_title(title)
            return "_".join([canon, *tail])
        new_name = canonicalise(new_name)

        # 9.5 – safety net: validate
        if not valid_filename(new_name):
            print(f"⚠️  Model returned invalid name for {src.name}: {new_name}")
            continue
        if new_name == "SKIP":
            print(f"🚫 {src.name} (model said skip)")
            continue

        # 9.6 – move the file
        dst = unique_path(DEST_DIR / new_name)
        shutil.move(src, dst)
        print(f"✅ {src.name}  →  {dst.name}")

        # ── place near the top of organise.py
        from difflib import SequenceMatcher
        TITLE_MAP: dict[str, str] = {}   # raw → canonical

        def shortest_title(title: str) -> str:      
            """
            Return a canonical form: the shortest earlier title that
            has ≥ 0.8 similarity to the new one; otherwise keep as‑is.
            """
            for seen, canon in TITLE_MAP.items():
                ratio = SequenceMatcher(None, title, seen).ratio()
                if ratio >= 0.8:
                    # choose the shorter of the two
                    best = canon if len(canon) < len(title) else title
                    TITLE_MAP[seen] = TITLE_MAP[title] = best
                    return best
            TITLE_MAP[title] = title
            return title

# 10 ─── Entry‑point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    organise_folder()