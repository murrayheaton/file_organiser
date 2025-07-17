#!/usr/bin/env python3
"""
Smart file organiser – v1
Run with:  python organise.py
"""

# 0 ─── Imports & setup ───────────────────────────────────────────────────────
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/murrayheaton/Documents/GitHub/file_organiser/.env", override=True)  # force .env values to win
import os, re, shutil, pathlib, json, logging
from datetime import date

TRAIN_DIR = pathlib.Path("training")
TRAIN_DIR.mkdir(exist_ok=True)

def log_training_row(src_name: str, answer: str) -> None:
    """
    Append ONE supervised row to today's JSONL.
    `answer` can be a real filename or the string 'SKIP'.
    """
    row = {
        "messages": [
            {"role": "user",      "content": f"filename: {src_name}"},
            {"role": "assistant", "content": answer},
        ]
    }
    fname = TRAIN_DIR / f"{date.today()}.jsonl"
    with fname.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

from typing import Final
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT"),
)
print("🔑", client.api_key[:10]+"…", "Project:", client.project)

# ── Model ID: use fine‑tuned model if present ─────────────────────
MODEL_FILE = pathlib.Path("config/model_id.txt")
if MODEL_FILE.exists():
    MODEL = MODEL_FILE.read_text().strip()
    print("🆕  Using fine‑tuned model:", MODEL)
else:
    MODEL = "gpt-3.5-turbo-0125"
    print("ℹ️  Falling back to base model:", MODEL)

# 1 ─── Config – edit these paths if you like ────────────────────────────────
import argparse, os, pathlib

def parse_args():
    p = argparse.ArgumentParser(description="Smart file organiser")
    p.add_argument("--src",  help="Source directory")
    p.add_argument("--dst",  help="Destination directory")
    p.add_argument("--max",  type=int, default=None,
                   help="Process at most N files")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--auto",   action="store_true",
                   help="Fully automatic – never ask (default).")
    g.add_argument("--fix",    action="store_true",
                   help="Ask only when model output is invalid.")
    g.add_argument("--review", action="store_true",
                   help="Ask y/n/s (accept/error/skip) on EVERY file.")
    return p.parse_args()

args = parse_args()

#  Order of precedence:  CLI flag  >  environment variable  >  default
SOURCE_DIR = pathlib.Path("/Users/murrayheaton/Desktop/DriveDumpTest")
DEST_DIR   = pathlib.Path("/Users/murrayheaton/Desktop/DumpDest")
ERROR_DIR: pathlib.Path = DEST_DIR / "__PARSING_ERROR"   # ← NEW
LOG_FILE    : Final = "organise.log"
SKIP_DIR = DEST_DIR / "__SKIPPED"
SKIP_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR = DEST_DIR / "__PARSING_ERROR"
for d in (DEST_DIR, SKIP_DIR, ERROR_DIR):
    d.mkdir(parents=True, exist_ok=True)

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

def ask_choice(prompt: str) -> str:
    """
    Return one of 'y', 'n', 's'.
    Empty input defaults to 'n' in review‑all mode, to '' in fix mode.
    """
    try:
        return input(prompt).strip().lower()[:1]
    except EOFError:
        return ""


def move_to_skip(src: pathlib.Path) -> None:
    """Move a file into DEST_DIR/__SKIPPED so we never scan it again."""
    dst = unique_path(SKIP_DIR / src.name)
    shutil.move(src, dst)
    print(f"⏭️  {src.name} moved to {dst.relative_to(DEST_DIR.parent)}")
    logging.info("SKIPPED %s → %s", src.name, dst)


# ── 5  NEW: handle_move lives at COLUMN 0 (module scope) ─────────
def handle_move(src: pathlib.Path, proposal: str, mode: str) -> None:
    """
    mode = 'auto' | 'fix' | 'review'
    proposal = string returned by LLM (may be 'SKIP' or invalid)
    """
    valid = valid_filename(proposal) and proposal != "SKIP"

    # --- review mode -------------------------------------------------
    if mode == "review":
        print(f"📝 {src.name}\n → {proposal}")
        choice = ask_choice("   [y] accept  [n] error  [s] skip  : ")
        if choice == "y":
            proposal = shortest_title(proposal)
            dst = unique_path(DEST_DIR / proposal) if valid else unique_path(ERROR_DIR / src.name)
            shutil.move(src, dst)
            log_training_row(src.name, proposal if valid else "SKIP")
        elif choice == "s":
            move_to_skip(src); log_training_row(src.name, "SKIP")
        elif choice == "n":
            corrected = input("   ↳ Enter correct filename or blank to skip: ").strip()
            if corrected:
                corrected = shortest_title(corrected)
                dst = unique_path(ERROR_DIR / corrected)
                shutil.move(src, dst)
                log_training_row(src.name, corrected)   # positive example
            else:
                dst = unique_path(ERROR_DIR / src.name)
                shutil.move(src, dst)
                log_training_row(src.name, "SKIP")    
        else:
            dst = unique_path(ERROR_DIR / src.name)
            shutil.move(src, dst)
            log_training_row(src.name, "SKIP")
        return

    # --- fix mode ----------------------------------------------------
    if mode == "fix" and not valid:
        print(f"⚠️  Needs manual fix: {src.name}")
        corrected = input("   ↳ New filename or blank to skip: ").strip()
        if corrected:
            dst = unique_path(DEST_DIR / corrected)
            shutil.move(src, dst)
            log_training_row(src.name, corrected)
        else:
            move_to_skip(src); log_training_row(src.name, "SKIP")
        return

    # --- auto / valid path ------------------------------------------
    if valid:
        proposal = shortest_title(proposal)
        dst = unique_path(DEST_DIR / proposal)
        shutil.move(src, dst)
        log_training_row(src.name, proposal)
    else:
        dst = unique_path(ERROR_DIR / src.name)
        shutil.move(src, dst)
        log_training_row(src.name, "SKIP")


# 6 ─── System prompt from the previous answer ───────────────────────────────
SYSTEM_PROMPT = """
You are a deterministic file‑renaming micro‑agent.

 █  INPUT
   { "filename": <str>, "instrument_hint": <str|null> }

█  HINT OVERRIDE
If "instrument_hint" is **not null**, you **must** output that exact
token for the <Instrument> slot and skip all instrument-guess rules.

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
You must output **exactly one** token from:
  Eb | Bb | Concert | BassClef | Chords | Lyrics
Rules
• If the filename ends with “.onsongarchive” and includes the substr "Lyrics" ⇒ choose **Lyrics**.
• If the filename ends with “.onsongarchive” and does not include the substr "Lyrics" ⇒ choose **Chords**.
• Otherwise use these mappings (case‑insensitive substr):
    Alto  | Bari | Baritone | Eb               → Eb
    Tenor | Trumpet | Tpt | Bb                 → Bb
    Trombone | Bass | Tbn                      → BassClef
    Guitar | Gtr | Concert | Piano | Keys      → Concert
    Lyrics                                     → Lyrics
• Never emit two instrument tokens.  
  “*_Eb_Chords_Chart.pdf*” is INVALID.

█  FILETYPE (audio only)
Canonical tokens:  Cues  SPL  Original
Detection order:
  1. filename contains “cue” | “cues” | “ableton”  → Cues
  2. contains “spl” | “sp”                         → SPL
  3. otherwise                                     → Original

█  SKIP RULES
Charts:  never skipped for part‑name words (Tenor, etc.)
Audio :  skip if filename (any case) contains  
         “Instrumental” | “Soprano” | “Alto” | “Tenor” |  
         “Acapella” | “Acappella” | “Everyone” | “All Parts”

█  SELF‑CHECK (before replying)
Return SKIP or match one of:
  ^[A-Za-z0-9]+_(Eb|Bb|Concert|BassClef|Chords|Lyrics)_Chart\\.(pdf|onsongarchive)$
  ^[A-Za-z0-9]+_(SPL|Cues|Original)\\.(wav|mp3)$

█  EXAMPLES (new cases)
Input                                               → Output
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
CHART_RE = re.compile(r"^[A-Za-z0-9]+_(Eb|Bb|Concert|BassClef|Chords|Lyrics)_Chart\.(pdf|onsongarchive)$")
AUDIO_RE = re.compile(r"^[A-Za-z0-9]+_(SPL|Cues|Original)\.(wav|mp3)$")
DOUBLE_TOKENS = re.compile(
    r"_[A-Z][A-Za-z]+_(Eb|Bb|Concert|BassClef|Chords|Lyrics)(?:_Chart)?\."
)
def clean_double_tokens(name: str) -> str:
    m = DOUBLE_TOKENS.search(name)
    if not m:
        return name
    # keep SongTitle and FIRST instrument token
    parts = name.split("_")
    clean = f"{parts[0]}_{parts[1]}_Chart{pathlib.Path(name).suffix}"
    return clean if valid_filename(clean) else "SKIP"
def valid_filename(name: str) -> bool:
    return bool(CHART_RE.fullmatch(name) or AUDIO_RE.fullmatch(name) or name=="SKIP")

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
    raw_content = resp.choices[0].message.content or ""
    answer = raw_content.strip()

    # retry once if we got two instrument tokens
    if DOUBLE_TOKENS.search(answer):
        logging.warning("Double‑instrument detected, retrying once: %s", answer)
        # ask again with a system nudge
        chat_messages.append({"role": "assistant", "content": answer})
        chat_messages.append(
            {"role": "user", "content": "You returned two instrument tokens; reply again with exactly one."}
        )

        answer = client.chat.completions.create(
            model=MODEL,
            messages=chat_messages,  # type: ignore[arg-type]
            temperature=0,
            max_tokens=30,
        ).choices[0].message.content or ""
        answer = answer.strip()

        answer = clean_double_tokens(answer)

    logging.info("LLM %s → %s", src_name, answer)
    return answer

# 8 ─── Validation: check the model obeyed the spec ──────────────────────────


from difflib import SequenceMatcher
TITLE_MAP: dict[str, str] = {}   # raw → canonical

def shortest_title(title: str) -> str:
    """
    Return a canonical form: the shortest earlier title that
    has ≥ 0.8 similarity to the new one; otherwise keep as-is.
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

# 9 ─── Core loop ─────────────────────────────────────────────────────────────
def organise_folder() -> None:
    # ── Title canonicalisation setup ───────────────────────────────
    # TITLE_MAP and shortest_title are now defined at module scope
    # ── Scan the source directory ─────────────────────────────────
    processed = 0  
    for src in SOURCE_DIR.glob("**/*"):
        # skip anything already moved to the skip folder
        if SKIP_DIR in src.parents:
            continue
        if args.max and processed >= args.max:
            break
        print(f"🔍 Scanning path: {src}")
        if not src.is_file():
            continue
        processed += 1

        # 9.1 – skip early if extension not of interest
        ext = src.suffix.lower()
        if ext not in {".pdf", ".onsongarchive", ".mp3", ".wav"}:
            print(f"⏭️  {src.name} (unsupported type)")
            continue

        # 9.2 – skip TENOR / SOPRANO refs
        if re.search(r"\b(TENOR|SOPRANO)\b", src.name):
            print(f"🚫 {src.name} (vocal reference – skipped)")
            continue

        # 9.3 – instrument hint
        instrument_hint: str | None = None
        if ext == ".onsongarchive":
            instrument_hint = "Lyrics" if "Lyrics" in src.name else "Chords"
        elif ext == ".pdf":
            for pat, token in INSTRUMENT_MAP.items():
                if re.search(pat, src.name, flags=re.I):
                    instrument_hint = token
                    break

        # 9.4 – ask the LLM
        proposal = propose_new_name(src.name, instrument_hint)
       

        # 9.5 – act according to the chosen mode
        mode = "review" if args.review else "fix" if args.fix else "auto"
        handle_move(src, proposal, mode)

        continue        # next file  (remove any old 9.6 block below)
    

    # (Remove the inner handle_move function entirely and use the global handle_move defined at module scope)

# 10 ─── Entry‑point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    organise_folder()