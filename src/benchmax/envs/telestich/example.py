"""
Example: TelestichEnv — a poem-writing env.

Train a model to write telestich poems — poems where the last letter
(or character, for Chinese) of each line spells out a hidden word.

The script doubles as a demo of ``benchmax.bundle``: running it bundles
``TelestichEnv`` and prints the captured plaintext source — the same
JSON a frontend would render as "what code is in this env."

Run it from the benchmax project root (the ``telestich`` extra pulls in the
env's word-list / rhyme dependencies):

    cd core/benchmax
    CASTFORM_API_KEY=sk_... \
        uv run --extra telestich python -m benchmax.envs.telestich.example

(``CASTFORM_LLM_API_KEY`` is optional — it defaults to ``CASTFORM_API_KEY``.)

This launches a real training run on the full committed seed dataset
(~90/10 train/eval split).
"""

import asyncio
import json
import os
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

from benchmax.envs.telestich import telestich_env as telestich_env_mod
from benchmax.envs.telestich.telestich_env import TelestichEnv
from benchmax.rubrics import rubric as rubric_mod

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
# Fill these in before running a real training job. Keep them empty in the
# committed file — the notebooks guidelines forbid checking in API keys.
#
# Defaults route through ``benchmax.config``: the prod LLM endpoint is
# ``https://llm.castform.com/v1`` and the platform control plane is
# ``https://api.castform.com``. Point at a different environment by setting
# ``CASTFORM_BASE_DOMAIN`` (or override URLs individually via
# ``CASTFORM_PLATFORM_URL`` / ``CASTFORM_LLM_URL``).
from benchmax import config

API_KEY = os.environ.get("CASTFORM_API_KEY", "")
# Local dataset generation only — NOT passed to the env's judge (it resolves its
# bearer via the platform act-as seam; see constructor_args below).
LLM_API_KEY = os.environ.get("CASTFORM_LLM_API_KEY") or API_KEY
LLM_BASE_URL = config.llm_url()
BASE_URL = config.platform_url()
EXPERIMENT_NAME = "telestich-2026-04-25"
EXPERIMENT_PREFIX = "telestich"
# Dataset sits next to this script so the example runs from any cwd.
DATASET_PATH = str(Path(__file__).parent / "telestich_dataset.jsonl")
NUM_EXAMPLES = 400
CONCURRENCY = 15

# (model, weight). Weights reflect observed reliability on our checks:
# - Both grok models leak banned example words and rubber-stamp the CoT self-check.
# - gpt-5.4-nano fails the exclusive-bullet and banned-list rules more than the
#   other gpt-5.4 variants, so it's downweighted too.
MODELS = [
    ("grok-4-1-fast-non-reasoning", 0.20),
    ("gpt-5.4", 0.35),
    ("gpt-5.4-mini", 0.30),
    ("gpt-5.4-nano", 0.15),
]

# Models that can generate Mandarin requests natively
MANDARIN_MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]

# ══════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# Sampling axes and prompt construction for creating new training examples.
# ══════════════════════════════════════════════════════════════════════

# ── Diversity axes ──
# Abstract concept descriptions only. No illustrative example words — the
# generator picks its own hidden word based on the description plus the
# other style axes. This avoids priming toward canonical choices.
RELATIONSHIPS = [
    "The hidden word is a sweet secret message related to the poem's content.",
    "The hidden word is a mean or snarky secret message that quietly undercuts the surface content.",
    "The hidden word is a confession — it admits wrongdoing or regret that the surface poem doesn't address.",
    "The hidden word is passive aggressive — dismissive or deflecting, undercutting the apologetic/warm surface.",
    "The hidden word contradicts or clashes with the poem's surface meaning.",
    "The hidden word reinforces and deepens the poem's theme.",
    "The hidden word reveals an unspoken emotion beneath the surface.",
    "The poem is for a specific occasion or person (birthday, wedding, holiday, retirement, graduation, anniversary, etc.).",
    "The hidden word creates humor through absurd contrast with the surface content.",
    "The hidden word is subversive — the surface is praise, the hidden word accuses or dissents.",
    "The poem is self-referential or puzzle-like.",
]

LANGUAGES = ["English", "Mandarin (简体中文)"]

# Each bullet is exclusive: use that phrasing ONLY, do not mix descriptors.
# This prevents the generator from belt-and-suspendering 'telestich where the
# last letters spell X' when only one descriptor was asked for.
HOW_TO_ASK = [
    "Direct mechanism: 'write a poem where the last letter of each line spells X'. Use this phrasing only — do not also name the form as 'telestich'.",
    "Hide/embed/sneak language: 'hide the word X in the last letters' or 'sneak X into the line endings'. Use only this phrasing — do not also say 'spell'.",
    "Casual shorthand: 'last letters spell X'. Keep it terse, one clause — do not also say 'telestich'.",
    "Reverse order: start with the hidden word, then say where it goes — e.g. 'the word X, spelled by the last letter of each line'.",
    "Minimal telegraphic: 'poem about Y. word: X, last letters'. No full sentences.",
    "Name the form only: call it a 'telestich' or 'telestich poem' hiding X. Do NOT also say 'last letters spell' or 'hide in last letters' — naming the form is the whole point.",
    "By analogy: 'like an acrostic but with the last letter of each line instead of the first'. Use this analogy only — don't also say 'telestich' or 'spell'.",
]

# How the hidden word is visually rendered inside the request text.
# Weighted toward plain lowercase but with real variation.
WORD_FORMATS = [
    (
        "plain lowercase — the word appears as-is with no special formatting (e.g. marry)",
        0.60,
    ),
    ('wrapped in double quotes (e.g. "marry")', 0.15),
    ("wrapped in single quotes (e.g. 'marry')", 0.10),
    (
        "ALL CAPS for emphasis (e.g. MARRY) — entire word uppercase, not Title Case",
        0.15,
    ),
]

VOICES = [
    # Internet-voice slot is split ~50/50: half gen Z, half gen-alpha brainrot.
    "gen Z slang (like rizz, no cap, bussin, lowkey), casual internet speak",
    "gen alpha brainrot (skibidi, 6-7, gyatt, Ohio, sigma, mewing, fanum tax, "
    "only in Ohio), chaotic meme-speak",
    "political speech, rhetorical and rallying",
    "chinese internet dissident — coded language, oblique references, 谐音梗",
    "corporate speak — professional, buzzwordy",
    "academic — formal, analytical",
    "classical/archaic — literary, elevated language",
    "文言文 — classical Chinese, terse and literary",
    "meme-speak — ironic, reference-heavy",
    "children's voice — simple, full of wonder",
]

TONES = [
    "funny",
    "romantic",
    "dark",
    "serene",
    "philosophical",
    "nostalgic",
    "celebratory",
    "absurd",
    "sarcastic",
    "bittersweet",
    "epic",
    "intimate",
    "melancholic",
    "playful",
    "angry",
]

TOPICS = [
    "nature",
    "love/relationships",
    "food",
    "technology",
    "animals",
    "seasons",
    "childhood",
    "space/cosmos",
    "daily life",
    "history",
    "work/career",
    "friendship",
    "mortality",
    "travel",
    "music",
    "politics",
    "family",
    "dreams",
    "city life",
    "war/conflict",
]

WORD_LENGTHS = ["3-4", "5-6", "7-8"]

# Weighted — most requests are very short
REQUEST_LENGTHS = [
    ("very short — under 15 words", 0.50),
    ("short — 1-2 sentences", 0.30),
    ("medium — 2-4 sentences", 0.15),
    ("long — a full paragraph", 0.05),
]


# ── Sampling functions ──
def is_blocked(voice, language, relationship_desc):
    """Only block truly invalid combos."""
    if voice and "文言文" in voice and "English" in language:
        return True
    if voice and "dissident" in voice and "English" in language:
        return True
    if voice and "children" in voice and "subversive" in relationship_desc.lower():
        return True
    return False


def pick_model(language):
    """Pick a weighted-random model, respecting language constraints."""
    if "Mandarin" in language:
        return random.choice(MANDARIN_MODELS)
    names = [m for m, _ in MODELS]
    weights = [w for _, w in MODELS]
    return random.choices(names, weights=weights)[0]


def sample_axes():
    """Sample a random combo of axes, omitting some for variety."""
    while True:
        # Always included
        relationship_desc = random.choice(RELATIONSHIPS)
        language = random.choices(LANGUAGES, weights=[0.9, 0.1])[0]
        how_to_ask = random.choice(HOW_TO_ASK)
        word_format = random.choices(
            [wf[0] for wf in WORD_FORMATS],
            weights=[wf[1] for wf in WORD_FORMATS],
        )[0]
        request_length = random.choices(
            [rl[0] for rl in REQUEST_LENGTHS],
            weights=[rl[1] for rl in REQUEST_LENGTHS],
        )[0]

        # Pick 2-4 optional axes
        optional_pool = {
            "voice": random.choice(VOICES),
            "tone": random.choice(TONES),
            "topic": random.choice(TOPICS),
            "word_length": random.choice(
                ["2-3", "3-4", "4-5", "5-6", "6-7"]
                if "Mandarin" in language
                else WORD_LENGTHS
            ),
        }
        n_optional = random.choice([2, 2, 3, 3, 4])
        chosen = random.sample(list(optional_pool.keys()), n_optional)
        voice = optional_pool["voice"] if "voice" in chosen else None
        tone = optional_pool["tone"] if "tone" in chosen else None
        topic = optional_pool["topic"] if "topic" in chosen else None
        word_length = optional_pool["word_length"] if "word_length" in chosen else None

        if not is_blocked(voice, language, relationship_desc):
            break

    return {
        "relationship": relationship_desc,
        "language": language,
        "word_format": word_format,
        "how_to_ask": how_to_ask,
        "request_length": request_length,
        "voice": voice,
        "tone": tone,
        "topic": topic,
        "word_length": word_length,
    }


def build_user_prompt(axes):
    """Build the generation prompt from sampled axes."""
    letter_or_char = "character (字)" if "Mandarin" in axes["language"] else "letter"

    brief_parts = []
    is_mandarin = "Mandarin" in axes["language"]
    if is_mandarin:
        brief_parts.append(
            f"The user wants a poem in {axes['language']} where the last {letter_or_char} of each line spells a hidden word. The hidden word must also be in {axes['language']} — no mixing languages for the hidden word."
        )
        brief_parts.append(
            "The hidden word must be at most 7 Chinese characters long (each character is one line of the poem, so longer hidden phrases mean longer poems)."
        )
    else:
        # English: don't mention language to the generator — it parrots "in
        # English" / "English poem" into the request. Language is already
        # implicit (the request itself is in English).
        brief_parts.append(
            f"The user wants a poem where the last {letter_or_char} of each line spells a hidden word."
        )
    brief_parts.append(
        f"The hidden word should relate to the poem like this: {axes['relationship']}"
    )
    brief_parts.append(f"How the user phrases the ask: {axes['how_to_ask']}")
    brief_parts.append(
        f"How the hidden word itself is rendered in the request text: {axes['word_format']}"
    )
    brief_parts.append(f"Keep the request {axes['request_length']}.")

    if axes["voice"]:
        brief_parts.append(
            f"The poem itself should be written in this voice/style: {axes['voice']}."
        )
    if axes["tone"]:
        brief_parts.append(f"The tone should feel {axes['tone']}.")
    if axes["topic"]:
        brief_parts.append(f"The poem should be about {axes['topic']}.")
    if axes["word_length"]:
        brief_parts.append(
            f"The hidden word should be {axes['word_length']} {letter_or_char}s long."
        )

    brief = " ".join(brief_parts)

    # Language-consistency rule is only load-bearing when the target language
    # is Mandarin (English defaults work fine without it).
    mandarin_rule = (
        "\n- LANGUAGE: The request text must make it clear the poem should be in Mandarin. "
        "Either write the whole request in Mandarin, OR if the request is in English, "
        "it must explicitly say 'in Mandarin' / 'in Chinese' / '中文'."
        if is_mandarin
        else ""
    )

    return f"""\
Generate training data for a poetry AI. Produce a realistic user request and \
the hidden target word.

{brief}

Think first, then write. Output format:

<thinking>
Hidden word: <the word you pick>
Fits the relationship: <one line — how it fits>
Real/common word: <one line — confirm it's a real, commonly-used word, not an abbreviation or invented token>
Fresh choice: <one line — confirm this is not the most cliché/obvious word for the relationship; you picked something specific to the poem's voice/tone/topic rather than the first dictionary entry for the concept>
</thinking>
<request>...the user's chat message...</request>
<hidden>...the lowercase unquoted hidden word...</hidden>

The content inside <request> must be exactly what a human would type in a chat \
box. The request MUST explicitly state the hidden word.

Rules for the request:
- Jump straight into the ask. No preamble or backstory.
- Plain text only. No markdown, no bold, no bullets.
- Natural capitalization in the request body — some people type all lowercase, some capitalize properly.
- The hidden word's casing and quoting must follow the "rendered" format above. The <hidden>...</hidden> tag always contains the lowercase unquoted form, regardless of how the request renders it.
- Don't repeat yourself or over-explain the constraint. Pick ONE way to describe the mechanism (per the "How the user phrases the ask" bullet above) and stick to it.
- Mention the hidden word EXACTLY ONCE in the request. Don't state it as a topic word and then again as the hidden word (e.g. 'love poem hiding love').
- Don't use the word 'telestich' more than once in a single request.
- No meta-commentary about why the telestich works or what readers will notice (e.g. no 'so readers notice X in the line endings', no 'end each line so the telestich gives X', no 'so the hidden word reads…', no 'while quietly insulting/accusing…'). Describe the poem and the hidden word, then stop.
- No semicolons — people don't type those in chat.
- UNAMBIGUOUS PHRASING: Phrases like "X at the end of every line" are ambiguous (could sound like every line ends with the word X itself). If you use that phrasing, the request MUST also contain the words "last letter" OR "spell" OR "telestich" to disambiguate.
- If the request length is "long — a full paragraph", it should be a focused paragraph with specifics and context — NOT stream-of-consciousness, NOT rambling, NOT a run-on sentence. 3-5 real sentences max.{mandarin_rule}

Good examples:
<thinking>
Hidden word: ashes
Fits the relationship: bleak residue contradicts the surface romance of a breakup poem
Real/common word: yes, everyday noun
Fresh choice: specific to the poem's aftermath-of-fire tone, not a generic "sad" word
</thinking>
<request>breakup poem where the last letters spell ashes</request>
<hidden>ashes</hidden>

<thinking>
Hidden word: galaxy
Fits the relationship: reinforces space theme
Real/common word: yes, common noun
Fresh choice: concrete and vivid, not a generic "space" or "star" placeholder
</thinking>
<request>write me a telestich about space, hidden word is galaxy</request>
<hidden>galaxy</hidden>

Now generate one. Output ONLY the three tags, nothing else."""


# ── Generation helpers (LLM call, parsing, validation) ──
async def _llm_call(client, model, messages, temperature=0.0, retries=3) -> str | None:
    """Make an LLM call with retries for rate limits."""
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                await asyncio.sleep(2**attempt)
                continue
            raise


def _parse_gen_tags(text):
    """Parse <request> and <hidden> tags from generation output."""
    req = re.search(r"<request>(.*?)</request>", text, re.DOTALL)
    hid = re.search(r"<hidden>(.*?)</hidden>", text, re.DOTALL)
    return (
        req.group(1).strip() if req else None,
        hid.group(1).strip() if hid else None,
    )


_HIDDEN_CUE_PATTERN = re.compile(
    r"(?:spell|spells|spelling|hide|hides|hiding|hidden|word|letter|letters|"
    r"sneak\w*|ending|endings|acrostic|telestich|"
    r"藏|拼|组成|末字|末一字|最后一[个字]|每行最后|词[:：])",
    re.IGNORECASE,
)


def _prompt_mentions_hidden_word(prompt: str, hidden: str) -> bool:
    """Return True if the prompt explicitly names the hidden word near a cue.

    Generation sometimes produces requests where the target word isn't actually
    mentioned, or appears only as free-floating content. Require either:
      1. the target quoted (single, double, or smart quotes), or
      2. the target in ALL CAPS (English only), or
      3. the target adjacent (within ~40 chars) to a cue word like "spell",
         "hide", "last letter", or the Chinese equivalents.
    """
    if not prompt or not hidden:
        return False
    p = prompt
    h = hidden.strip()
    if not h or h.lower() not in p.lower():
        return False
    # Quoted forms
    for lq, rq in (('"', '"'), ("'", "'"), ("\u2018", "\u2019"), ("\u201c", "\u201d")):
        if f"{lq}{h}{rq}".lower() in p.lower():
            return True
    # All-caps (English only)
    if re.fullmatch(r"[A-Za-z]+", h) and h.upper() in p:
        return True
    # Cue word adjacent to hidden word
    esc = re.escape(h)
    near_before = re.compile(
        rf"{_HIDDEN_CUE_PATTERN.pattern}[^.!?\n]{{0,40}}{esc}", re.IGNORECASE
    )
    near_after = re.compile(
        rf"{esc}[^.!?\n]{{0,40}}{_HIDDEN_CUE_PATTERN.pattern}", re.IGNORECASE
    )
    return bool(near_before.search(p) or near_after.search(p))


async def generate_one(client, model, axes):
    """Generate one telestich request and extract the target word."""
    user_prompt = build_user_prompt(axes)
    try:
        content = await _llm_call(
            client,
            model,
            [{"role": "user", "content": user_prompt}],
            temperature=1.0,
        )
        if content is None:
            return None, None
        return _parse_gen_tags(content)
    except Exception as e:
        print(f"  [ERROR {model}]: {e}")
        return None, None


async def generate_dataset(n, path, concurrency=CONCURRENCY):
    """Generate n telestich examples, appending each to path as it completes."""
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    saved = 0

    async def generate_and_save(model, axes):
        nonlocal saved
        prompt, target = await generate_one(client, model, axes)
        if not prompt or not target:
            return None
        if not _prompt_mentions_hidden_word(prompt, target):
            print(
                f"  [REJECT {model}] hidden word {target!r} not clearly stated: "
                f"{prompt[:120]!r}"
            )
            return None
        example = {
            "prompt": prompt,
            "ground_truth": target,
            "metadata": {
                "model": model,
                "language": axes["language"],
                "relationship": axes["relationship"][:60],
                "request_length": axes["request_length"],
            },
        }
        async with lock:
            with open(path, "a") as f:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            saved += 1
            if saved % 25 == 0:
                print(f"  ... {saved}/{n} saved")
        return example

    async def limited_generate(model, axes):
        async with sem:
            return await generate_and_save(model, axes)

    # Build all tasks
    coros = []
    for _ in range(n):
        axes = sample_axes()
        model = pick_model(axes["language"])
        coros.append(limited_generate(model, axes))

    print(f"Generating {n} examples across models (concurrency={concurrency})...")
    outputs = await asyncio.gather(*coros, return_exceptions=True)

    failures = sum(
        1 for o in outputs if isinstance(o, (Exception, type(None))) or o is None
    )
    await client.close()
    print(f"Generated {saved}/{n} examples ({failures} failures)")
    return saved


def load_dataset(path):
    """Load examples from JSONL."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return examples


# ── Dataset loading for the trainer ──
def get_dataset():
    """Load the curated English dataset IN ORDER. The file is already English-only
    (no Mandarin filter needed) and ordered to favor simpler examples first, so we
    do NOT shuffle — the order IS the curriculum — and do NOT generate; the
    committed file is the source of truth."""
    existing = load_dataset(DATASET_PATH)
    print(f"Dataset: {len(existing)} examples (curriculum order preserved)")
    return existing


# ══════════════════════════════════════════════════════════════════════
# DEMO: BUNDLE + VIEW
# ══════════════════════════════════════════════════════════════════════
# Demonstrates how ``benchmax.bundle`` packages this env class into a
# ``.pkl`` + ``.json`` pair, and how the captured plaintext source travels
# alongside the pickle so a UI can show "what code is in this env" without
# unpickling.
if __name__ == "__main__":
    import uuid

    from benchmax.platform.client import TrainerClient
    from benchmax.platform.training_run import upload_training_run
    from benchmax.platform.validation import validate_env

    if not API_KEY:
        raise SystemExit("Set CASTFORM_API_KEY before running this example.")

    print(f"Platform URL: {BASE_URL}")
    print(f"LLM URL:      {LLM_BASE_URL}\n")

    # 1. Build the dataset from the committed seed file (curriculum order). Hold out a
    #    representative eval set at random; keep TRAIN in curriculum order (simpler first)
    #    so the difficulty ramp is preserved.
    examples = get_dataset()
    if len(examples) < 2:
        raise SystemExit(f"Need >=2 examples, got {len(examples)}.")
    n_eval = max(1, len(examples) // 10)
    eval_idx = set(random.sample(range(len(examples)), n_eval))
    eval_data = [e for i, e in enumerate(examples) if i in eval_idx]
    train_data = [e for i, e in enumerate(examples) if i not in eval_idx]
    print(f"{len(train_data)} train (curriculum order) / {len(eval_data)} eval.\n")

    # 2. Bundle the env class and upload everything to platform storage.
    # Bundle config, defined once so the pre-flight validation below exercises
    # the EXACT same env_args / by-value modules / deps as the launch.
    #  - local_modules: ship env + rubric by value (the platform's installed
    #    benchmax may not contain this version of these modules).
    #  - judge bearer resolves at runtime via the device-auth / platform seam.
    constructor_args = {"judge_base_url": LLM_BASE_URL}
    local_modules = [telestich_env_mod, rubric_mod]
    # All three are still required (is_valid_word → correctness; pronouncing →
    # rhyme). Removing word_bank did NOT free any of them.
    pip_dependencies = ["english_words", "openai", "pronouncing", "wordfreq"]

    # 2. Pre-flight: validate locally + a remote smoke rollout before spending a
    #    launch. The remote leg catches bundle/instantiation failures the local
    #    checks can't (e.g. the worker's benchmax missing this env module).
    print("\nValidating env (local contract + remote smoke) ...")
    if not validate_env(
        env_class=TelestichEnv,
        env_args=constructor_args,
        train_dataset=train_data[:5],
        eval_dataset=eval_data[:2],
        local_modules=local_modules,
        pip_dependencies=pip_dependencies,
        api_key=API_KEY,
        base_url=BASE_URL,
        llm_base_url=LLM_BASE_URL,
        llm_api_key="",
        remote_examples=2,
    ):
        raise SystemExit(
            "Env validation failed — aborting before launch (see output above)."
        )

    # 3. Bundle the env class and upload everything to platform storage.
    run_name = f"telestich-full-{uuid.uuid4().hex[:8]}"
    print(f"\nUploading bundle + datasets as {run_name!r} ...")
    uploaded = upload_training_run(
        env_class=TelestichEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=run_name,
        api_key=API_KEY,
        base_url=BASE_URL,
        local_modules=local_modules,
        constructor_args=constructor_args,
        pip_dependencies=pip_dependencies,
    )
    for label, path in (
        ("env_cls", uploaded.env_cls_path),
        ("env_metadata", uploaded.env_metadata_path),
        ("train_dataset", uploaded.train_dataset_path),
        ("eval_dataset", uploaded.eval_dataset_path),
    ):
        print(f"  {label:<14}: {path}")

    # 4. Launch the training run. ``simple`` is the deployed 4B/gpu4 template.
    print("\nLaunching training run ...")
    with TrainerClient(api_key=API_KEY, base_url=BASE_URL) as trainer:
        run_id = trainer.launch_training_run(
            training_run_type="simple",
            env_cls_path=uploaded.env_cls_path,
            env_metadata_path=uploaded.env_metadata_path,
            train_dataset_path=uploaded.train_dataset_path,
            eval_dataset_path=uploaded.eval_dataset_path,
            name=run_name,
            # num_epochs: passes over the train set (platform default is 5).
            # max_response_len 3000: a brief reason + 1-2 tool rounds + poem fits well
            # under this; lowered from 4000 to cut off in-head enumeration rambles
            # sooner (they truncate to a 0-reward anyway).
            launcher_args={"max_response_len": 3000, "num_epochs": 10},
        )

    print(f"\n✓ Launched run_id={run_id}")
    print(f"  View / cancel at: {config.web_app_url()}/train/{run_id}")
