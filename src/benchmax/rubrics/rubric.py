import asyncio
import fcntl
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, cast
import numpy as np

from openai import AsyncOpenAI
from benchmax.envs.tracking import log_env


@dataclass
class Rubric:
    title: str
    description: str
    type: Literal["positive", "negative"] = "positive"
    score_map: Optional[Dict[float, str]] = None


def _cache_dict_to_rubric(d: Dict, rubric_type: Literal["positive", "negative"]) -> "Rubric":
    return Rubric(title=d["title"], description=d["description"], type=rubric_type)


# Local file-based cache for rubrics
RUBRIC_CACHE_FILE = os.path.join(tempfile.gettempdir(), "rubric_cache.json")


def load_rubric_cache() -> Dict[str, Dict]:
    """Load rubric cache from local file with file locking."""
    if not os.path.exists(RUBRIC_CACHE_FILE):
        return {}

    try:
        with open(RUBRIC_CACHE_FILE, "r") as f:
            # Acquire shared lock for reading (multiple readers allowed)
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except Exception as e:
        print(f"Error loading rubric cache: {e}")
        return {}


def save_rubric_cache(cache: Dict[str, Dict]) -> None:
    """Save rubric cache to local file with file locking."""
    try:
        # Use a temporary file and atomic rename to prevent corruption
        temp_file = RUBRIC_CACHE_FILE + f".tmp.{os.getpid()}"

        with open(temp_file, "w") as f:
            # Acquire exclusive lock for writing (blocks all other access)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(cache, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Atomic rename (replaces old file)
        os.rename(temp_file, RUBRIC_CACHE_FILE)

    except Exception as e:
        print(f"Error saving rubric cache: {e}")
        # Clean up temp file if it exists
        if "temp_file" in locals() and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def _empty_cache_entry() -> Dict:
    return {"positive_rubrics": [], "negative_rubrics": [], "scores_history": {}}


def _extract_completion_text(completion: str | List[Dict]) -> str:
    if isinstance(completion, list):
        if not completion or completion[-1]["role"] != "assistant":
            return ""
        return completion[-1]["content"].strip()
    return str(completion).strip()


def _format_cached_rubrics_for_prompt(cache: Dict) -> Optional[str]:
    if not cache["positive_rubrics"] and not cache["negative_rubrics"]:
        return None
    lines = ["Positive Rubrics:"]
    lines += [f"- {r['title']}: {r['description']}" for r in cache["positive_rubrics"]]
    lines += ["\nNegative Rubrics:"]
    lines += [f"- {r['title']}: {r['description']}" for r in cache["negative_rubrics"]]
    return "\n".join(lines)


async def _zero_rubric_result() -> Dict[str, Any]:
    return {"score": 0, "reasoning": "Empty response", "llm_output": ""}


def _build_rubric_eval_tasks(
    completion_texts: List[str],
    rubrics: List[Rubric],
    *,
    question: str,
    model_name: str,
    base_url: str,
    timeout: Optional[float],
) -> tuple[List, List[tuple]]:
    tasks, meta = [], []
    for i, text in enumerate(completion_texts):
        for rubric in rubrics:
            if not text:
                tasks.append(_zero_rubric_result())
            else:
                tasks.append(
                    evaluate_single_rubric(
                        rubric=rubric,
                        question=question,
                        response=text,
                        model_name=model_name,
                        base_url=base_url,
                        timeout=timeout,
                    )
                )
            meta.append((i, rubric.type, rubric))
    return tasks, meta


def get_cache_for_question(question_hash: str) -> Dict:
    """Get cache entry for a specific question, creating if needed."""
    cache = load_rubric_cache()
    if question_hash not in cache:
        cache[question_hash] = _empty_cache_entry()
    return cache


def atomic_cache_update(update_fn, max_retries: int = 5) -> None:
    """
    Atomically update the cache using a read-modify-write pattern with retries.

    Args:
        update_fn: Function that takes the cache dict and modifies it in-place
        max_retries: Maximum number of retries if there are conflicts
    """
    for attempt in range(max_retries):
        try:
            # Load current cache
            cache = load_rubric_cache()

            # Apply the update function
            update_fn(cache)

            # Save back
            save_rubric_cache(cache)
            return

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to update cache after {max_retries} attempts: {e}")
                raise
            # Retry with exponential backoff
            time.sleep(0.1 * (2**attempt))


RUBRIC_EVALUATION_PROMPT = """You are evaluating a response against a specific quality criterion.

**Criterion Type**: {rubric_type}
**Criterion Title**: {title}
**Criterion Description**: {description}

**Question**: {question}
{ground_truth_block}

**Response to Evaluate**: {response}

Your task is to score this response on how well it meets (or violates) the criterion.

For POSITIVE rubrics:
- Score 1 if the response clearly demonstrates this quality
- Score 0 if the response does not demonstrate this quality

For NEGATIVE rubrics:
- Score 1 if the response clearly exhibits this flaw/problem
- Score 0 if the response does not exhibit this flaw

Respond ONLY with a JSON object:
{{
  "score": <0 or 1>,
  "reasoning": "<brief explanation>"
}}"""

RUBRIC_RANGED_EVALUATION_PROMPT = """You are evaluating a response against a specific quality criterion.

**Criterion Type**: {rubric_type}
**Criterion Title**: {title}
**Criterion Description**: {description}

**Question**: {question}
{ground_truth_block}

**Response to Evaluate**: {response}

Your task is to score this response on how well it meets (or violates) the criterion.

Use exactly one of the following scores:
{score_rubric}

Respond ONLY with a JSON object:
{{
  "score": <one of {allowed_scores}>,
  "reasoning": "<brief explanation>"
}}"""

INSTANCE_WISE_RUBRIC_GENERATION_PROMPT = """
You are an expert evaluator generating adaptive rubrics to assess model responses.

## Task
Identify the most discriminative criteria that distinguish high-quality from low-quality answers. Capture subtle quality differences that existing rubrics miss.

## Output Components
- **Description**: Detailed, specific description of what makes a response excellent/problematic
- **Title**: Concise abstract label (general, not question-specific)

## Categories
1. **Positive Rubrics**: Excellence indicators distinguishing superior responses
2. **Negative Rubrics**: Critical flaws definitively degrading quality

## Core Guidelines

### 1. Discriminative Power
- Focus ONLY on criteria meaningfully separating quality levels
- Each rubric must distinguish between otherwise similar responses
- Exclude generic criteria applying equally to all responses

### 2. Novelty & Non-Redundancy
With existing/ground truth rubrics:
- Never duplicate overlapping rubrics in meaning/scope
- Identify uncovered quality dimensions
- Add granular criteria if existing ones are broad
- Return empty lists if existing rubrics are comprehensive

### 5. Ground Truth Alignment (when ground truth is provided)
- Use the ground truth as a reference for what a complete, correct answer looks like
- Identify rubrics that capture factual accuracy, completeness, or key claims present in the ground truth
- Penalize responses that contradict or omit information central to the ground truth

### 3. Avoid Mirror Rubrics
Never create positive/negative versions of same criterion:
- ❌ "Provides clear explanations" + "Lacks clear explanations"
- ✅ Choose only the more discriminative direction

### 4. Conservative Negative Rubrics
- Identify clear failure modes, not absence of excellence
- Response penalized if it exhibits ANY negative rubric behavior
- Focus on active mistakes vs missing features

## Selection Strategy

### Quantity: 1-5 total rubrics (fewer high-quality > many generic)

### Distribution Based on Response Patterns:
- **More positive**: Responses lack sophistication but avoid major errors
- **More negative**: Systematic failure patterns present
- **Balanced**: Both excellence gaps and failure modes exist
- **Empty lists**: Existing rubrics already comprehensive

## Analysis Process
1. If ground truth is provided, identify key facts, claims, and requirements it contains
2. Group responses by quality level (use ground truth alignment as a quality signal when available)
3. Find factors separating higher/lower clusters
4. Check if factors covered by existing rubrics
5. Select criteria with highest discriminative value

## Output Format
```json
{
  "question": "<original question verbatim>",
  "positive_rubrics": [
    {"description": "<detailed excellence description>", "title": "<abstract label>"}
  ],
  "negative_rubrics": [
    {"description": "<detailed failure description>", "title": "<abstract label>"}
  ]
}
```

## Examples

**Positive:**
```json
{"description": "Anticipates and addresses potential edge cases or exceptions to the main solution, demonstrating thorough problem understanding", "title": "Edge Case Handling"}
```

**Negative:**
```json
{"description": "Conflates correlation with causation when interpreting data or making recommendations", "title": "Causal Misattribution"}
```

## Inputs
1. **Question**: Original question being answered
2. **Ground Truth** (optional): Reference answer representing an ideal response — use it to anchor what correct/complete looks like and to identify factual gaps or deviations in model responses
3. **Responses**: Multiple model responses (Response 1, Response 2, etc.)
4. **Existing Rubrics** (optional): Previously generated/ground truth rubrics

## Critical Reminders
- Each rubric must distinguish between actual provided responses
- Exclude rubrics applying equally to all responses
- Prefer empty lists over redundancy when existing rubrics are comprehensive
- Focus on observable, objective, actionable criteria
- Quality over quantity: 2 excellent rubrics > 5 mediocre ones

Generate only the most impactful, non-redundant rubrics revealing meaningful quality differences.
"""


def _static_rubric_key(title: str) -> str:
    key = title.lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return f"rubric_{key.strip('_')}"


def _extract_json(s: str) -> dict:
    """Extract JSON from a response string, handling markdown code blocks."""
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = "\n".join(s.splitlines()[1:-1]).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError("Response did not contain valid JSON.")


async def generate_instance_wise_adaptive_rubrics(
    question: str,
    ground_truth: str,
    response_list: List[str],
    model_name: Optional[str] = None,
    existing_rubrics: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = "",
    timeout: Optional[float] = None,
) -> Optional[dict]:
    """
    Generate instance-wise adaptive rubrics using OpenAI async client.

    Args:
        question: The original question
        ground_truth: The reference answer
        response_list: List of model responses to analyze
        existing_rubrics: Optional existing rubrics to consider
        model_name: Model name for rubric generation (defaults to RUBRIC_GENERATION_MODEL env var)
        base_url: Base URL for the OpenAI-compatible endpoint
        api_key: API key for authentication (defaults to empty string)
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing positive_rubrics and negative_rubrics, or None if generation fails
    """
    prompt_suffix = f"Question: {question}\nGround Truth: {ground_truth}\nResponses:\n"
    for i, response in enumerate(response_list):
        prompt_suffix += f"Response {i + 1}:\n{response}\n\n"

    if existing_rubrics:
        prompt_suffix += f"\n\nExisting Rubrics:\n{existing_rubrics}"

    prompt = INSTANCE_WISE_RUBRIC_GENERATION_PROMPT + prompt_suffix

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=3)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout,
        )

        content = response.choices[0].message.content.strip() if response.choices else ""
        if not content:
            print("Empty response from model")
            return None

        obj = _extract_json(content)
        print(f"Generated instance-wise adaptive rubrics: {obj}")
        return obj

    except Exception as e:
        print(f"Prompt: {prompt}")
        print(f"Error generating instance-wise adaptive rubrics: {e}")
        return None


async def evaluate_single_rubric(
    rubric: Rubric,
    question: str,
    ground_truth: Optional[str],
    response: str,
    model_name: str,
    base_url: str,
    api_key: str = "",
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single response against a single rubric.

    Args:
        rubric: Rubric with title, description, type, and optional score_map
        question: The original question
        ground_truth: Optional reference answer to ground evaluation
            - For generated rubrics, this may not be needed as the generation
            should capture relevant information from the ground truth already
        response: The response to evaluate
        model_name: Model to use for evaluation
        base_url: API base URL
        api_key: API key
        timeout: Request timeout

    Returns:
        Dict with "score" and "reasoning"
    """
    ground_truth_text = str(ground_truth or "").strip()
    ground_truth_block = (
        f"**Ground Truth (Optional)**: {ground_truth_text}\n" if ground_truth_text else ""
    )
    if rubric.score_map:
        allowed_scores = ", ".join(str(score) for score in rubric.score_map.keys())
        score_rubric = "\n".join(
            f"- {score}: {description}" for score, description in rubric.score_map.items()
        )
        prompt = RUBRIC_RANGED_EVALUATION_PROMPT.format(
            rubric_type=rubric.type,
            title=rubric.title,
            description=rubric.description,
            question=question,
            ground_truth_block=ground_truth_block,
            response=response,
            allowed_scores=allowed_scores,
            score_rubric=score_rubric,
        )
    else:
        prompt = RUBRIC_EVALUATION_PROMPT.format(
            rubric_type=rubric.type,
            title=rubric.title,
            description=rubric.description,
            question=question,
            ground_truth_block=ground_truth_block,
            response=response,
        )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=3)

    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout,
        )

        content = resp.choices[0].message.content.strip() if resp.choices else ""
        if not content:
            return {"score": 0, "reasoning": "Empty response", "llm_output": ""}

        result = _extract_json(content)
        return {
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "llm_output": content,
        }

    except Exception as e:
        print(f"Error evaluating rubric '{rubric.title}': {e}")
        return {"score": 0, "reasoning": f"Error: {e}", "llm_output": ""}


def filter_and_cache_rubrics(
    question_hash: str,
    new_rubrics: Dict[str, List[Dict]],
    rubric_type: str,
    scores: List[float],
) -> None:
    """
    Filter rubrics based on variance and keep only top 3 by standard deviation.
    Thread-safe using atomic cache updates.

    Args:
        question_hash: Hash of the question to use as cache key
        new_rubrics: Dict with rubric data
        rubric_type: "positive_rubrics" or "negative_rubrics"
        scores: List of scores for this rubric across all evaluated responses
    """
    rubrics_list = new_rubrics.get(rubric_type, [])

    def update_cache(all_cache: Dict[str, Dict]) -> None:
        """Update function to be executed atomically."""
        if question_hash not in all_cache:
            all_cache[question_hash] = _empty_cache_entry()

        cache = all_cache[question_hash]

        # Add scores to history for each rubric
        for rubric in rubrics_list:
            rubric_key = f"{rubric_type}_{rubric['title']}"

            # Don't store rubrics if all scores are 0s and 1s with no variance
            if len(set(scores)) <= 1:
                print(f"Skipping rubric '{rubric['title']}' - all scores are identical: {scores}")
                continue

            # Calculate standard deviation
            std = np.std(scores)

            # Store rubric with its std
            if rubric_key not in cache["scores_history"]:
                cache["scores_history"][rubric_key] = []
            cache["scores_history"][rubric_key].extend(scores)

            # Update the rubric in cache if it's new or update the existing one
            existing_rubrics = cache[rubric_type]
            rubric_with_std = {**rubric, "std": float(std), "key": rubric_key}

            # Check if this rubric already exists
            exists = False
            for j, existing in enumerate(existing_rubrics):
                if existing.get("key") == rubric_key:
                    existing_rubrics[j] = rubric_with_std
                    exists = True
                    break

            if not exists:
                existing_rubrics.append(rubric_with_std)

        # Keep only top 3 rubrics by std for this type
        cache[rubric_type] = sorted(
            cache[rubric_type], key=lambda x: x.get("std", 0), reverse=True
        )[:3]

        print(f"Cached {rubric_type}: {len(cache[rubric_type])} rubrics (top 3 by std)")
        for r in cache[rubric_type]:
            print(f"  - {r['title']}: std={r.get('std', 0):.3f}")

    # Execute the update atomically
    atomic_cache_update(update_cache)


async def _generate_and_cache_rubrics(
    completion_texts: List[str],
    user_prompt: str,
    ground_truth: str,
    model_name: str,
    llm_judge_url: str,
    timeout: Optional[float],
    question_hash: str,
    existing_rubrics: Optional[str],
    cache: Dict,
) -> Dict:
    """Generate adaptive rubrics, evaluate variance across responses, and update cache."""
    print(f"Generating rubrics for {len(completion_texts)} responses...")

    existing_rubrics_str = _format_cached_rubrics_for_prompt(cache) or existing_rubrics

    rubric_result = await generate_instance_wise_adaptive_rubrics(
        question=user_prompt,
        ground_truth=ground_truth,
        response_list=[c for c in completion_texts if c],
        model_name=model_name,
        existing_rubrics=existing_rubrics_str,
        base_url=llm_judge_url,
        timeout=timeout,
    )

    if rubric_result:
        for rubric_type, rtype in [
            ("positive_rubrics", cast(Literal["positive", "negative"], "positive")),
            ("negative_rubrics", cast(Literal["positive", "negative"], "negative")),
        ]:
            for rubric_dict in rubric_result.get(rubric_type, []):
                rubric = _cache_dict_to_rubric(rubric_dict, rtype)
                eval_tasks = [
                    evaluate_single_rubric(
                        rubric=rubric,
                        question=user_prompt,
                        response=resp,
                        model_name=model_name,
                        base_url=llm_judge_url,
                        timeout=timeout,
                    )
                    for resp in completion_texts
                ]
                scores = [r["score"] for r in await asyncio.gather(*eval_tasks)]
                filter_and_cache_rubrics(
                    question_hash=question_hash,
                    new_rubrics={rubric_type: [rubric_dict]},
                    rubric_type=rubric_type,
                    scores=scores,
                )

    return load_rubric_cache().get(question_hash, _empty_cache_entry())


async def group_rubric_based_reward_function(
    rollout_ids: List[str],
    completions: List[str | List[Dict]],
    ground_truths: List[Any],
    llm_judge_url: str = "",
    prompt: str = "",
    model: str = "",
    timeout: Optional[float] = None,
    existing_rubrics: Optional[str] = None,
    use_adaptive_rubrics: bool = False,
    static_rubrics: Optional[List[Rubric]] = None,
) -> List[Dict[str, float]]:
    """
    Group reward function that scores completions against rubrics.

    Static rubrics each produce their own reward key (rubric_<title>).
    Adaptive rubrics are aggregated into a single "rubric_adaptive" key.

    Args:
        rollout_ids: Identifiers for each rollout (used for logging)
        completions: List of model responses (str or message-list format)
        ground_truths: Reference answers, one per completion
        llm_judge_url: Base URL of the judging LLM endpoint (required)
        prompt: The original question/prompt (required)
        model: Model name for the judging LLM (required)
        timeout: Request timeout in seconds
        use_adaptive_rubrics: Whether to generate/use adaptive rubrics (default: False)
        existing_rubrics: Existing rubrics to seed adaptive generation
        static_rubrics: Fixed rubrics, each scored as its own reward.
            Format: [Rubric(title=..., description=..., type="positive"|"negative"), ...]

    Returns:
        List of dicts mapping reward name to score, one per completion.
        Static rubric keys: rubric_<title_snake_case> (1=good for pos, 1=no flaw for neg)
        Adaptive key: rubric_adaptive (normalized aggregate, only if use_adaptive_rubrics=True)
    """
    user_prompt = prompt
    model_name = model
    static_rubrics = static_rubrics or []
    static_positive = [r for r in static_rubrics if r.type == "positive"]
    static_negative = [r for r in static_rubrics if r.type == "negative"]

    if not llm_judge_url:
        raise ValueError("llm_judge_url must be provided in kwargs")
    if not user_prompt:
        raise ValueError("prompt must be provided in kwargs")
    if not model_name:
        raise ValueError("model must be provided in kwargs")

    completion_texts = [_extract_completion_text(c) for c in completions]
    ground_truth = ground_truths[0] if ground_truths else ""
    question_hash = str(abs(hash(str(user_prompt))))
    log_buffer: Dict[str, List[str]] = {rid: [] for rid in rollout_ids}
    for rid in rollout_ids:
        log_buffer[rid].append(
            f"[ground_truth]\n{ground_truth}\n{len([t for t in completion_texts if t])}"
        )

    # Adaptive rubrics are generated per-instance and aggregated into a single reward component.
    adaptive_raw = [0.0] * len(completion_texts)
    n_adaptive_pos, n_adaptive_neg = 0, 0

    if use_adaptive_rubrics:
        cache = load_rubric_cache().get(question_hash, _empty_cache_entry())
        cache = await _generate_and_cache_rubrics(
            completion_texts=completion_texts,
            user_prompt=user_prompt,
            ground_truth=ground_truth,
            model_name=model_name,
            llm_judge_url=llm_judge_url,
            timeout=timeout,
            question_hash=question_hash,
            existing_rubrics=existing_rubrics,
            cache=cache,
        )

        n_adaptive_pos = len(cache["positive_rubrics"])
        n_adaptive_neg = len(cache["negative_rubrics"])
        adap_tasks, adap_meta = [], []
        for tasks, meta in [
            _build_rubric_eval_tasks(
                completion_texts,
                [_cache_dict_to_rubric(r, "positive") for r in cache["positive_rubrics"]],
                question=user_prompt,
                model_name=model_name,
                base_url=llm_judge_url,
                timeout=timeout,
            ),
            _build_rubric_eval_tasks(
                completion_texts,
                [_cache_dict_to_rubric(r, "negative") for r in cache["negative_rubrics"]],
                question=user_prompt,
                model_name=model_name,
                base_url=llm_judge_url,
                timeout=timeout,
            ),
        ]:
            adap_tasks.extend(tasks)
            adap_meta.extend(meta)

        for (i, rubric_type, rubric), result in zip(
            adap_meta, await asyncio.gather(*adap_tasks) if adap_tasks else []
        ):
            sign = 1.0 if rubric_type == "positive" else -1.0
            adaptive_raw[i] += sign * result["score"]
            marker = "+" if rubric_type == "positive" else "-"
            log_buffer[rollout_ids[i]].append(
                f"  [{marker}][adaptive] {rubric.title}: score={result['score']} reasoning={result['reasoning']}"
            )

    # Static rubrics (each scored independently)
    static_rewards: List[Dict[str, float]] = [{} for _ in completions]
    stat_tasks, stat_meta = [], []
    for tasks, meta in [
        _build_rubric_eval_tasks(
            completion_texts,
            static_positive,
            question=user_prompt,
            model_name=model_name,
            base_url=llm_judge_url,
            timeout=timeout,
        ),
        _build_rubric_eval_tasks(
            completion_texts,
            static_negative,
            question=user_prompt,
            model_name=model_name,
            base_url=llm_judge_url,
            timeout=timeout,
        ),
    ]:
        stat_tasks.extend(tasks)
        stat_meta.extend(meta)

    for (i, rubric_type, rubric), result in zip(
        stat_meta, await asyncio.gather(*stat_tasks) if stat_tasks else []
    ):
        raw = result["score"]
        score = raw if rubric_type == "positive" else 1.0 - raw
        key = _static_rubric_key(rubric.title)
        static_rewards[i][key] = score
        marker = "+" if rubric_type == "positive" else "-"
        log_buffer[rollout_ids[i]].append(
            f"  [{marker}][static] {rubric.title} ({key}): score={score} reasoning={result['reasoning']}\n    llm_output: {result.get('llm_output', '')}"
        )

    # Final Reward dict
    rewards: List[Dict[str, float]] = []
    for idx, rollout_id in enumerate(rollout_ids):
        reward = dict(static_rewards[idx])
        if use_adaptive_rubrics:
            score_range = (n_adaptive_pos + n_adaptive_neg) or 1
            normalized = max(0.0, min(1.0, (adaptive_raw[idx] + n_adaptive_neg) / score_range))
            reward["rubric_adaptive"] = normalized
            log_buffer[rollout_id].append(
                f"rubric_adaptive: raw={adaptive_raw[idx]:.3f} normalized={normalized:.3f} "
                f"({n_adaptive_pos} pos / {n_adaptive_neg} neg adaptive rubrics)"
            )
        rewards.append(reward)

    try:
        for rid in rollout_ids:
            log_env(rid, "\n".join(log_buffer[rid]))
    except Exception as e:
        print(f"Error logging rubric evaluation details: {e}")
    return rewards


async def single_rubric_based_reward_function(
    rollout_id: str,
    completion: str | List[Dict],
    ground_truth: Any,
    rubrics: List[Rubric],
    llm_judge_url: str,
    prompt: str,
    model: str,
    timeout: Optional[float] = None,
) -> Dict[str, float]:
    """
    Score a single completion against a list of rubrics.

    Positive rubrics: 1.0 if the quality is demonstrated, 0.0 otherwise.
    Negative rubrics: 1.0 if the flaw is absent, 0.0 if present.

    Returns:
        Dict mapping rubric_<title_snake_case> -> score.
    """
    text = _extract_completion_text(completion)
    log_lines = [f"[ground_truth]\n{ground_truth}"]

    tasks = [
        _zero_rubric_result()
        if not text
        else evaluate_single_rubric(
            rubric=rubric,
            question=prompt,
            response=text,
            model_name=model,
            base_url=llm_judge_url,
            timeout=timeout,
        )
        for rubric in rubrics
    ]

    scores: Dict[str, float] = {}
    for rubric, result in zip(rubrics, await asyncio.gather(*tasks) if tasks else []):
        raw = result["score"]
        score = raw if rubric.type == "positive" else 1.0 - raw
        key = _static_rubric_key(rubric.title)
        scores[key] = score
        marker = "+" if rubric.type == "positive" else "-"
        log_lines.append(
            f"  [{marker}] {rubric.title} ({key}): score={score} reasoning={result['reasoning']}"
        )

    try:
        log_env(rollout_id, "\n".join(log_lines))
    except Exception as e:
        print(f"Error logging rubric evaluation details: {e}")

    return scores
