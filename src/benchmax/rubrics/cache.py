import fcntl
import json
import os
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

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
            except:  # noqa: E722
                pass


def _empty_cache_entry() -> Dict:
    return {"positive_rubrics": [], "negative_rubrics": [], "scores_history": {}}


def _format_cached_rubrics_for_prompt(cache: Dict) -> Optional[str]:
    if not cache["positive_rubrics"] and not cache["negative_rubrics"]:
        return None
    lines = ["Positive Rubrics:"]
    lines += [f"- {r['title']}: {r['description']}" for r in cache["positive_rubrics"]]
    lines += ["\nNegative Rubrics:"]
    lines += [f"- {r['title']}: {r['description']}" for r in cache["negative_rubrics"]]
    return "\n".join(lines)


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
