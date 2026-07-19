import asyncio
import logging
from typing import Any, Dict, List, Optional

from benchmax.auth import ModelAuth

from ._utils import (
    _extract_completion_text,
    _static_rubric_key,
    _zero_rubric_result,
)
from .adaptive import _generate_and_cache_rubrics
from .cache import _empty_cache_entry, load_rubric_cache
from .rubric import (
    Rubric,
    _cache_dict_to_rubric,
    evaluate_rubric_ranking,
    evaluate_single_rubric,
)

logger = logging.getLogger(__name__)


def _build_rubric_eval_tasks(
    completion_texts: List[str],
    rubrics: List[Rubric],
    *,
    question: str,
    model_name: str,
    base_url: str,
    timeout: Optional[float],
    judge_auth: ModelAuth | None = None,
    api_key: str = "",
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
                        auth=judge_auth,
                        api_key=api_key,
                    )
                )
            meta.append((i, rubric.type, rubric))
    return tasks, meta


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
    judge_auth: ModelAuth | None = None,
    api_key: str = "",
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
            judge_auth=judge_auth,
            api_key=api_key,
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
                judge_auth=judge_auth,
                api_key=api_key,
            ),
            _build_rubric_eval_tasks(
                completion_texts,
                [_cache_dict_to_rubric(r, "negative") for r in cache["negative_rubrics"]],
                question=user_prompt,
                model_name=model_name,
                base_url=llm_judge_url,
                timeout=timeout,
                judge_auth=judge_auth,
                api_key=api_key,
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
            judge_auth=judge_auth,
            api_key=api_key,
        ),
        _build_rubric_eval_tasks(
            completion_texts,
            static_negative,
            question=user_prompt,
            model_name=model_name,
            base_url=llm_judge_url,
            timeout=timeout,
            judge_auth=judge_auth,
            api_key=api_key,
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

    # log_buffer is keyed by rollout_id but rubric evaluation runs outside
    # any per-rollout env_service context, so the auto-capture handler can't
    # bind these to a rollout. Emit at INFO with the rid embedded — if the
    # caller wants rollout-scoped capture they should call this inside a
    # rollout_context block.
    for rid in rollout_ids:
        logger.info("[rubric rid=%s]\n%s", rid, "\n".join(log_buffer[rid]))
    return rewards


async def group_rubric_ranked_reward_function(
    rollout_ids: List[str],
    completions: List[str | List[Dict]],
    ground_truths: List[Any],
    llm_judge_url: str = "",
    prompt: str = "",
    model: str = "",
    api_key: str = "",
    timeout: Optional[float] = None,
    static_rubrics: Optional[List[Rubric]] = None,
    include_ground_truth: bool = True,
    judge_auth: ModelAuth | None = None,
) -> List[Dict[str, float]]:
    """
    Group reward function that ranks all completions against each rubric in a
    single judge call per rubric, then converts ranks to scores in [0, 1].

    When `include_ground_truth=True` (default), the first non-empty entry of
    `ground_truths` is added as an extra blind response in each ranking; scores
    are anchored to its position (see `evaluate_rubric_ranking`).

    Produces the same output shape as `group_rubric_based_reward_function`:
    one dict per rollout with `rubric_<title_snake_case>` keys.
    """
    user_prompt = prompt
    model_name = model
    static_rubrics = static_rubrics or []

    if not llm_judge_url:
        raise ValueError("llm_judge_url must be provided in kwargs")
    if not user_prompt:
        raise ValueError("prompt must be provided in kwargs")
    if not model_name:
        raise ValueError("model must be provided in kwargs")

    completion_texts = [_extract_completion_text(c) for c in completions]
    ground_truth = ground_truths[0] if ground_truths else ""
    log_buffer: Dict[str, List[str]] = {rid: [] for rid in rollout_ids}
    for rid in rollout_ids:
        log_buffer[rid].append(
            f"[ground_truth]\n{ground_truth}\n{len([t for t in completion_texts if t])}"
        )

    gt_str = str(ground_truth) if ground_truth else ""
    gt_for_ranking = gt_str if include_ground_truth and gt_str.strip() else None
    tasks = [
        evaluate_rubric_ranking(
            rubric=rubric,
            question=user_prompt,
            responses=completion_texts,
            model_name=model_name,
            base_url=llm_judge_url,
            api_key=api_key,
            auth=judge_auth,
            timeout=timeout,
            ground_truth=gt_for_ranking,
        )
        for rubric in static_rubrics
    ]
    results = await asyncio.gather(*tasks) if tasks else []

    rewards: List[Dict[str, float]] = [{} for _ in completions]
    for rubric, result in zip(static_rubrics, results):
        key = _static_rubric_key(rubric.title)
        marker = "+" if rubric.type == "positive" else "-"
        for i, sc in enumerate(result["scores"]):
            rewards[i][key] = sc
            log_buffer[rollout_ids[i]].append(
                f"  [{marker}][ranked][rollout_idx={i}] {rubric.title} ({key}): score={sc:.3f}"
            )
            log_buffer[rollout_ids[i]].append(
                f"  [ranking][rollout_idx={i}] {rubric.title}: {result['ranking']} reasoning={result['reasoning']}"
            )

    for rid in rollout_ids:
        logger.info("[rubric rid=%s]\n%s", rid, "\n".join(log_buffer[rid]))
    return rewards


async def single_rubric_based_reward_function(
    rollout_id: str,
    completion: str | List[Dict],
    ground_truth: Any,
    rubrics: List[Rubric],
    llm_judge_url: str,
    prompt: str,
    model: str,
    api_key: str = "",
    timeout: Optional[float] = None,
    judge_auth: ModelAuth | None = None,
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
            ground_truth=ground_truth,
            api_key=api_key,
            auth=judge_auth,
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

    logger.info("[rubric rid=%s]\n%s", rollout_id, "\n".join(log_lines))
    return scores
