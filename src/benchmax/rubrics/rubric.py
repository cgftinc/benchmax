import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from openai import AsyncOpenAI

from benchmax.platform.credentials import resolve_judge_key

from ._utils import _extract_json
from .prompts import (
    RUBRIC_EVALUATION_PROMPT,
    RUBRIC_RANGED_EVALUATION_PROMPT,
    RUBRIC_RANKING_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class Rubric:
    title: str
    description: str
    type: Literal["positive", "negative"] = "positive"
    score_map: Optional[Dict[float, str]] = None


def _cache_dict_to_rubric(d: Dict, rubric_type: Literal["positive", "negative"]) -> "Rubric":
    return Rubric(title=d["title"], description=d["description"], type=rubric_type)


async def evaluate_single_rubric(
    rubric: Rubric,
    question: str,
    response: str,
    model_name: str,
    base_url: str,
    ground_truth: Optional[str] = None,
    api_key: str = "",
    timeout: Optional[float] = None,
    enable_logging: bool = True,
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

    # Explicit api_key wins; otherwise resolve the Castform platform credential
    # seam (ACT_AS_TOKEN_PATH in training, PLATFORM_API_KEY in playground /
    # self-serve) — the same surface the search clients use. Falls back to None
    # (→ OPENAI_API_KEY) when no platform credential is present, for direct use.
    client = AsyncOpenAI(
        base_url=base_url, api_key=resolve_judge_key(api_key, base_url), max_retries=3
    )

    content = ""
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
        out = {
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", ""),
            "llm_output": content,
        }
        if enable_logging:
            logger.info(
                "\n┌─ rubric: %s ─────────────────────\n"
                "│ ground_truth : %s\n"
                "│ score        : %s\n"
                "│ reasoning    : %s\n"
                "│ llm_output   :\n%s\n"
                "└──────────────────────────────────────────────────",
                rubric.title,
                str(ground_truth or "").strip() or "(none)",
                out["score"],
                out["reasoning"],
                content,
            )
        return out

    except Exception as e:
        print(f"Error evaluating rubric '{rubric.title}': {e}\njudge output:\n{content}")
        return {"score": 0, "reasoning": f"Error: {e}", "llm_output": content}


async def evaluate_rubric_ranking(
    rubric: Rubric,
    question: str,
    responses: List[str],
    model_name: str,
    base_url: str,
    api_key: str = "",
    timeout: Optional[float] = None,
    ground_truth: Optional[str] = None,
    enable_logging: bool = True,
) -> Dict[str, Any]:
    """
    Rank N responses against a single rubric in one judge call and convert the
    ranking into per-response scores in [0, 1]. Empty responses score 0 and are
    excluded from the ranking sent to the judge.

    Without `ground_truth`: a response in a tier covering ranked positions [a, b]
    (0 = best) gets `1 - mid / (m - 1)`, where `mid = (a + b) / 2` and `m` is the
    number of non-empty responses.

    With a non-empty `ground_truth`: GT is added as an extra unlabeled response in
    the ranking (blind to the judge). Each response is then scored relative to
    GT's tier midpoint `g`:
      - tier midpoint `p < g` → 0.5 + 0.5 * (g - p) / g          (above GT)
      - `p == g`              → 0.5                                (tied with GT)
      - `p > g`               → 0.3 * (1 - (p - g) / (max_pos - g)) (below GT)
    The below-GT branch is discontinuous with the tied score: the best-ranked
    response below GT scores 0.3 (not ~0.5), making "worse than GT" notably
    penalized.
    The GT's own slot is not returned.
    """
    n = len(responses)
    scores = [0.0] * n
    nonempty = [(i, r) for i, r in enumerate(responses) if r]

    if not nonempty:
        return {"scores": scores, "ranking": [], "reasoning": "All responses empty", "llm_output": ""}

    use_gt = bool(ground_truth and str(ground_truth).strip())
    m = len(nonempty)

    if m == 1 and not use_gt:
        scores[nonempty[0][0]] = 1.0
        return {"scores": scores, "ranking": [[nonempty[0][0]]], "reasoning": "Only one non-empty response", "llm_output": ""}

    items = [r for _, r in nonempty]
    gt_local = m if use_gt else None
    if use_gt:
        assert ground_truth is not None
        items.append(ground_truth)
    max_local = len(items) - 1
    responses_block = "\n\n".join(f"--- Response {j} ---\n{r}" for j, r in enumerate(items))
    prompt = RUBRIC_RANKING_PROMPT.format(
        rubric_type=rubric.type,
        title=rubric.title,
        description=rubric.description,
        question=question,
        responses_block=responses_block,
        n_minus_1=max_local,
    )

    # Explicit api_key wins; otherwise resolve the Castform platform credential
    # seam — see resolve_judge_key / evaluate_single_rubric.
    client = AsyncOpenAI(
        base_url=base_url, api_key=resolve_judge_key(api_key, base_url), max_retries=3
    )

    content = ""
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout,
        )
        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not content:
            return {"scores": scores, "ranking": [], "reasoning": "Empty judge response", "llm_output": ""}

        result = _extract_json(content)
        ranking = result.get("ranking", [])

        pos_of: Dict[int, float] = {}
        position = 0
        for tier in ranking:
            if not isinstance(tier, list):
                tier = [tier]
            tier_size = len(tier)
            if tier_size == 0:
                continue
            mid = position + (tier_size - 1) / 2.0
            for j in tier:
                if isinstance(j, int) and 0 <= j <= max_local and j not in pos_of:
                    pos_of[j] = mid
            position += tier_size

        max_pos = max_local
        if use_gt:
            assert gt_local is not None
            gt_pos = pos_of.get(gt_local)
            if gt_pos is None:
                for j, p in pos_of.items():
                    if 0 <= j < m:
                        scores[nonempty[j][0]] = 1.0 - p / max_pos if max_pos > 0 else 1.0
            else:
                for j, p in pos_of.items():
                    if j == gt_local:
                        continue
                    if p < gt_pos:
                        sc = 0.5 + 0.5 * (gt_pos - p) / gt_pos if gt_pos > 0 else 0.5
                    elif p > gt_pos:
                        denom = max_pos - gt_pos
                        sc = 0.3 * (1.0 - (p - gt_pos) / denom) if denom > 0 else 0.3
                    else:
                        sc = 0.5
                    scores[nonempty[j][0]] = sc
        else:
            for j, p in pos_of.items():
                scores[nonempty[j][0]] = 1.0 - p / max_pos if max_pos > 0 else 1.0

        out = {
            "scores": scores,
            "ranking": ranking,
            "reasoning": result.get("reasoning", ""),
            "llm_output": content,
        }
        if enable_logging:
            scores_fmt = "  ".join(f"[{i}]={s:.3f}" for i, s in enumerate(scores))
            ranking_fmt = " > ".join(
                f"[{', '.join(str(j) for j in tier)}]" if isinstance(tier, list) else str(tier)
                for tier in ranking
            )
            logger.info(
                "\n┌─ ranked rubric: %s ────────────────────\n"
                "│ ground_truth : %s\n"
                "│ ranking      : %s\n"
                "│ scores       : %s\n"
                "│ reasoning    : %s\n"
                "│ llm_output   :\n%s\n"
                "└──────────────────────────────────────────────────",
                rubric.title,
                str(ground_truth or "").strip() or "(none)",
                ranking_fmt or "(empty)",
                scores_fmt,
                out["reasoning"],
                content,
            )
        return out
    except Exception as e:
        print(f"Error ranking rubric '{rubric.title}': {e}\njudge output:\n{content}")
        return {"scores": scores, "ranking": [], "reasoning": f"Error: {e}", "llm_output": content}
