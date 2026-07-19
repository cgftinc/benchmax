import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

from openai import AsyncOpenAI

from benchmax.auth import ModelAuth

from ._utils import _extract_json, _judge_call_with_retry
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


def _cache_dict_to_rubric(
    d: Dict, rubric_type: Literal["positive", "negative"]
) -> "Rubric":
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
    *,
    auth: ModelAuth | None = None,
    token_provider: Optional[Callable[[], str]] = None,
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
        f"**Ground Truth (Optional)**: {ground_truth_text}\n"
        if ground_truth_text
        else ""
    )
    if rubric.score_map:
        allowed_scores = ", ".join(str(score) for score in rubric.score_map.keys())
        score_rubric = "\n".join(
            f"- {score}: {description}"
            for score, description in rubric.score_map.items()
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

    # The client is built inside _judge_call_with_retry. Authentication is
    # explicit and is resolved immediately before this judge request.
    content = ""

    async def _call(client: AsyncOpenAI) -> Dict[str, Any]:
        nonlocal content
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

    try:
        return await _judge_call_with_retry(
            base_url,
            model_name,
            auth,
            api_key,
            token_provider,
            _call,
        )
    except Exception as e:
        # Loud: a judge failure must never masquerade as a low reward. The
        # error/error_type keys let callers surface a _judge_error metric.
        logger.error(
            "rubric '%s' evaluation failed after retries: %s: %s",
            rubric.title,
            type(e).__name__,
            e,
        )
        print(
            f"Error evaluating rubric '{rubric.title}': {e}\njudge output:\n{content}"
        )
        return {
            "score": 0,
            "reasoning": f"Error: {e}",
            "llm_output": content,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _monotonic_seams(seams: List[tuple]) -> List[tuple]:
    """Collapse anchor-ranking inversions so a higher-edge anchor is never ranked
    below a lower-edge one. `seams` = [(position, edge)] sorted by position ascending;
    edges are expected to DESCEND (the best-ranked anchor carries the highest edge).
    When the judge inverts two anchors (e.g. ranks the `great` reference below the
    `acceptable` one), the offending anchors are merged into a single seam at the
    better (smaller) position carrying the higher edge — i.e. the better-ranked
    reference is treated as the higher band, and the two references collapse to the
    same bar. A no-op when anchors are already in monotonic order.
    """
    out: List[tuple] = []
    for pos, edge in seams:
        # an edge higher than a better-positioned anchor's edge is an inversion: fold
        # them together (keep the better position, the higher edge), then re-check.
        while out and edge > out[-1][1]:
            ppos, pedge = out.pop()
            pos, edge = ppos, max(pedge, edge)
        out.append((pos, edge))
    return out


def _band_score(p: float, seams: List[tuple], max_pos: float) -> float:
    """Score a ranked position `p` (0 = best) against anchor seam points.

    `seams` = [(anchor_position, seam_score)] sorted by position ascending (so
    seam scores descend). Interpolates linearly: above the best anchor →
    (best_edge, 1.0]; between two anchors → their edges; below the worst anchor →
    [0, worst_edge). Monotonically non-increasing in `p`. Empty seams → plain
    positional score (the no-anchor formula).
    """
    if not seams:
        return 1.0 - p / max_pos if max_pos > 0 else 1.0
    g0, e0 = seams[0]  # best-ranked anchor → highest seam
    gL, eL = seams[-1]  # worst-ranked anchor → lowest seam
    if p <= g0:  # above the best anchor
        return e0 + (1.0 - e0) * ((g0 - p) / g0 if g0 > 0 else 0.0)
    if p >= gL:  # below the worst anchor
        denom = max_pos - gL
        return eL * ((max_pos - p) / denom if denom > 0 else 0.0)
    for (ga, ea), (gb, eb) in zip(seams, seams[1:]):  # between two anchors
        if ga <= p <= gb:
            span = gb - ga
            return eb + (ea - eb) * ((gb - p) / span if span > 0 else 0.0)
    return eL


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
    below_gt_ceiling: float = 0.3,
    anchors: Optional[List[str]] = None,
    band_edges: Optional[List[float]] = None,
    anchor_labels: Optional[List[str]] = None,
    auth: ModelAuth | None = None,
    token_provider: Optional[Callable[[], str]] = None,
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
      - `p > g`               → C * (1 - (p - g) / (max_pos - g))  (below GT)
    where `C = below_gt_ceiling` (default 0.3). The below-GT branch is
    discontinuous with the tied score: the best-ranked response below GT scores
    `C` (not ~0.5), making "worse than GT" notably penalized. Raise `C` (e.g.
    0.5) to widen the below-GT band when GT is a hard bar few responses clear,
    trading away that discontinuity for more resolution below the bar.
    The GT's own slot is not returned.

    Multi-anchor mode (`anchors` + `band_edges`, supersedes `ground_truth`):
    several reference responses are inserted blind, ordered worst→best, with
    `band_edges[i]` the score a response earns when it ties anchor `i`
    (ascending, e.g. acceptable→0.1, great→0.5). A response is then scored by
    linear interpolation between the anchors' ranked positions: above the best
    anchor → (best_edge, 1.0]; between two anchors → their edges; below the
    worst anchor → [0, worst_edge). This gives the score an absolute zero-point
    (a group of all-bad responses lands below the floor anchor → near 0), which
    pure relative ranking cannot. Anchor slots are not returned.
    """
    n = len(responses)
    scores = [0.0] * n
    nonempty = [(i, r) for i, r in enumerate(responses) if r]

    if not nonempty:
        return {
            "scores": scores,
            "ranking": [],
            "reasoning": "All responses empty",
            "llm_output": "",
        }

    if anchors and not band_edges:
        raise ValueError("anchors require band_edges (one score seam per anchor)")
    _alabels = anchor_labels or []
    clean_pairs = [
        (str(a), float(e), (_alabels[i] if i < len(_alabels) else None))
        for i, (a, e) in enumerate(zip(anchors or [], band_edges or []))
        if a and str(a).strip()
    ]
    clean_anchors = [(a, e) for a, e, _ in clean_pairs]
    clean_anchor_labels = [
        p[2] for p in clean_pairs
    ]  # parallel to clean_anchors; for logging
    use_anchors = bool(clean_anchors)
    use_gt = bool(ground_truth and str(ground_truth).strip()) and not use_anchors
    m = len(nonempty)

    if m == 1 and not use_gt and not use_anchors:
        scores[nonempty[0][0]] = 1.0
        return {
            "scores": scores,
            "ranking": [[nonempty[0][0]]],
            "reasoning": "Only one non-empty response",
            "llm_output": "",
        }

    items = [r for _, r in nonempty]
    gt_local = m if use_gt else None
    if use_gt:
        assert ground_truth is not None
        items.append(ground_truth)
    anchor_local_edges = []  # [(local_index_in_items, seam_edge)]
    if use_anchors:
        for a, e in clean_anchors:
            anchor_local_edges.append((len(items), e))
            items.append(a)
    max_local = len(items) - 1
    responses_block = "\n\n".join(
        f"--- Response {j} ---\n{r}" for j, r in enumerate(items)
    )
    prompt = RUBRIC_RANKING_PROMPT.format(
        rubric_type=rubric.type,
        title=rubric.title,
        description=rubric.description,
        question=question,
        responses_block=responses_block,
        n_minus_1=max_local,
    )

    # Client built per attempt inside _judge_call_with_retry (re-resolves the
    # credential on an auth retry); see evaluate_single_rubric.
    content = ""

    async def _call(client: AsyncOpenAI) -> Dict[str, Any]:
        nonlocal content
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=timeout,
        )
        content = (
            (resp.choices[0].message.content or "").strip() if resp.choices else ""
        )
        if not content:
            return {
                "scores": scores,
                "ranking": [],
                "reasoning": "Empty judge response",
                "llm_output": "",
            }

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
        if use_anchors:
            # seams: each present anchor's ranked position paired with its edge,
            # sorted by position (best rank first). Responses score by interpolation.
            seams = sorted(
                ((pos_of[loc], e) for loc, e in anchor_local_edges if loc in pos_of),
                key=lambda x: x[0],
            )
            # the judge can rank a higher-band anchor below a lower one; collapse such
            # inversions so the seams stay monotonic (else _band_score interpolates
            # backwards and a poem just below the floor anchor can outscore one above it).
            seams = _monotonic_seams(seams)
            for j, p in pos_of.items():
                if j >= m:  # an anchor slot, not a returned response
                    continue
                scores[nonempty[j][0]] = _band_score(p, seams, max_pos)
        elif use_gt:
            assert gt_local is not None
            gt_pos = pos_of.get(gt_local)
            if gt_pos is None:
                for j, p in pos_of.items():
                    if 0 <= j < m:
                        scores[nonempty[j][0]] = (
                            1.0 - p / max_pos if max_pos > 0 else 1.0
                        )
            else:
                for j, p in pos_of.items():
                    if j == gt_local:
                        continue
                    if p < gt_pos:
                        sc = 0.5 + 0.5 * (gt_pos - p) / gt_pos if gt_pos > 0 else 0.5
                    elif p > gt_pos:
                        denom = max_pos - gt_pos
                        sc = (
                            below_gt_ceiling * (1.0 - (p - gt_pos) / denom)
                            if denom > 0
                            else below_gt_ceiling
                        )
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
            # Label every judged item by its judge index (so the reasoning's
            # "Response j" maps directly) plus its role: a model response, the
            # blind ground truth, or a named anchor. Then print each poem under
            # its label so a reader can see exactly what got ranked where.
            seam_of = dict(anchor_local_edges)  # local item index -> seam edge
            labels: List[str] = []
            for j in range(len(items)):
                if j < m:
                    labels.append(f"resp{j} (model)")
                elif use_gt and j == gt_local:
                    labels.append(f"resp{j} (ground-truth)")
                else:
                    k = j - m - (1 if use_gt else 0)
                    role = (
                        clean_anchor_labels[k]
                        if k < len(clean_anchor_labels) and clean_anchor_labels[k]
                        else f"anchor@{seam_of.get(j, 0.0):g}"
                    )
                    labels.append(f"resp{j} ({role})")

            def _lab(j: int) -> str:
                return labels[j] if 0 <= j < len(labels) else f"resp{j}"

            ranking_fmt = (
                " > ".join(
                    "["
                    + ", ".join(
                        _lab(j) for j in (tier if isinstance(tier, list) else [tier])
                    )
                    + "]"
                    for tier in ranking
                )
                or "(empty)"
            )
            scores_fmt = (
                "  ".join(f"{_lab(j)}={scores[nonempty[j][0]]:.3f}" for j in range(m))
                or "(none)"
            )

            poem_blocks = []
            for j, text in enumerate(items):
                if j < m:
                    head = f"{_lab(j)} · score {scores[nonempty[j][0]]:.3f}"
                elif use_gt and j == gt_local:
                    head = f"{_lab(j)} · reference"
                else:
                    head = f"{_lab(j)} · seam {seam_of.get(j, 0.0):.2f}"
                body = "\n".join(f"│   {ln}" for ln in (str(text).splitlines() or [""]))
                poem_blocks.append(f"│ ── {head} ──\n{body}")
            poems_fmt = "\n│\n".join(poem_blocks)

            logger.info(
                "\n┌─ ranked rubric: %s ────────────────────\n"
                "│ ranking  : %s\n"
                "│ scores   : %s\n"
                "│ reasoning: %s\n"
                "│\n%s\n"
                "└──────────────────────────────────────────────────",
                rubric.title,
                ranking_fmt,
                scores_fmt,
                out["reasoning"],
                poems_fmt,
            )
        return out

    try:
        return await _judge_call_with_retry(
            base_url,
            model_name,
            auth,
            api_key,
            token_provider,
            _call,
        )
    except Exception as e:
        logger.error(
            "rubric ranking '%s' failed after retries: %s: %s",
            rubric.title,
            type(e).__name__,
            e,
        )
        print(f"Error ranking rubric '{rubric.title}': {e}\njudge output:\n{content}")
        return {
            "scores": scores,
            "ranking": [],
            "reasoning": f"Error: {e}",
            "llm_output": content,
            "error": str(e),
            "error_type": type(e).__name__,
        }
