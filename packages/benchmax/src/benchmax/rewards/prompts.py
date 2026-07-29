"""Prompt construction for judge-backed rewards.

This module owns the representation of judge prompts. Reward logic supplies
domain objects and receives a complete prompt; it does not know template
placeholders, response numbering, or how existing rubrics are rendered.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rubric import Rubric

__all__ = [
    "build_adaptive_rubric_prompt",
    "build_diversity_prompt",
    "build_rubric_evaluation_prompt",
    "build_rubric_ranking_prompt",
]


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

RUBRIC_RANKING_PROMPT = """You are evaluating multiple responses to the same question against a single quality criterion. Rank them from BEST to WORST on this criterion only.

**Criterion Type**: {rubric_type}
**Criterion Title**: {title}
**Criterion Description**: {description}

**Question**: {question}

**Responses**:
{responses_block}

For POSITIVE rubrics, BEST = most clearly demonstrates the quality described.
For NEGATIVE rubrics, BEST = least exhibits the flaw described.

Compare the responses against each other on this rubric. Do not consider any other dimension of quality.

Return ONLY a JSON object of the form:
{{
  "ranking": [[<indices in best tier>], [<next tier>], ..., [<worst tier>]],
  "reasoning": "<brief explanation>"
}}

Rules:
- Indices are 0-based and refer to "Response 0", "Response 1", etc. above.
- Group tied responses in the same inner list.
- Every index from 0 to {n_minus_1} must appear exactly once across the tiers.
- If every response is equivalent on this criterion, return a single tied group containing all indices.
"""

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

DEFAULT_DIVERSITY_INSTRUCTIONS = """\
Cluster entries by underlying tactic or approach rather than exact wording.
Paraphrases of the same tactic belong together. Empty entries and refusals
belong to one null cluster."""

DIVERSITY_PROMPT = """\
You are clustering text entries for a diversity reward.

Context:
{context}

Clustering instructions:
{instructions}

Every index from 0 to {last_index} must appear exactly once.

{items}

Return only valid JSON in this shape:
{{"assignments": [{{"index": 0, "cluster_id": "tactic", "label": "description"}}]}}
"""


def build_rubric_evaluation_prompt(
    rubric: Rubric,
    *,
    question: str,
    response: str,
    ground_truth: str | None,
) -> str:
    """Build a complete prompt for scoring one response."""

    ground_truth_text = str(ground_truth or "").strip()
    ground_truth_block = (
        f"**Ground Truth (Optional)**: {ground_truth_text}\n" if ground_truth_text else ""
    )
    values = {
        "rubric_type": rubric.polarity,
        "title": rubric.title,
        "description": rubric.description,
        "question": question,
        "ground_truth_block": ground_truth_block,
        "response": response,
    }
    if rubric.score_map is None:
        return RUBRIC_EVALUATION_PROMPT.format(**values)
    return RUBRIC_RANGED_EVALUATION_PROMPT.format(
        **values,
        allowed_scores=", ".join(str(score) for score in rubric.allowed_scores),
        score_rubric="\n".join(
            f"- {score}: {rubric.score_map[score]}" for score in rubric.allowed_scores
        ),
    )


def build_rubric_ranking_prompt(
    rubric: Rubric,
    *,
    question: str,
    responses: Sequence[str],
) -> str:
    """Build a complete prompt for ranking a group of responses."""

    responses_block = "\n\n".join(
        f"--- Response {index} ---\n{text}" for index, text in enumerate(responses)
    )
    return RUBRIC_RANKING_PROMPT.format(
        rubric_type=rubric.polarity,
        title=rubric.title,
        description=rubric.description,
        question=question,
        responses_block=responses_block,
        n_minus_1=len(responses) - 1,
    )


def build_adaptive_rubric_prompt(
    *,
    question: str,
    ground_truth: str,
    responses: Sequence[str],
    existing_rubrics: Sequence[Rubric] = (),
) -> str:
    """Build a complete prompt for generating discriminative rubrics."""

    response_block = "\n\n".join(
        f"Response {index}:\n{response}" for index, response in enumerate(responses, start=1)
    )
    prompt = (
        INSTANCE_WISE_RUBRIC_GENERATION_PROMPT
        + f"\nQuestion: {question}\n"
        + f"Ground Truth: {ground_truth}\n"
        + f"Responses:\n{response_block}\n"
    )
    existing_text = format_rubrics(existing_rubrics)
    if existing_text:
        prompt += f"\nExisting Rubrics:\n{existing_text}\n"
    return prompt


def build_diversity_prompt(
    texts: Sequence[str],
    *,
    context: str = "",
    instructions: str = DEFAULT_DIVERSITY_INSTRUCTIONS,
) -> str:
    """Build a clustering prompt without exposing its formatting contract."""

    if not texts:
        raise ValueError("diversity prompt requires at least one text")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("diversity instructions must be non-empty")
    items = "\n\n".join(f"[{index}]\n{text}" for index, text in enumerate(texts))
    return DIVERSITY_PROMPT.format(
        context=context or "(none)",
        instructions=instructions.strip(),
        last_index=len(texts) - 1,
        items=items,
    )


def format_rubrics(rubrics: Sequence[Rubric]) -> str | None:
    """Render rubric definitions for inclusion in a generation prompt."""

    sections: list[str] = []
    for heading, polarity in (
        ("Positive rubrics", "positive"),
        ("Negative rubrics", "negative"),
    ):
        selected = tuple(rubric for rubric in rubrics if rubric.polarity == polarity)
        if selected:
            sections.append(heading + ":")
            sections.extend(f"- {rubric.title}: {rubric.description}" for rubric in selected)
    return "\n".join(sections) if sections else None
