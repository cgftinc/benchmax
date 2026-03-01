import re
from typing import List

def extract_solution(solution_str: str):
    matches = list(re.finditer(r'<code>(.*?)</code>', solution_str, re.DOTALL))

    # If nonempty matches and exactly one <code> block exists
    if(matches and len(matches) == 1):
        return matches[0].group(1).strip()
    return None

def filter_empty_lines(lines: List[str]):
    return list(filter(lambda line : line.strip() != "", lines))

def calc_score(answer: str, ground_truth: str):
    answer = answer.strip()
    ground_truth = ground_truth.strip()

    if(answer == ground_truth):
        return 1.0
    else:
        answer_lines = filter_empty_lines(answer.splitlines(True))
        ground_truth_lines = filter_empty_lines(ground_truth.splitlines(True))
        # Give small positive reward if lines are almost correct
        if(answer_lines == ground_truth_lines):
            return 0.2
        return 0

def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score=0.0, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return 0
    else:
        return calc_score(answer, ground_truth)