from benchmax.envs.fast_apply.utils import extract_solution, filter_empty_lines, calc_score, compute_score

# Tests for extract_solution

def test_extract_solution_single_code_block():
    s = "<code>   foo bar  </code>"
    assert extract_solution(s) == "foo bar"


def test_extract_solution_multiple_code_blocks():
    s = "<code>foo</code> something <code>bar</code>"
    assert extract_solution(s) is None


def test_extract_solution_no_code_block():
    s = "no code here"
    assert extract_solution(s) is None

# Tests for filter_empty_lines

def test_filter_empty_lines():
    lines = ["foo\n", "\n", "  \t  \n", "bar\n", ""]
    assert filter_empty_lines(lines) == ["foo\n", "bar\n"]

# Tests for calc_score

def test_calc_score_exact_match():
    assert calc_score("answer", "answer") == 1.0


def test_calc_score_line_match_after_filter():
    answer = "a\n\n b"
    ground_truth = "a\n b\n"
    assert calc_score(answer, ground_truth) == 0.2


def test_calc_score_mismatch():
    assert calc_score("foo", "bar") == 0

# Tests for compute_score

def test_compute_score_no_solution():
    sol_str = "no code"
    assert compute_score(None, sol_str, "anything") == 0


def test_compute_score_correct_solution():
    sol_str = "<code>baz</code>"
    assert compute_score(None, sol_str, "baz") == 1.0


def test_compute_score_incorrect_solution():
    sol_str = "<code>baz</code>"
    assert compute_score(None, sol_str, "qux") == 0


def test_compute_score_partial_match():
    sol_str = "<code>a\n\nb</code>"
    assert compute_score(None, sol_str, "a\nb") == 0.2
