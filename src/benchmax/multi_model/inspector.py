"""Inspection and display utilities for benchmark results."""

from typing import Literal

from .models import BenchmarkResult, ModelResult
from .pricing import SortBy

CostUnit = Literal["cents", "dollars"]


def _format_cost(cost: float, unit: CostUnit) -> str:
    """Format cost value with appropriate unit symbol."""
    if unit == "cents":
        return f"{cost * 100:>7.4f}¢"
    else:
        return f"${cost:>9.6f}"


class BenchmarkInspector:
    """Inspect and display benchmark results."""

    def __init__(self, result: BenchmarkResult, cost_unit: CostUnit = "cents"):
        """Initialize inspector with benchmark results.

        Args:
            result: BenchmarkResult from call_multi_model
            cost_unit: Display costs in 'cents' or 'dollars' (default: cents)
        """
        self.result = result
        self.cost_unit = cost_unit

    def _sorted_results(self, sort_by: SortBy = "cost") -> list[ModelResult]:
        """Get results sorted by cost or latency."""
        if sort_by == "cost":
            return self.result.sorted_by_cost()
        else:
            return self.result.sorted_by_latency()

    def summary(self, sort_by: SortBy = "cost") -> str:
        """Generate a summary table of results.

        Args:
            sort_by: Sort by 'cost' or 'latency'

        Returns:
            Formatted string table
        """
        results = self._sorted_results(sort_by)

        cost_header = "Cost" if self.cost_unit == "dollars" else "Cost (¢)"
        lines = [
            "=" * 104,
            f"Benchmark Summary (sorted by {sort_by})",
            "=" * 104,
            f"{'Model':<45} | {'N':>3} | {'In Tokens':>9} | {'Out Tokens':>10} | "
            f"{'Avg Latency':>11} | {cost_header:>10}",
            "-" * 104,
        ]

        for r in results:
            lines.append(
                f"{r.model:<45} | {r.num_responses:>3} | {r.total_input_tokens:>9} | "
                f"{r.total_output_tokens:>10} | {r.avg_latency_ms:>8.0f} ms | "
                f"{_format_cost(r.total_cost, self.cost_unit)}"
            )

        lines.append("-" * 104)
        total_responses = sum(r.num_responses for r in results)
        total_input = sum(r.total_input_tokens for r in results)
        total_output = sum(r.total_output_tokens for r in results)
        total_cost = self.result.total_cost

        lines.append(
            f"{'TOTAL':<45} | {total_responses:>3} | {total_input:>9} | "
            f"{total_output:>10} | {'-':>11} | {_format_cost(total_cost, self.cost_unit)}"
        )
        lines.append("=" * 104)

        return "\n".join(lines)

    def print_summary(self, sort_by: SortBy = "cost") -> None:
        """Print summary table to stdout."""
        print(self.summary(sort_by))

    def print_responses(
        self,
        sort_by: SortBy = "cost",
        max_response_chars: int = 500,
        include_thinking: bool = False,
    ) -> None:
        """Print all model responses, sorted by cost or latency.

        Args:
            sort_by: Sort by 'cost' or 'latency'
            max_response_chars: Maximum characters to show per response (0 for unlimited)
            include_thinking: Whether to include thinking traces
        """
        results = self._sorted_results(sort_by)

        print("=" * 100)
        print(f"Model Responses (sorted by {sort_by})")
        print("=" * 100)

        for i, model_result in enumerate(results, 1):
            cost_str = _format_cost(model_result.total_cost, self.cost_unit)
            latency_str = f"{model_result.avg_latency_ms:.0f}ms"

            print(f"\n{'─' * 100}")
            print(
                f"[{i}] {model_result.model} | "
                f"Cost: {cost_str} | Avg Latency: {latency_str} | "
                f"Responses: {model_result.num_responses}"
            )
            print("─" * 100)

            for j, response in enumerate(model_result.responses, 1):
                if model_result.num_responses > 1:
                    print(f"\n  Response {j}:")

                # Truncate answer if needed
                answer = response.answer
                if max_response_chars > 0 and len(answer) > max_response_chars:
                    answer = answer[:max_response_chars] + "..."

                # Indent the answer for readability
                indented = "\n  ".join(answer.split("\n"))
                print(f"  {indented}")

                if include_thinking and response.thinking:
                    thinking = response.thinking
                    if max_response_chars > 0 and len(thinking) > max_response_chars:
                        thinking = thinking[:max_response_chars] + "..."
                    print(f"\n  [Thinking]: {thinking[:200]}...")

                print(
                    f"\n  ({response.input_tokens} in, {response.output_tokens} out, "
                    f"{response.latency_ms:.0f}ms)"
                )

        print("\n" + "=" * 100)
