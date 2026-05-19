"""
Multi-Model Reference Guide

This module demonstrates various usage patterns of call_multi_model_sync.

call_multi_model_sync allows you to:
- Call multiple LLM models in parallel with the same prompt
- Compare responses across different models
- Get cost and latency metrics for each model
- Use different model configurations

Available Models (Example):
- Claude: claude-sonnet-4-5, claude-haiku-4-5
- GPT: gpt-5.2, gpt-5.2-chat, gpt-5-mini, gpt-5-nano
- Grok: grok-4-fast-non-reasoning
- DeepSeek: DeepSeek-V3.2-Speciale
- Llama: Llama-4-Maverick-17B-128E-Instruct-FP8
- Kimi: Kimi-K2-Thinking
- Phi: Phi-4

Tips:
1. Use print_sections() to debug prompts before expensive API calls
2. Start with fewer models when testing, then expand
3. Set appropriate timeouts for complex tasks
4. Use cost_unit="cents" for better readability with small costs
5. Sort by cost to find cost-effective models, sort by latency for speed
6. Use max_response_chars=0 to see full responses without truncation
"""

from benchmax.multi_model.caller import call_multi_model_sync
from benchmax.multi_model.clients import ClientRegistry
from benchmax.multi_model.inspector import BenchmarkInspector
from benchmax.rag.qa_generation.helpers import print_sections, render_template


def initialize_registry():
    """Initialize client registry with your API credentials."""
    # @@@@@ MODIFY THIS @@@@@
    API_KEY = "YOUR_API_KEY_HERE"
    ENDPOINTS = {
        "anthropic": "https://your-endpoint.com/anthropic/",
        "openai": "https://your-endpoint.com/openai/v1/",
    }
    # @@@@@ END MODIFY THIS @@@@@

    return ClientRegistry(api_key=API_KEY, endpoints=ENDPOINTS)


def example_1_basic_multi_model_call(registry: ClientRegistry):
    """
    Example 1: Basic Multi-Model Call

    Call multiple models with the same prompt and compare responses.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Multi-Model Call")
    print("=" * 80)

    # Define system and user prompts
    system_prompt = "You are a helpful assistant that provides concise answers."
    user_prompt = "What are the key differences between Python and JavaScript?"

    # List of models to test
    models = [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5.2-chat",
    ]

    # Call all models in parallel
    benchmark = call_multi_model_sync(
        registry=registry,
        prompt=user_prompt,
        system_prompt=system_prompt,
        models=models,
        max_tokens=500,
        timeout=30.0,
    )

    # Inspect results
    inspector = BenchmarkInspector(benchmark, cost_unit="cents")
    inspector.print_summary(sort_by="cost")
    inspector.print_responses(sort_by="cost", max_response_chars=500)


def example_2_with_template_variables(registry: ClientRegistry):
    """
    Example 2: With Template Variables

    Use the template rendering helpers to create dynamic prompts.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: With Template Variables")
    print("=" * 80)

    # Define a template with variables
    prompt_template = """Analyze the following topic and provide insights:

Topic: {topic}
Context: {context}
Focus Areas: {focus_areas}
"""

    # Prepare variables
    variables = {
        "topic": "Machine Learning",
        "context": "Modern AI development practices",
        "focus_areas": "reinforcement learning, neural networks, optimization",
    }

    # Render the prompt
    rendered_prompt = render_template(prompt_template, variables)

    # Call models
    models = ["grok-4-fast-non-reasoning", "Llama-4-Maverick-17B-128E-Instruct-FP8"]

    benchmark = call_multi_model_sync(
        registry=registry,
        prompt=rendered_prompt,
        system_prompt="You are a technical analyst.",
        models=models,
        max_tokens=1000,
        timeout=30.0,
    )

    inspector = BenchmarkInspector(benchmark, cost_unit="cents")
    inspector.print_summary(sort_by="latency")


def example_3_openai_specific_arguments(registry: ClientRegistry):
    """
    Example 3: With OpenAI-Specific Arguments

    Pass model-specific arguments like reasoning effort for OpenAI models.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: With OpenAI-Specific Arguments")
    print("=" * 80)

    # Models that support OpenAI-style arguments
    models = [
        "gpt-5.2",
        "gpt-5.2-chat",
        "DeepSeek-V3.2-Speciale",
    ]

    prompt = "Explain the trade-offs between microservices and monolithic architectures."

    benchmark = call_multi_model_sync(
        registry=registry,
        prompt=prompt,
        system_prompt="You are a software architecture expert.",
        models=models,
        max_tokens=1500,
        timeout=30.0,
        oai_args={
            "verbosity": "medium",
            "reasoning": {
                "effort": "medium",
            },
        },
    )

    inspector = BenchmarkInspector(benchmark, cost_unit="cents")
    inspector.print_summary(sort_by="cost")
    inspector.print_responses(sort_by="cost", max_response_chars=0)  # 0 = no truncation


def example_4_debugging_prompts(registry: ClientRegistry):
    """
    Example 4: Debugging Prompts Before Running

    Use the print_sections helper to preview your prompts.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Debugging Prompts Before Running")
    print("=" * 80)

    # Define prompts
    system_prompt = """You are a knowledge extraction specialist.
Extract key entities and themes from the provided text."""

    user_template = """Analyze the following content:

Title: {title}
Content: {content}

Provide:
1. Main themes
2. Key entities
3. Technical terms
"""

    variables = {
        "title": "Introduction to RAG Systems",
        "content": "Retrieval-Augmented Generation combines retrieval and generation...",
    }

    user_prompt = render_template(user_template, variables)

    # Preview prompts before running
    print_sections(
        ("SYSTEM PROMPT", system_prompt),
        ("USER PROMPT", user_prompt),
    )

    # Then run the actual call if needed
    # benchmark = call_multi_model_sync(...)


def example_5_testing_multiple_models(registry: ClientRegistry):
    """
    Example 5: Testing Multiple Models for Selection

    Test different models to choose the best one for your use case.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Testing Multiple Models for Selection")
    print("=" * 80)

    # Test a variety of models across different providers
    test_models = [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5.2-chat",
        "gpt-5-mini",
        "grok-4-fast-non-reasoning",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "Phi-4",
    ]

    test_prompt = "Summarize the key concepts in reinforcement learning."

    print("Testing models for selection...")
    benchmark = call_multi_model_sync(
        registry=registry,
        prompt=test_prompt,
        system_prompt="You are a concise technical educator.",
        models=test_models,
        max_tokens=500,
        timeout=30.0,
    )

    # Sort by cost to find cheapest options
    inspector = BenchmarkInspector(benchmark, cost_unit="cents")
    print("\n=== Sorted by Cost ===")
    inspector.print_summary(sort_by="cost")

    # Sort by latency to find fastest options
    print("\n=== Sorted by Latency ===")
    inspector.print_summary(sort_by="latency")

    # Print all responses for quality comparison
    print("\n=== All Responses for Quality Review ===")
    inspector.print_responses(sort_by="cost", max_response_chars=300)


def example_6_loop_through_prompts(registry: ClientRegistry):
    """
    Example 6: Loop Through Multiple Prompts

    Test models on multiple different prompts in sequence.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Loop Through Multiple Prompts")
    print("=" * 80)

    # Multiple test cases
    test_cases = [
        {
            "name": "Technical Summary",
            "prompt": "Explain how neural networks learn through backpropagation.",
        },
        {
            "name": "Code Analysis",
            "prompt": "What are the benefits of using async/await in Python?",
        },
        {
            "name": "Architecture Design",
            "prompt": "Compare event-driven vs request-response architectures.",
        },
    ]

    models = ["grok-4-fast-non-reasoning", "claude-haiku-4-5"]

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'=' * 80}")

        benchmark = call_multi_model_sync(
            registry=registry,
            prompt=test_case["prompt"],
            system_prompt="You are a technical expert.",
            models=models,
            max_tokens=800,
            timeout=30.0,
        )

        inspector = BenchmarkInspector(benchmark, cost_unit="cents")
        inspector.print_summary(sort_by="cost")
        inspector.print_responses(sort_by="cost", max_response_chars=400)


def main():
    """Run all examples."""
    # Initialize the client registry
    registry = initialize_registry()

    # Run individual examples
    # Uncomment the ones you want to run:

    # example_1_basic_multi_model_call(registry)
    # example_2_with_template_variables(registry)
    # example_3_openai_specific_arguments(registry)
    # example_4_debugging_prompts(registry)
    # example_5_testing_multiple_models(registry)
    # example_6_loop_through_prompts(registry)

    print("\nDone! Uncomment the examples you want to run in the main() function.")


if __name__ == "__main__":
    main()
