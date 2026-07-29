# math extensions

these optional environments extend the canonical [`MathEnv`](../main.py) while reusing its dataset, upload, validation, and launch workflow.

## group scoring

```bash
cd examples/math/extensions
uv run python math_group_env.py validate
```

`MathGroupEnv` runs the normal math tool loop but waits until every sibling finishes before assigning the correctness reward. replace `validate` with `launch` to validate and launch this environment.

## error stress testing

```bash
uv run python stress_test_env.py validate
```

`StressTestMathEnv` keeps the first example healthy so local and hosted validation can pass. across the full dataset, later examples cycle through a crash that succeeds on retry, rollout setup and cleanup failures, tool failures, and individual and group reward failures.

```bash
uv run python stress_test_env.py launch
```

launching exercises trainer retry, settlement, and replacement behavior against those failures. like the canonical example, this command validates first and asks for confirmation before spending credits.
