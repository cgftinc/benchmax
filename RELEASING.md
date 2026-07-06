# Releasing

This one source tree publishes to **two PyPI channels** (verified against PyPI
2026-07-06):

| channel | PyPI name | versions | consumers |
|---|---|---|---|
| dev | `benchmax` | `0.1.1.dev2` … `0.1.2.dev36` (the `.devN` counter in `pyproject.toml`) | internal — sandbox-runtime image bakes, trainer deploys |
| public | `castform` | `0.1`, `0.1.1` — **final versions only** | `pip install castform` users |

The importable package is always `benchmax`; the `castform` distribution is a
rename applied at cut time (the publish machinery lives outside this repo).

## Rules

- **Dev bakes**: bump the `.devN` suffix (`0.1.2.dev36` → `0.1.2.dev37`) and
  publish as `benchmax`. Never publish a bare final version to the `benchmax`
  channel — a final `0.1.2` would outrank every future `0.1.2.devN` and pin
  dev consumers.
- **Public cuts**: strip the `.devN` suffix and publish as `castform`
  (next cut: `0.1.2`). Never publish a `.devN` as `castform` —
  `pip install --upgrade castform` excludes dev releases by default, so users
  would silently stay on the previous final (today: `0.1.1`) even though
  `0.1.2.devN` PEP-440-sorts above it.
- After a public cut, continue the dev counter from the next patch version
  (`0.1.3.dev0`).
