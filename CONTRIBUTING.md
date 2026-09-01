# Contributing to PromptPrism

Thanks for taking the time to contribute. This project is a statistics library first,
so correctness of the designs and the ANOVA engine matters more than anything else.

## Getting set up

```bash
git clone https://github.com/robsanmx/prompt_prism.git
cd prompt_prism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"
```

## Before you open a pull request

Run the same checks CI runs:

```bash
pytest                                      # 48 tests, ~5s, no network required
black --check prompt_prism tests examples
isort --check-only prompt_prism tests examples
flake8 prompt_prism tests examples
```

`black` and `isort` are configured in `pyproject.toml` and `flake8` in `.flake8`; run the
formatters without `--check` to apply them. All three are CI gates, and `examples/` is in
scope for every one of them — an example that no longer imports is a broken example.

## What a good change looks like

- **Tests come with the change.** Every new metric, design, or analysis path needs a
  test. Statistical claims need a test that pins the expected number, ideally against a
  worked example from Box-Hunter or Montgomery.
- **No network in tests.** Use `MockLLM` or a plain `fn(prompt) -> str` callable. The
  suite must stay runnable offline.
- **New designs go in the catalog.** Add generators to `prompt_prism/design/catalog.py`
  and let `aliasing.py` derive the alias structure and resolution rather than hardcoding
  them, so the resolution claim is checked rather than asserted.
- **Keep the public surface stable.** `prompt_prism/__init__.py` is the API; changing an
  exported signature is a breaking change and should be called out in the PR.

## Reporting bugs

Open an issue with the factor definitions, the design ID, and the output you got versus
what you expected. For a statistical discrepancy, include the design matrix — `prism
design --factors K --runs N --output design.csv` — so the result can be reproduced
without your dataset.

## License

Contributions are accepted under the [MIT License](LICENSE).
