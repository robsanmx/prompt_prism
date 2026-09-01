# Contributing to PromptPrism

Thank you for your interest in contributing to **PromptPrism**! We welcome contributions ranging from bug reports and documentation enhancements to new experimental designs and statistical metrics.

---

## 🛠️ Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/robsanmx/prompt_prism.git
cd prompt_prism
```

### 2. Create and activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies in editable mode
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,viz,deepeval,cloud]"
```

---

## 🧪 Testing and Quality Checks

Before submitting a pull request, ensure all tests and linter checks pass locally.

### Run Unit & Integration Tests
```bash
pytest
```

### Code Formatting & Linting
We enforce code formatting using **Black**, **isort**, **pyflakes**, and **flake8**:

```bash
# Format code
black prompt_prism tests examples
isort prompt_prism tests examples

# Check formatting & linting
black --check prompt_prism tests examples
isort --check prompt_prism tests examples
pyflakes prompt_prism tests examples
flake8 prompt_prism tests examples
```

### Run Example Scripts
```bash
python examples/prompt_optimization_tutorial.py
python examples/rag_prompt_optimization.py
python examples/deepeval_golden_dataset_optimization.py
```

---

## 📐 Guidelines for Contributions

- **Statistical Rigor**: Any new experimental design, ANOVA computation, or effect estimation must include mathematical verification tests in `tests/test_statistical_correctness.py` or `tests/test_design_properties.py`.
- **Type Annotations**: All public APIs, classes, and functions should include type hints.
- **Docstrings & Comments**: Provide clear Google-style docstrings for public classes and methods.
- **Backwards Compatibility**: Aim to preserve API stability; document any breaking changes in your pull request description.

---

## 🚀 Pull Request Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. Make your changes and commit with clear, descriptive commit messages (following Conventional Commits, e.g., `feat:`, `fix:`, `docs:`, `test:`).
3. Push your branch to your fork or origin:
   ```bash
   git push origin feat/my-new-feature
   ```
4. Open a Pull Request on GitHub against the `main` branch.

---

## 📜 Code of Conduct

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.
