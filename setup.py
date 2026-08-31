from setuptools import setup, find_packages

setup(
    name="prompt_prism",
    version="1.0.0",
    description="Universal framework for LLM Prompt Optimization using Fractional Factorial Design of Experiments (DoE) & ANOVA",
    author="PromptPrism Team",
    packages=find_packages(include=["prompt_prism*"]),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "pydantic>=2.0.0",
        "jinja2>=3.0.0",
        "tabulate>=0.8.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.3.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0"],
        "cloud": ["google-cloud-aiplatform>=1.0.0", "vertexai>=1.0.0", "openai>=1.0.0"],
        "deepeval": ["deepeval>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "prompt-prism = prompt_prism.cli.main:main",
            "prism = prompt_prism.cli.main:main",
        ],
    },
)
