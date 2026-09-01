"""Compatibility shim for tooling that still invokes ``setup.py`` directly.

All packaging metadata lives in ``pyproject.toml`` ``[project]``, which setuptools
treats as authoritative. Restating it here produced two declarations that drifted
apart - the duplicate ``dev`` extra had already lost black/isort/flake8/pyflakes and
``cloud`` had lost anthropic/litellm - so this file deliberately declares nothing.
"""

from setuptools import setup

setup()
