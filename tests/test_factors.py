"""
Unit tests for Factor, Level, FactorType, and FactorSet.
"""

import pytest
from prompt_prism.core.factors import Factor, FactorSet, FactorType, Level


def test_level_creation():
    lvl0 = Level(code=0, name="Off", content="")
    lvl1 = Level(code=1, name="On", content="You are an expert.")
    assert lvl0.code == 0
    assert lvl0.name == "Off"
    assert lvl1.content == "You are an expert."


def test_factor_binary_creation():
    factor = Factor.binary(
        name="persona",
        level_0_content="",
        level_1_content="You are a data scientist.",
        id="A",
        description="Persona factor",
    )
    assert factor.id == "A"
    assert factor.name == "persona"
    assert len(factor.levels) == 2
    assert factor.get_level(0).content == ""
    assert factor.get_level(1).content == "You are a data scientist."
    assert factor.get_level("On").content == "You are a data scientist."


def test_factor_set_management():
    f_set = FactorSet()
    f1 = Factor.binary(name="persona", level_1_content="Expert persona")
    f2 = Factor.binary(name="few_shot", level_1_content="3 examples")
    f3 = Factor.binary(name="cot", level_1_content="Step by step")

    f_set.add(f1)
    f_set.add(f2)
    f_set.add(f3)

    assert len(f_set) == 3
    assert f1.id == "A"
    assert f2.id == "B"
    assert f3.id == "C"

    assert f_set.ids == ["A", "B", "C"]
    assert f_set.names == ["persona", "few_shot", "cot"]
    assert f_set["A"].name == "persona"
    assert f_set["few_shot"].id == "B"
    assert f_set[2].name == "cot"


def test_factor_set_serialization():
    factors = [
        Factor.binary(name="persona", level_1_content="p1"),
        Factor.binary(name="cot", level_1_content="c1"),
    ]
    f_set = FactorSet(factors)
    dict_data = f_set.to_dict()
    reconstructed = FactorSet.from_dict(dict_data)

    assert len(reconstructed) == 2
    assert reconstructed["persona"].get_level(1).content == "p1"
