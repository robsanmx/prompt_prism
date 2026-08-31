"""
Unit tests for AliasStructure and confounding analysis.
"""

from prompt_prism.design.aliasing import AliasStructure, multiply_words


def test_multiply_words():
    assert multiply_words("A", "A") == ""
    assert multiply_words("AB", "BC") == "AC"
    assert multiply_words("ABCE", "BCDF") == "ADEF"
    assert multiply_words("ABCD", "ABCD") == ""


def test_alias_structure_res_iv():
    # 2(6-2)IV with generators E=ABC, F=BCD -> I = ABCE = BCDF = ADEF
    alias = AliasStructure(["E=ABC", "F=BCD"])
    assert alias.resolution == 4
    assert set(alias.defining_relation) == {"ABCE", "BCDF", "ADEF"}

    # In Res IV, main effect A is aliased with 3-factor interactions: BCE, DEF
    a_aliases = alias.get_aliases_for_term("A", max_order=3)
    assert "BCE" in a_aliases or "DEF" in a_aliases

    # 2-factor interaction AB is aliased with CE
    ab_aliases = alias.get_aliases_for_term("AB", max_order=2)
    assert "CE" in ab_aliases


def test_alias_structure_res_v():
    # 2(5-1)V with generator E=ABCD -> I = ABCDE
    alias = AliasStructure(["E=ABCD"])
    assert alias.resolution == 5
    assert alias.defining_relation == ["ABCDE"]
    # Main effects aliased with 4-factor interactions only
    a_aliases = alias.get_aliases_for_term("A", max_order=3)
    assert a_aliases == []
