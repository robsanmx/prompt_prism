"""
Design of Experiments (DoE) module: Generators, Catalogs, Aliasing, and Recommender.
"""

from .aliasing import AliasStructure
from .catalog import CATALOG_DESIGNS, get_catalog_entry, list_available_plans
from .generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    PlackettBurmanGenerator,
)
from .recommender import recommend_design

__all__ = [
    "CATALOG_DESIGNS",
    "get_catalog_entry",
    "list_available_plans",
    "FractionalFactorialGenerator",
    "PlackettBurmanGenerator",
    "FullFactorialGenerator",
    "AliasStructure",
    "recommend_design",
]
