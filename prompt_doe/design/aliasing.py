"""
Alias and Confounding Structure Analyzer for Fractional Factorial Designs.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Set, Tuple


def multiply_words(w1: str, w2: str) -> str:
    """
    Multiply two factor words using modulo-2 factor cancellation (A * A = I).
    Example: multiply_words('ABCE', 'BCDF') -> 'ADEF'
    """
    all_chars = sorted(list(w1) + list(w2))
    counts: Dict[str, int] = {}
    for c in all_chars:
        counts[c] = counts.get(c, 0) + 1
    # Odd counts remain
    res = [c for c, cnt in counts.items() if cnt % 2 == 1]
    return "".join(sorted(res))


class AliasStructure:
    """
    Computes defining relations, resolution, and confounding / alias chains for 2^(k-p) designs.
    """

    def __init__(self, generators: Sequence[str]):
        """
        Args:
            generators: e.g. ['E=ABC', 'F=BCD'] or ['D=AB', 'E=AC']
        """
        self.generators = list(generators)
        self.basic_words: List[str] = []
        for gen in self.generators:
            parts = gen.replace(" ", "").split("=")
            if len(parts) == 2:
                word = "".join(sorted(parts[0] + parts[1]))
                self.basic_words.append(word)

        self.defining_relation = self._compute_full_defining_relation()
        self.resolution = self._compute_resolution()

    def _compute_full_defining_relation(self) -> List[str]:
        """Compute all 2^p - 1 words in the defining relation."""
        if not self.basic_words:
            return []

        all_words: Set[str] = set()
        p = len(self.basic_words)
        # All non-empty subsets of basic words
        for r in range(1, p + 1):
            for subset in itertools.combinations(self.basic_words, r):
                prod = subset[0]
                for w in subset[1:]:
                    prod = multiply_words(prod, w)
                if prod:
                    all_words.add(prod)

        return sorted(list(all_words), key=lambda x: (len(x), x))

    def _compute_resolution(self) -> int:
        """Resolution is the length of the shortest word in the defining relation."""
        if not self.defining_relation:
            return 99  # No confounding (Full factorial)
        return min(len(w) for w in self.defining_relation)

    def get_aliases_for_term(self, term: str, max_order: int = 3) -> List[str]:
        """
        Find all terms confounded/aliased with the given term.
        Example: term='A' in Res IV design -> ['BCE', 'DEF', ...]
        """
        aliases = []
        for word in self.defining_relation:
            alias = multiply_words(term, word)
            if alias and len(alias) <= max_order:
                aliases.append(alias)
        return sorted(aliases, key=lambda x: (len(x), x))

    def get_all_aliases(self, factors: Sequence[str], max_order: int = 2) -> Dict[str, List[str]]:
        """
        Get alias structure for all main effects and 2-factor interactions.
        """
        result: Dict[str, List[str]] = {}
        # Main effects
        for f in factors:
            aliases = self.get_aliases_for_term(f, max_order=max_order)
            result[f] = aliases

        # 2-factor interactions
        for f1, f2 in itertools.combinations(factors, 2):
            term = "".join(sorted([f1, f2]))
            aliases = self.get_aliases_for_term(term, max_order=max_order)
            result[term] = aliases

        return result

    def summary(self) -> str:
        """Format a human-readable summary of the alias structure and resolution."""
        lines = [
            f"Design Resolution: {self.resolution} ({self.resolution_name})",
            f"Defining Relation: I = " + " = ".join(self.defining_relation) if self.defining_relation else "I (No fractional confounding)",
        ]
        if self.resolution == 3:
            lines.append("Note: In Resolution III, main effects are aliased with 2-factor interactions.")
        elif self.resolution == 4:
            lines.append("Note: In Resolution IV, main effects are clean of 2-factor interactions; 2-factor interactions are aliased with each other.")
        elif self.resolution >= 5:
            lines.append("Note: In Resolution V+, main effects and 2-factor interactions are unaliased with other main effects or 2-factor interactions.")
        return "\n".join(lines)

    @property
    def resolution_name(self) -> str:
        res_map = {3: "Res III", 4: "Res IV", 5: "Res V", 6: "Res VI", 7: "Res VII", 8: "Res VIII", 99: "Full Factorial"}
        return res_map.get(self.resolution, f"Res {self.resolution}")
