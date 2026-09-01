"""
Alias and Confounding Structure Analyzer for Fractional Factorial Designs.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Set


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
        for r in range(1, p + 1):
            for combo in itertools.combinations(self.basic_words, r):
                current = combo[0]
                for next_w in combo[1:]:
                    current = multiply_words(current, next_w)
                if current:
                    all_words.add(current)

        return sorted(list(all_words), key=lambda w: (len(w), w))

    def _compute_resolution(self) -> int:
        """
        Resolution is the length of the shortest word in the defining relation.
        If no generators, resolution is treated as 8 (Full Factorial).
        """
        if not self.defining_relation:
            return 8
        return min(len(w) for w in self.defining_relation)

    def get_aliases_for_term(self, term: str, max_order: int = 3) -> List[str]:
        """
        Find all aliased terms for a given factor or interaction (e.g. 'A' or 'AB').
        """
        term_clean = "".join(sorted(list(term.replace(" ", ""))))
        aliases: Set[str] = set()

        for word in self.defining_relation:
            alias = multiply_words(term_clean, word)
            if alias and alias != term_clean and len(alias) <= max_order:
                aliases.add(alias)

        return sorted(list(aliases), key=lambda a: (len(a), a))

    def get_all_aliases(
        self, factors: Sequence[str], max_order: int = 2
    ) -> Dict[str, List[str]]:
        """
        Map each main factor and 2-factor interaction to its alias chain.
        """
        result: Dict[str, List[str]] = {}
        for f in factors:
            aliases = self.get_aliases_for_term(f, max_order=max_order)
            if aliases:
                result[f] = aliases

        if max_order >= 2:
            for f1, f2 in itertools.combinations(factors, 2):
                pair = f1 + f2
                aliases = self.get_aliases_for_term(pair, max_order=max_order)
                if aliases:
                    result[pair] = aliases

        return result

    def summary(self) -> str:
        """Format a human-readable summary of the alias structure and resolution."""
        res_roman = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "Full/VIII"}.get(
            self.resolution, str(self.resolution)
        )
        lines = [
            f"Design Resolution: {self.resolution} (Res {res_roman})",
            f"Defining Relation: I = {' = '.join(self.defining_relation) if self.defining_relation else 'None'}",
        ]
        if self.resolution == 3:
            lines.append(
                "⚠️ WARNING: Resolution III design — Main effects are confounded with 2-factor interactions."
            )
        elif self.resolution == 4:
            lines.append(
                "ℹ️ Resolution IV design — Main effects are unaliased with 2-factor interactions, but 2-factor interactions are aliased with each other."
            )
        elif self.resolution >= 5:
            lines.append(
                "✅ Resolution V+ design — Main effects and 2-factor interactions are unaliased with each other."
            )
        return "\n".join(lines)
