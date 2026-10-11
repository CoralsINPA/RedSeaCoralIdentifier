"""Check D of the parity notebook: the expert set evaluated against its own stored predictions.

The reference is the real `assaf_expert_test_set.csv` (77 rows), not the 78-crop matrix of the
memory notes: the two are slightly different versions of the same review. Against the matrix
the file lacks one correct Favia, one correct Favites, one correct Porites and two correct
Stylophora, and has two correct Seriatopora (`docs/agent_memory/sessions/
2026-10-11_parity_run_1.md`). The 18 rows without a genus split by `is_correct` into
unidentifiable and missing; 16 and 2 are expected but not yet confirmed on the file.
"""

from __future__ import annotations

from collections.abc import Mapping

from coralid.evaluation import EvaluationReport
from coralid.parity import CheckResult, equal, info

# What the real CSV holds (first Colab run, 2026-10-11). The 16 / 2 split is from the memory notes.
EXPERT_CSV_EXPECTED = {
    "rows": 77,
    "in_set": 54,
    "out_of_set": 5,
    "no_genus": 18,  # unidentifiable + missing
    "unidentifiable": 16,
    "missing": 2,
    "correct": 36,
    "identifiable": 59,  # in_set + out_of_set
}

# Sparse confusion matrix of the real file: true genus -> predicted genus -> count.
EXPERT_CSV_CELLS: dict[str, dict[str, int]] = {
    "Acropora": {"Acropora": 5},
    "Platygyra": {"Platygyra": 5, "Galaxea": 1},
    "Porites": {"Porites": 4, "Dipsastraea": 1, "Cyphastrea": 1},
    "Millepora": {"Millepora": 4},
    "Montipora": {"Acropora": 1, "Montipora": 3},
    "Favites": {"Favites": 2, "Goniastrea": 1, "Favia": 1},
    "Stylophora": {"Stylophora": 1},
    "Goniastrea": {"Goniastrea": 2, "Hydnophora": 1, "Favia": 1},
    "Hydnophora": {"Hydnophora": 2},
    "Echinopora": {"Porites": 1, "Echinopora": 2},
    "Pocillopora": {"Pocillopora": 2},
    "Dipsastraea": {"Favites": 1, "Dipsastraea": 1, "Favia": 2, "Acanthastrea": 1, "Leptastrea": 2},
    "Pavona": {"Pavona": 1},
    "Acanthastrea": {"Lobophyllia": 2},
    "Cyphastrea": {"Pavona": 1},
    "Seriatopora": {"Seriatopora": 2},
    "Cladiella": {"Acropora": 1, "Porites": 1, "Millepora": 1},
    "Litophyton": {"Acropora": 1},
    "Sinularia": {"Acropora": 1},
}


def check_expert_confusion(
    report: EvaluationReport,
    expected_cells: Mapping[str, Mapping[str, int]] = EXPERT_CSV_CELLS,
    expected: Mapping[str, int] = EXPERT_CSV_EXPECTED,
    strict_split: bool = False,
) -> list[CheckResult]:
    """Check D: counts, accuracy and every confusion cell against the real file.

    `expected_cells` is `{true: {predicted: count}}`; `expected` holds `rows`, `in_set`,
    `out_of_set`, `no_genus`, `unidentifiable`, `missing`, `correct` and `identifiable`.
    The rows without a genus are compared as a sum (D3); the split into unidentifiable and
    missing is information (D4) unless `strict_split`.
    """
    counts = report.status_counts_joined
    no_genus = counts["unidentifiable"] + counts["missing"]
    split = [counts["unidentifiable"], counts["missing"]]
    want_split = [expected["unidentifiable"], expected["missing"]]
    results = [
        equal(
            "D", "D1 expert rows with a prediction", report.coverage["n_joined"], expected["rows"]
        ),
        equal(
            "D",
            "D2 rows in the label set / soft corals",
            [counts["in_set"], counts["out_of_set"]],
            [expected["in_set"], expected["out_of_set"]],
        ),
        equal("D", "D3 rows without a genus", no_genus, expected["no_genus"]),
        equal("D", "D4 unidentifiable / missing", split, want_split, "from is_correct")
        if strict_split
        else info(
            "D",
            "D4 unidentifiable / missing (from is_correct)",
            split,
            want_split,
            "informational: the 16 / 2 split is from the memory notes, not yet confirmed",
        ),
        equal(
            "D",
            "D5 correct / in-set / in-set + soft corals",
            [
                report.accuracy_in_set["correct"],
                report.accuracy_in_set["total"],
                report.accuracy_in_plus_out["total"],
            ],
            [expected["correct"], expected["in_set"], expected["identifiable"]],
        ),
    ]
    results += _confusion_rows(report, expected_cells)
    return results


def _confusion_rows(
    report: EvaluationReport, expected_cells: Mapping[str, Mapping[str, int]]
) -> list[CheckResult]:
    expected_pairs: dict[tuple[str, str], int] = {
        (t, p): n for t, row in expected_cells.items() for p, n in row.items()
    }
    actual_pairs = report.confusion_cells()
    cells = sorted(set(expected_pairs) | set(actual_pairs))
    rows: list[CheckResult] = []
    wrong = 0
    for true, predicted in cells:
        want, got = expected_pairs.get((true, predicted), 0), actual_pairs.get((true, predicted), 0)
        wrong += want != got
        cell = equal("D", f"cell {true} -> {predicted}", got, want)
        cell.kind = "cell"
        rows.append(cell)
    summary = equal(
        "D", f"D6 confusion matrix cells equal ({len(cells)} compared)", wrong, 0, f"{wrong} differ"
    )
    return [summary, *rows]
