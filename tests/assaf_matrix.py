"""The authoritative confusion matrix of Assaf's review, as plain data for the tests."""

# The authoritative confusion matrix of Assaf's review, sparse: true genus -> predicted -> count
# (docs/agent_memory/assaf_review_confusion_matrix.md). 62 identifiable crops, 39 on the diagonal.
ASSAF = {
    "Acropora": {"Acropora": 5},
    "Platygyra": {"Platygyra": 5, "Galaxea": 1},
    "Porites": {"Porites": 5, "Dipsastraea": 1, "Cyphastrea": 1},
    "Millepora": {"Millepora": 4},
    "Montipora": {"Acropora": 1, "Montipora": 3},
    "Favites": {"Favites": 3, "Goniastrea": 1, "Favia": 1},
    "Stylophora": {"Stylophora": 3},
    "Goniastrea": {"Goniastrea": 2, "Hydnophora": 1, "Favia": 1},
    "Hydnophora": {"Hydnophora": 2},
    "Echinopora": {"Porites": 1, "Echinopora": 2},
    "Pocillopora": {"Pocillopora": 2},
    "Dipsastraea": {"Favites": 1, "Dipsastraea": 1, "Favia": 2, "Acanthastrea": 1, "Leptastrea": 2},
    "Pavona": {"Pavona": 1},
    "Favia": {"Favia": 1},
    "Acanthastrea": {"Lobophyllia": 2},
    "Cyphastrea": {"Pavona": 1},
    "Cladiella": {"Acropora": 1, "Porites": 1, "Millepora": 1},
    "Litophyton": {"Acropora": 1},
    "Sinularia": {"Acropora": 1},
}
ASSAF_RECALL = {"Acropora": 100, "Platygyra": 83, "Porites": 71, "Millepora": 100, "Montipora": 75}
ASSAF_RECALL |= {"Favites": 60, "Stylophora": 100, "Goniastrea": 50, "Hydnophora": 100}
ASSAF_RECALL |= {"Echinopora": 67, "Pocillopora": 100, "Dipsastraea": 14, "Pavona": 100}
ASSAF_RECALL |= {"Favia": 100, "Acanthastrea": 0, "Cyphastrea": 0, "Cladiella": 0}
ASSAF_PRECISION = {"Acropora": 56, "Platygyra": 100, "Porites": 71, "Millepora": 80}
ASSAF_PRECISION |= {"Montipora": 100, "Favites": 75, "Stylophora": 100, "Goniastrea": 67}
ASSAF_PRECISION |= {"Hydnophora": 67, "Echinopora": 100, "Pocillopora": 100, "Dipsastraea": 50}
ASSAF_PRECISION |= {"Pavona": 50, "Favia": 20, "Acanthastrea": 0, "Cyphastrea": 0}
ASSAF_PRECISION |= {"Lobophyllia": 0, "Leptastrea": 0, "Galaxea": 0}
