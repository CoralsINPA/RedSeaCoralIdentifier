import hashlib
import json
import logging

import pytest

from coralid.genera import (
    PROMPT_FORMATS,
    GeneraError,
    build_prompts,
    default_genera_path,
    load_genera,
    prompt_set_names,
)

HARD20 = [
    "Platygyra",
    "Acropora",
    "Stylophora",
    "Millepora",
    "Favites",
    "Dipsastraea",
    "Favia",
    "Echinopora",
    "Hydnophora",
    "Porites",
    "Acanthastrea",
    "Cyphastrea",
    "Goniastrea",
    "Seriatopora",
    "Pocillopora",
    "Montipora",
    "Leptastrea",
    "Galaxea",
    "Lobophyllia",
    "Pavona",
]

# SHA-256 of json.dumps(list(prompts.items())) for each prompt set, computed once from the
# dicts in the legacy notebooks (weak_classification.ipynb cell 14, V2 cell 9). They pin the
# YAML copy: any edit to a prompt, a genus key or the order changes the digest.
LEGACY_DIGESTS = {
    "taxonomy_v1": "1241ba257c9948b21a14d68b760767e40b466a05bc0eae4bc6c72341887e951a",
    "descriptive": "2ac0ea3ca7cb42f41ba5b8a15bb767c063239f36ba260afed9f53c3a5d831450",
    "common": "d678e7a93091d2861be882bd5f3b91ad5e62bb1b1cf8147737e7489149835410",
    "taxonomy_v2": "f98889d735942519d94290072a5a9b28cb9769aeed5e35077dacd5a24ec2d418",
    "soft4_taxonomy": "1580bef038298b3401b9551e1643c706996b399a6c9383e5dec4e8b0bec8a33d",
    "hybrid": "fe52b225ee07096bacb2bf26729e6617c41fba95e5085104c4d445677ddae4c7",
}


def _digest(prompts):
    return hashlib.sha256(
        json.dumps(list(prompts.items()), ensure_ascii=False).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def cfg():
    return load_genera()


def test_hard20_has_the_exact_legacy_order(cfg):
    assert cfg.genera("hard20") == HARD20
    assert list(cfg.label_sets) == ["hard20"]


def test_soft_corals_are_known_but_not_a_label_set(cfg):
    assert cfg.soft_genera == ("Cladiella", "Sinularia", "Litophyton", "Sarcophyton")
    assert cfg.known_genera() == HARD20 + list(cfg.soft_genera)
    with pytest.raises(GeneraError, match="Unknown label set 'soft4'"):
        cfg.genera("soft4")


@pytest.mark.parametrize("name", sorted(LEGACY_DIGESTS))
def test_prompt_sets_are_exact_copies_of_the_legacy_dicts(cfg, name):
    assert _digest(cfg.prompt_set(name).prompts) == LEGACY_DIGESTS[name]


def test_prompt_set_sizes_and_kinds(cfg):
    sizes = {n: len(p.prompts) for n, p in cfg.prompt_sets.items()}
    assert sizes == {
        "taxonomy_v1": 20,
        "descriptive": 20,
        "common": 20,
        "taxonomy_v2": 20,
        "soft4_taxonomy": 4,
        "hybrid": 10,
    }
    assert {n: p.kind for n, p in cfg.prompt_sets.items()} == {
        "taxonomy_v1": "taxonomy",
        "descriptive": "descriptive",
        "common": "common",
        "taxonomy_v2": "taxonomy",
        "soft4_taxonomy": "taxonomy",
        "hybrid": "hybrid",
    }
    assert [n for n, p in cfg.prompt_sets.items() if p.incomplete] == ["hybrid"]


def test_the_two_taxonomy_variants_differ_where_the_legacy_ones_did(cfg):
    """WC files Favites under Faviidae, V2 under Merulinidae (plan section 2.1 #6)."""
    v1 = cfg.prompt_set("taxonomy_v1").prompts
    v2 = cfg.prompt_set("taxonomy_v2").prompts
    assert v1["Favites"].endswith("Faviidae Favites")
    assert v2["Favites"].endswith("Merulinidae Favites")
    assert v2["Leptastrea"].endswith("Incertae Leptastrea")
    assert v1["Acropora"] == v2["Acropora"]


def test_hybrid_with_hard20_raises_a_clear_error():
    with pytest.raises(GeneraError) as err:
        build_prompts("hard20", "hybrid")
    message = str(err.value)
    assert "'hybrid'" in message and "10 of the 20 genera" in message
    assert "Pavona" in message and "incomplete" in message


def test_build_prompts_follow_label_set_order(cfg):
    (prompts,) = build_prompts("hard20", "taxonomy")
    assert prompts == [cfg.prompt_set("taxonomy_v1").prompts[g] for g in HARD20]
    (desc,) = build_prompts("hard20", "descriptive")
    assert desc[0].startswith("underwater photo of Platygyra")
    (common,) = build_prompts("hard20", "common")
    assert common[1] == "staghorn coral Acropora"


def test_taxonomy_prompt_set_can_be_chosen(cfg):
    (v2,) = build_prompts("hard20", "taxonomy", prompt_set="taxonomy_v2")
    assert v2 == [cfg.prompt_set("taxonomy_v2").prompts[g] for g in HARD20]
    with pytest.raises(GeneraError, match="needs a taxonomy set"):
        build_prompts("hard20", "taxonomy", prompt_set="descriptive")
    with pytest.raises(GeneraError, match="needs a taxonomy set"):
        build_prompts("hard20", "ensemble", prompt_set="common")


def test_other_formats_ignore_the_taxonomy_prompt_set(cfg):
    assert build_prompts("hard20", "descriptive", prompt_set="taxonomy_v2") == build_prompts(
        "hard20", "descriptive"
    )


def test_prompt_set_names_per_format_and_family():
    assert prompt_set_names("taxonomy") == ["taxonomy_v1"]
    assert prompt_set_names("taxonomy", "taxonomy_v2") == ["taxonomy_v2"]
    assert prompt_set_names("descriptive") == ["descriptive"]
    assert prompt_set_names("common") == ["common"]
    assert prompt_set_names("hybrid") == ["hybrid"]
    assert prompt_set_names("ensemble") == ["taxonomy_v1", "descriptive", "common"]
    # SigLIP: ensemble skips taxonomy, taxonomy becomes descriptive
    assert prompt_set_names("ensemble", family="siglip") == ["descriptive", "common"]
    assert prompt_set_names("descriptive", family="siglip") == ["descriptive"]


def test_siglip_taxonomy_is_replaced_by_descriptive_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="coralid.genera"):
        assert prompt_set_names("taxonomy", family="siglip") == ["descriptive"]
    assert "switching to descriptive" in caplog.text
    # a SigLIP taxonomy request never trips over the taxonomy-kind check
    assert build_prompts("hard20", "taxonomy", prompt_set="descriptive", family="siglip")


def test_ensemble_builds_three_prompt_lists_for_open_clip_and_two_for_siglip():
    assert [len(p) for p in build_prompts("hard20", "ensemble")] == [20, 20, 20]
    assert [len(p) for p in build_prompts("hard20", "ensemble", family="siglip")] == [20, 20]


def test_unknown_names_raise():
    assert set(PROMPT_FORMATS) == {"taxonomy", "descriptive", "common", "hybrid", "ensemble"}
    with pytest.raises(GeneraError, match="Unknown prompt format"):
        build_prompts("hard20", "poetry")
    with pytest.raises(GeneraError, match="Unknown label set"):
        build_prompts("soft99", "taxonomy")
    with pytest.raises(GeneraError, match="Unknown prompt set"):
        build_prompts("hard20", "taxonomy", prompt_set="taxonomy_v9")
    with pytest.raises(GeneraError, match="Unknown encoder family"):
        prompt_set_names("taxonomy", family="resnet")


def test_load_genera_from_an_explicit_path_and_missing_file(tmp_path):
    custom = tmp_path / "g.yaml"
    custom.write_text(
        "label_sets:\n  tiny:\n    genera: [A, B]\n"
        "prompt_sets:\n  hybrid:\n    kind: hybrid\n    incomplete: true\n    prompts: {A: a}\n",
        encoding="utf-8",
    )
    tiny = load_genera(custom)
    assert tiny.genera("tiny") == ["A", "B"]
    with pytest.raises(GeneraError, match="missing: B"):
        build_prompts("tiny", "hybrid", config=tiny)
    with pytest.raises(FileNotFoundError, match="pass the path"):
        load_genera(tmp_path / "nope.yaml")


def test_default_path_points_into_the_repository():
    assert default_genera_path().name == "genera.yaml"
    assert default_genera_path().parent.name == "configs"
