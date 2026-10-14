import hashlib
import json
import logging
from dataclasses import replace

import pytest

from coralid.genera import (
    PROMPT_FORMATS,
    GeneraError,
    PromptSet,
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
# dicts in the legacy notebooks (weak_classification.ipynb cell 14). They pin the
# YAML copy: any edit to a prompt, a genus key or the order changes the digest.
LEGACY_DIGESTS = {
    "taxonomy_v1": "1241ba257c9948b21a14d68b760767e40b466a05bc0eae4bc6c72341887e951a",
    "descriptive": "2ac0ea3ca7cb42f41ba5b8a15bb767c063239f36ba260afed9f53c3a5d831450",
    "common": "d678e7a93091d2861be882bd5f3b91ad5e62bb1b1cf8147737e7489149835410",
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
    # the soft corals of the expert test set (Cladiella 3 crops, Litophyton 1, Sinularia 1)
    assert cfg.soft_genera == ("Cladiella", "Sinularia", "Litophyton")
    assert cfg.label_set_genera() == HARD20
    assert cfg.known_genera() == HARD20 + list(cfg.soft_genera)
    with pytest.raises(GeneraError, match="Unknown label set 'soft_corals'"):
        cfg.genera("soft_corals")


@pytest.mark.parametrize("name", sorted(LEGACY_DIGESTS))
def test_prompt_sets_are_exact_copies_of_the_legacy_dicts(cfg, name):
    assert _digest(cfg.prompt_set(name).prompts) == LEGACY_DIGESTS[name]


def test_prompt_set_sizes_and_kinds(cfg):
    sizes = {n: len(p.prompts) for n, p in cfg.prompt_sets.items()}
    assert sizes == {
        "taxonomy_v1": 20,
        "descriptive": 20,
        "common": 20,
        "hybrid": 10,
    }
    assert {n: p.kind for n, p in cfg.prompt_sets.items()} == {
        "taxonomy_v1": "taxonomy",
        "descriptive": "descriptive",
        "common": "common",
        "hybrid": "hybrid",
    }
    assert [n for n, p in cfg.prompt_sets.items() if p.incomplete] == ["hybrid"]


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


def _with_a_second_taxonomy(cfg):
    """The shipped config plus a made-up taxonomy variant, to test choosing between two."""
    variant = PromptSet("taxonomy_test", "taxonomy", "a test", {g: f"Animalia {g}" for g in HARD20})
    return replace(cfg, prompt_sets={**cfg.prompt_sets, "taxonomy_test": variant})


def test_taxonomy_prompt_set_can_be_chosen(cfg):
    two = _with_a_second_taxonomy(cfg)
    (chosen,) = build_prompts("hard20", "taxonomy", prompt_set="taxonomy_test", config=two)
    assert chosen == [f"Animalia {g}" for g in HARD20]
    (default,) = build_prompts("hard20", "taxonomy", config=two)
    assert default == [cfg.prompt_set("taxonomy_v1").prompts[g] for g in HARD20]
    with pytest.raises(GeneraError, match="needs a taxonomy set"):
        build_prompts("hard20", "taxonomy", prompt_set="descriptive")
    with pytest.raises(GeneraError, match="needs a taxonomy set"):
        build_prompts("hard20", "ensemble", prompt_set="common")


def test_other_formats_ignore_the_taxonomy_prompt_set(cfg):
    two = _with_a_second_taxonomy(cfg)
    assert build_prompts(
        "hard20", "descriptive", prompt_set="taxonomy_test", config=two
    ) == build_prompts("hard20", "descriptive")


def test_prompt_set_names_per_format_and_family():
    assert prompt_set_names("taxonomy") == ["taxonomy_v1"]
    assert prompt_set_names("taxonomy", "taxonomy_test") == ["taxonomy_test"]
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
