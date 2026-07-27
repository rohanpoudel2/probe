from __future__ import annotations

from pathlib import Path

import pytest

from cli.common import load_yaml
from cli.run_ablation_suite import _merge, _validate_run_config


@pytest.mark.parametrize(
    "path",
    [
        Path("experiments/protocol/ablation_suite.yaml"),
        Path("experiments/controls/extended_ablation_suite.yaml"),
    ],
)
def test_registered_ablation_configs_have_reference_traffic(path: Path) -> None:
    config = load_yaml(path)
    for ablation in config["ablations"]:
        run_config = _merge(config["base"], ablation.get("overrides", {}))
        _validate_run_config(
            run_config,
            ablation_name=str(ablation["name"]),
        )
        assert set(str(run_config["views"]).split(",")).issubset(
            {"answer", "full_text"}
        )
        assert int(run_config["min_reference_groups"]) == 10_000


def test_ablation_validation_rejects_missing_reference_features() -> None:
    config = load_yaml("experiments/protocol/ablation_suite.yaml")
    run_config = _merge(config["base"], {})
    del run_config["models"][0]["reference_feature_dir"]
    with pytest.raises(ValueError, match="reference_feature_dir"):
        _validate_run_config(run_config, ablation_name="broken")
