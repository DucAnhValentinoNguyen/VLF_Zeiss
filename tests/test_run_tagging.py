import pandas as pd
import pytest

from vlfz.report.aggregate import delta_view
from vlfz.report.pivot import _suspect
from vlfz.run_paths import result_tag, ssl_dir


def test_empty_and_tagged_paths_are_additive():
    assert ssl_dir("/tmp/ckpts", "lejepa", "imagenet", "gastronet") == \
        "/tmp/ckpts/lejepa_imagenet_gastronet_full"
    assert ssl_dir("/tmp/ckpts", "lejepa", "imagenet", "gastronet", run_tag="s60") == \
        "/tmp/ckpts/lejepa_imagenet_gastronet_full__s60"
    assert result_tag("imagenet_lejepa_gastronet_post") == "imagenet_lejepa_gastronet_post"


def test_tagged_post_delta_uses_shared_pre_baseline():
    common = dict(dataset="hyperkvasir", init="imagenet", task="hkv_tract",
                  protocol="knn", temp_scaled=False)
    df = pd.DataFrame([
        {**common, "objective": "none", "corpus": "none", "stage": "pre", "run_tag": "", "accuracy": .90},
        {**common, "objective": "lejepa", "corpus": "gastronet", "stage": "post", "run_tag": "s60", "accuracy": .80},
    ])
    out = delta_view(df)
    assert len(out) == 1 and out.iloc[0].d_accuracy == pytest.approx(-.10)


def test_tract_caveat_does_not_remove_generic_warning():
    assert _suspect({"dataset": "hyperkvasir", "task": "hkv_tract", "accuracy": .99}) == " †"
    assert _suspect({"dataset": "other", "task": "other", "accuracy": .99}) == " ⚠"
