"""Behavior of combatpy.combat that sva parity does not cover."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

import combatpy

BATCH = np.repeat(["a", "b", "c"], 4)
GROUP = np.tile(["x", "y"], 6)
DATA = np.random.default_rng(0).normal(size=(40, 12)) + np.repeat([0.0, 2.0, 4.0], 4)


def test_pandas_frames_keep_their_labels():
    frame = pd.DataFrame(
        DATA, index=[f"f{i}" for i in range(40)], columns=list("ABCDEFGHIJKL")
    )
    adjusted = combatpy.combat(frame, BATCH)
    assert isinstance(adjusted, pd.DataFrame)
    assert adjusted.index.equals(frame.index)
    assert adjusted.columns.equals(frame.columns)
    np.testing.assert_allclose(adjusted, combatpy.combat(DATA, BATCH), rtol=1e-12)


def test_polars_frames_keep_their_columns():
    frame = pl.DataFrame(DATA, schema=list("ABCDEFGHIJKL"), orient="row")
    adjusted = combatpy.combat(frame, pl.Series(BATCH))
    assert isinstance(adjusted, pl.DataFrame)
    assert adjusted.columns == frame.columns
    np.testing.assert_allclose(
        adjusted.to_numpy(), combatpy.combat(DATA, BATCH), rtol=1e-12
    )


@pytest.mark.parametrize(
    "covariates",
    [
        (GROUP == "y").astype(float),
        np.column_stack([np.ones(12), GROUP == "y"]),
        {"group": GROUP},
        pl.DataFrame({"group": GROUP}),
    ],
    ids=["one-hot", "model-matrix", "dict", "polars"],
)
def test_covariate_encodings_agree(covariates):
    expected = combatpy.combat(DATA, BATCH, GROUP)
    np.testing.assert_allclose(
        combatpy.combat(DATA, BATCH, covariates), expected, rtol=1e-12
    )


def test_mixed_covariates_match_their_design():
    age = np.linspace(20, 70, 12)
    design = np.column_stack([GROUP == "y", age])
    expected = combatpy.combat(DATA, BATCH, design)
    adjusted = combatpy.combat(DATA, BATCH, pd.DataFrame({"group": GROUP, "age": age}))
    np.testing.assert_allclose(adjusted, expected, rtol=1e-12)


def test_batch_label_type_does_not_matter():
    expected = combatpy.combat(DATA, BATCH)
    np.testing.assert_array_equal(
        combatpy.combat(DATA, np.repeat([7, 8, 9], 4)), expected
    )


def test_reference_batch_is_unchanged():
    adjusted = combatpy.combat(DATA, BATCH, ref_batch="b")
    np.testing.assert_array_equal(adjusted[:, BATCH == "b"], DATA[:, BATCH == "b"])
    assert not np.allclose(adjusted[:, BATCH == "a"], DATA[:, BATCH == "a"])


def test_missing_values_stay_missing():
    data = DATA.copy()
    data[3, 0] = data[7, 5] = np.nan
    adjusted = combatpy.combat(data, BATCH)
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(data))


@pytest.mark.parametrize("parametric", [True, False])
def test_unestimable_features_pass_through(parametric):
    data = DATA.copy()
    data[0, BATCH == "a"] = 5.0
    data[1, np.flatnonzero(BATCH == "b")[1:]] = np.nan
    data[2, GROUP == "y"] = np.nan
    with pytest.warns(UserWarning, match="3 of 40 features"):
        adjusted = combatpy.combat(data, BATCH, GROUP, parametric=parametric)
    np.testing.assert_array_equal(adjusted[:3], data[:3])
    assert np.isfinite(adjusted[3:]).all()
    assert not np.allclose(adjusted[3:], data[3:])


def test_single_sample_batch_adjusts_means_only():
    batch = np.array(["a"] + ["b"] * 5 + ["c"] * 6)
    with pytest.warns(UserWarning, match="single sample"):
        adjusted = combatpy.combat(DATA, batch)
    np.testing.assert_array_equal(
        adjusted, combatpy.combat(DATA, batch, mean_only=True)
    )


@pytest.mark.parametrize(
    ("args", "options", "message"),
    [
        ((DATA[0], BATCH), {}, "2-D"),
        ((DATA, BATCH[:-1]), {}, "features × samples"),
        ((DATA, np.zeros(12)), {}, "two batches"),
        ((DATA, BATCH), {"ref_batch": "z"}, "ref_batch"),
        ((DATA, BATCH, GROUP[:-1]), {}, "rows"),
        ((DATA, BATCH, np.r_[np.nan, np.ones(11)]), {}, "missing"),
        ((DATA, BATCH, BATCH), {}, "confounded"),
    ],
    ids=[
        "1-D",
        "transposed",
        "one-batch",
        "ref-batch",
        "covariate-rows",
        "covariate-nan",
        "confounded",
    ],
)
def test_invalid_input_raises(args, options, message):
    with pytest.raises(ValueError, match=message):
        combatpy.combat(*args, **options)
