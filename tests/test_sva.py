"""Parity with R's sva::ComBat on the simulated data of sva's own unit tests.

scripts/sva_reference.R writes the reference files in tests/data/sva.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import combatpy

DATA = Path(__file__).parent / "data" / "sva"
Y = np.genfromtxt(DATA / "y.csv", delimiter=",")
Y_MISSING = np.genfromtxt(DATA / "y_missing.csv", delimiter=",")
SAMPLES = pl.read_csv(DATA / "samples.csv")
BATCH = SAMPLES["batch"].to_numpy()
TREAT = SAMPLES["treat"].to_numpy()
KEYS = ["parametric", "covariates", "mean_only", "ref_batch", "missing"]
REFERENCE = {
    key: frame.drop(KEYS).to_numpy()
    for key, frame in pl.read_csv(
        DATA / "combat.csv.gz", infer_schema_length=None
    ).group_by(KEYS, maintain_order=True)
}


def case_id(key):
    parametric, covariates, mean_only, ref_batch, missing = key
    return "-".join(
        [
            "parametric" if parametric else "nonparametric",
            "treat" if covariates else "batch",
            "mean" if mean_only else "mean+variance",
            f"ref{ref_batch}" if ref_batch else "noref",
            "missing" if missing else "complete",
        ]
    )


@pytest.mark.parametrize(
    ("key", "expected"), REFERENCE.items(), ids=map(case_id, REFERENCE)
)
def test_matches_sva(key, expected):
    parametric, covariates, mean_only, ref_batch, missing = key
    adjusted = combatpy.combat(
        Y_MISSING if missing else Y,
        BATCH,
        TREAT if covariates else None,
        parametric=parametric,
        mean_only=mean_only,
        ref_batch=ref_batch,
    )
    np.testing.assert_allclose(adjusted, expected, rtol=1e-10, atol=1e-12)


# Row 1 of each case in sva's tests/testthat/test_combat_sim.R, which drops
# samples 2-5 for its single-sample batch.
SINGLE = np.r_[0, 5:15]


@pytest.mark.filterwarnings("ignore:a batch has a single sample")
@pytest.mark.parametrize(
    ("data", "batch", "covariates", "options", "expected"),
    [
        pytest.param(
            Y, BATCH, None, {},
            [6.5275687, 1.9434116, 3.3016615, 6.5353399, 5.8317968, -0.3314552, 3.9485709, 4.5368716, 4.8050618, 7.0422784, 4.2176928, 2.6528653, 2.3037212, 7.1688336, 3.1533711],
            id="batch",
        ),
        pytest.param(
            Y, BATCH, None, {"parametric": False},
            [6.5041159, 1.9798940, 3.3203856, 6.5117856, 5.8174409, -0.2101846, 3.9735703, 4.5486383, 4.8107961, 6.9976908, 4.1925793, 2.6502177, 2.3060862, 7.1013511, 3.1435378],
            id="batch-nonparametric",
        ),
        pytest.param(
            Y, BATCH, TREAT, {},
            [6.9550540, 1.3750580, 3.0128757, 6.9644248, 6.1160719, 0.8808531, 5.5188802, 6.0339045, 6.2686899, 8.2272474, 2.8499106, 1.3062056, 0.9617743, 5.8187037, 1.7999553],
            id="treat",
        ),
        pytest.param(
            Y, BATCH, TREAT, {"parametric": False},
            [6.9621810, 1.4557761, 2.9477604, 6.9707174, 6.1979029, 0.9018672, 5.5466543, 6.0315322, 6.2525748, 8.0964907, 2.7563817, 1.3122575, 0.9900446, 5.8403030, 1.7741566],
            id="treat-nonparametric",
        ),
        pytest.param(
            Y_MISSING, BATCH, TREAT, {},
            [7.0249052, 1.4302766, 3.0979753, np.nan, 6.1706164, 0.7848869, np.nan, 5.9650280, 6.2100799, 8.2542787, 2.8907924, 1.3319739, 0.9841704, 5.8422657, 1.8305576],
            id="treat-missing",
        ),
        pytest.param(
            Y, BATCH, TREAT, {"mean_only": True},
            [7.2030495, 1.6138619, 3.2698945, 7.2125245, 6.3547367, 0.5918231, 5.2032061, 5.8370527, 6.1260060, 8.5364265, 2.7343902, 1.1717875, 0.8231397, 5.6813352, 1.6715816],
            id="treat-mean",
        ),
        pytest.param(
            Y, BATCH, TREAT, {"mean_only": True, "parametric": False},
            [6.9822181, 1.3930305, 3.0490631, 6.9916932, 6.1339053, 0.7107136, 5.3220966, 5.9559432, 6.2448965, 8.6553171, 2.8420714, 1.2794687, 0.9308209, 5.7890163, 1.7792628],
            id="treat-mean-nonparametric",
        ),
        pytest.param(
            Y[:, SINGLE], BATCH[SINGLE], TREAT[SINGLE], {},
            [7.141462, 1.001405, 5.612788, 6.246634, 6.535587, 8.946008, 2.961106, 1.398503, 1.049856, 5.908051, 1.898298],
            id="treat-single-sample-batch",
        ),
    ],
)  # fmt: skip
def test_matches_sva_unit_tests(data, batch, covariates, options, expected):
    adjusted = combatpy.combat(data, batch, covariates, **options)
    np.testing.assert_allclose(adjusted[0], expected, atol=1e-6)
