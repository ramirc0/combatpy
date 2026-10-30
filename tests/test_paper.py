"""Real-world tests on the data of Chen et al. 2025.

Q. Chen et al., "Protein-level batch-effect correction enhances robustness in
MS-based proteomics", Nat Commun 16, 9735 (2025). The inputs, the paper's
sva::ComBat outputs and its source data come from Figshare files 57576382 and
57576448 (CC BY 4.0). The paper's code is at
github.com/QiaochuChen/proteomics-batch-effect-correction-benchmarking.

The reproduction tests rerun the paper's ComBat pipeline. They match its
published outputs and its SNR and CV values. The baseline tests pin how ComBat
performs when used as intended, so later changes have a reference. The first run
downloads about 14 MB into $XDG_CACHE_HOME/combatpy.
"""

import functools
import io
import os
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import pytest

import combatpy

pytestmark = [
    pytest.mark.paper,
    pytest.mark.filterwarnings("ignore:.*returned unadjusted:UserWarning"),
]

ARCHIVES = {"data": 57576382, "tables": 57576448}
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "combatpy"
DATASETS = {"simulated": "Simulated", "quartet": "Quartet"}
DESIGNS = {"balanced": "B", "confounded": "C"}
METHODS = ["maxlfq", "ibaq", "toppep3"]
CASES = [(d, s, m) for d in DATASETS for s in DESIGNS for m in METHODS]


class _RemoteFile(io.RawIOBase):
    """Seekable, read-only view of a remote file served with HTTP range requests."""

    def __init__(self, url):
        self.url, self.position = url, 0
        with urllib.request.urlopen(self._request(0, 0)) as response:
            self.size = int(response.headers["Content-Range"].rpartition("/")[2])

    def _request(self, start, end):
        return urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def readinto(self, buffer):
        end = min(self.position + len(buffer), self.size) - 1
        if end < self.position:
            return 0
        with urllib.request.urlopen(self._request(self.position, end)) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


@functools.cache
def _archive(name):
    url = f"https://ndownloader.figshare.com/files/{ARCHIVES[name]}"
    return zipfile.ZipFile(io.BufferedReader(_RemoteFile(url), 1 << 20))


def fetch(archive, member):
    """Return the cached path of a member of one of the paper's Figshare archives."""
    path = CACHE / member
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(_archive(archive).read(member))
        partial.replace(path)
    return path


def paper_combat(log_intensity, batch, groups):
    """Apply combatpy as combat_solve() in the paper's utils/BEC.R applies sva::ComBat."""
    intensity = np.where(log_intensity >= 0, np.exp(log_intensity), 0)
    normalized = intensity * 1e6 / intensity.sum(axis=0)
    adjusted = combatpy.combat(
        np.log(np.where(normalized > 0, normalized, 1)), batch, groups
    )
    return np.where(adjusted == 0, np.nan, adjusted)


class Case(NamedTuple):
    scenario: str
    method: str
    batch: np.ndarray
    groups: np.ndarray
    proteins: np.ndarray
    uncorrected: np.ndarray
    corrected: np.ndarray
    published: np.ndarray


@functools.cache
def load(dataset, design, method):
    """Load a protein-level matrix of the paper and correct it as the paper did."""
    root = f"data/expfiles/{dataset}/{design}"
    meta = pl.read_csv(fetch("data", f"{root}/meta.csv"))
    runs = meta["run_id"].to_list()

    def matrix(name):
        path = fetch("data", f"{root}/protein/{method}/expdata_{name}.csv")
        frame = pl.read_csv(path, infer_schema_length=None)
        return frame["protein"].to_numpy(), frame.select(
            pl.col(runs).cast(pl.Float64)
        ).to_numpy()

    proteins, uncorrected = matrix("log")
    published_proteins, published = matrix("combat")
    assert (proteins == published_proteins).all()
    batch, groups = meta["batch"].to_numpy(), meta["sample"].to_numpy()
    return Case(
        f"{DATASETS[dataset]}-{DESIGNS[design]}",
        method,
        batch,
        groups,
        proteins,
        uncorrected,
        paper_combat(uncorrected, batch, groups),
        published,
    )


@functools.cache
def source_data(name):
    """Load the protein-level rows of a source-data table of the paper."""
    table = pl.read_excel(fetch("tables", f"results/tables/{name}"))
    return table.filter(pl.col("data_level") == "Protein-level")


def published(name, case, corrected):
    """Return the source-data row of an uncorrected or ComBat-corrected case."""
    level = "Protein-corrected" if corrected else "Uncorrected"
    rows = source_data(name).filter(
        (pl.col("scenario") == case.scenario)
        & (pl.col("quant_method") == case.method)
        & (pl.col("correct_level") == level)
    )
    if corrected:
        rows = rows.filter(pl.col("correct_method") == "Combat")
    (row,) = rows.iter_rows(named=True)
    return row


def snr(values, groups):
    """Return the PCA-based signal-to-noise ratio of calculate_snr() in utils/PCA.R."""
    complete = values[~np.isnan(values).any(axis=1)]
    complete = complete[complete.std(axis=1) != 0]
    z = (complete.T - complete.mean(axis=1)) / complete.std(axis=1, ddof=1)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    scores = u[:, :2] * s[:2]
    # R's summary.prcomp rounds the proportions of variance to five decimals.
    weights = np.round(s[:2] ** 2 / np.sum(s**2), 5)
    distance = (scores[:, None] - scores[None]) ** 2 @ weights
    same = groups[:, None] == groups[None]
    intra = same & ~np.eye(len(groups), dtype=bool)
    return 10 * np.log10(distance[~same].mean() / distance[intra].mean())


def cv_summary(values, groups):
    """Summarize coefficients of variation per feature and group on the linear scale."""
    linear = np.exp(values)
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        cv = np.column_stack(
            [
                np.nanstd(x, axis=1, ddof=1) / np.nanmean(x, axis=1)
                for x in (linear[:, groups == g] for g in np.unique(groups))
            ]
        )
    valid = np.isfinite(cv)
    flat = cv[valid]
    return {
        "cv_median": np.median(flat),
        "cv_mean": flat.mean(),
        "cv_sd": flat.std(ddof=1),
        "feature_n": valid.any(axis=1).sum(),
        "total_n": valid.sum(),
    }


@functools.cache
def truth():
    """Return the simulated log fold changes, one row per protein."""
    mapping = pl.read_csv(fetch("data", "data/expfiles/simulated/mapping.csv"))
    return mapping.unique("protein", maintain_order=True)


def recovery(values, case):
    """Return the MCC and RC of estimated fold changes against the simulation truth.

    The paper's calculate_p() in utils/DEP.R indexes a named vector with a factor,
    which assigns group labels by position and scrambles them. Its published MCC
    and RC values are therefore not comparable. This version uses the correct
    labels and counts each protein and contrast once. Fold changes below 1.2
    count as no change.
    """
    row = {protein: i for i, protein in enumerate(case.proteins)}
    values = values[[row[protein] for protein in truth()["protein"]]]
    means = {
        g: values[:, case.groups == g].mean(axis=1) for g in np.unique(case.groups)
    }
    estimate = np.concatenate(
        [means[g] - means["group1"] for g in ("group2", "group3")]
    )
    true = np.concatenate(
        [truth()[f"{g}/group1"].to_numpy() for g in ("group2", "group3")]
    )
    called = np.where(np.abs(estimate) < np.log(1.2), 0, estimate)
    tp = np.sum(true * called > 0)
    tn = np.sum((true == 0) & (called == 0))
    fn = np.sum((true != 0) & (called == 0))
    fp = called.size - tp - tn - fn
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc, np.corrcoef(estimate, true)[0, 1]


def normalize(log_intensity):
    """Scale each sample to a total intensity of 1e6, with missing values as NaN.

    The MaxLFQ matrices mark missing values with -1e6, and the paper's
    preprocessing treats every negative log intensity as missing.
    """
    values = np.where(log_intensity < 0, np.nan, log_intensity)
    return values + np.log(1e6) - np.log(np.nansum(np.exp(values), axis=0))


CASE_IDS = ["-".join(case) for case in CASES]
LEVELS = pytest.mark.parametrize(
    "corrected", [False, True], ids=["uncorrected", "combat"]
)


@pytest.mark.parametrize("key", CASES, ids=CASE_IDS)
def test_matches_published_combat(key):
    case = load(*key)
    np.testing.assert_allclose(case.corrected, case.published, rtol=1e-8)


@LEVELS
@pytest.mark.parametrize("key", CASES, ids=CASE_IDS)
def test_reproduces_published_snr(key, corrected):
    case = load(*key)
    values = case.corrected if corrected else case.uncorrected
    expected = published("4-SNR-source_data.xlsx", case, corrected)
    assert snr(values, case.groups) == pytest.approx(expected["snr"], abs=1e-6)


@LEVELS
@pytest.mark.parametrize("key", CASES, ids=CASE_IDS)
def test_reproduces_published_cv(key, corrected):
    case = load(*key)
    values = case.corrected if corrected else case.uncorrected
    expected = published("1-CV-source_data.xlsx", case, corrected)
    actual = cv_summary(values, case.groups)
    assert actual == pytest.approx({key: expected[key] for key in actual}, rel=1e-9)


# (MCC uncorrected, MCC ComBat, RC uncorrected, RC ComBat). The simulation has
# small loading effects and large group effects that total-intensity scaling
# would distort, so ComBat runs on the log intensities directly.
RECOVERY = {
    ("balanced", "maxlfq"): (0.8801, 0.8895, 0.9945, 0.9945),
    ("balanced", "ibaq"): (0.5812, 0.5844, 0.9766, 0.9770),
    ("balanced", "toppep3"): (0.5242, 0.5167, 0.9694, 0.9697),
    ("confounded", "maxlfq"): (0.5017, 0.6847, 0.9670, 0.9883),
    ("confounded", "ibaq"): (0.3273, 0.4925, 0.8964, 0.9666),
    ("confounded", "toppep3"): (0.2752, 0.4399, 0.8455, 0.9573),
}


@pytest.mark.parametrize("key", RECOVERY, ids="-".join)
def test_fold_change_recovery_baseline(key):
    case = load("simulated", *key)
    adjusted = combatpy.combat(case.uncorrected, case.batch, case.groups)
    (mcc_before, rc_before), (mcc_after, rc_after) = (
        recovery(values, case) for values in (case.uncorrected, adjusted)
    )
    actual = (mcc_before, mcc_after, rc_before, rc_after)
    assert actual == pytest.approx(RECOVERY[key], abs=1e-4)


# (SNR uncorrected, SNR ComBat, median CV uncorrected, median CV ComBat). Quartet
# runs differ in loading within batches, so samples are scaled to equal total
# intensity first, as the paper does. Missing values stay NaN.
MISSING = {
    ("balanced", "maxlfq"): (0.0180, 13.8189, 0.4173, 0.2854),
    ("balanced", "ibaq"): (-0.2206, 2.4734, 0.5187, 0.3934),
    ("balanced", "toppep3"): (-0.1537, 12.6316, 0.3899, 0.3760),
    ("confounded", "maxlfq"): (0.3707, 21.1804, 0.3018, 0.1999),
    ("confounded", "ibaq"): (-0.1005, 9.9975, 0.3976, 0.2713),
    ("confounded", "toppep3"): (0.5812, 14.9554, 0.2697, 0.2147),
}


@pytest.mark.parametrize("key", MISSING, ids="-".join)
def test_missing_value_baseline(key):
    case = load("quartet", *key)
    normalized = normalize(case.uncorrected)
    adjusted = combatpy.combat(normalized, case.batch, case.groups)
    np.testing.assert_array_equal(np.isnan(adjusted), np.isnan(normalized))
    snr_before, snr_after = (snr(v, case.groups) for v in (normalized, adjusted))
    cv_before, cv_after = (
        cv_summary(v, case.groups)["cv_median"] for v in (normalized, adjusted)
    )
    actual = (snr_before, snr_after, cv_before, cv_after)
    assert actual == pytest.approx(MISSING[key], abs=1e-4)
