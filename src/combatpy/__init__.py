"""ComBat batch-effect correction, ported from R's ``sva::ComBat``."""

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["combat"]


def combat(
    data: Any,
    batch: ArrayLike,
    covariates: ArrayLike | Mapping[str, ArrayLike] | None = None,
    *,
    parametric: bool = True,
    mean_only: bool = False,
    ref_batch: Any = None,
) -> Any:
    """Remove batch effects with ComBat's empirical Bayes method.

    ComBat fits an additive and a multiplicative effect for each batch and
    feature, shrinks both toward priors pooled across features, and removes
    them [1]_. Results match R's ``sva::ComBat``.

    Parameters
    ----------
    data : array_like, shape (n_features, n_samples)
        Normalized log-scale values with features in rows and samples in
        columns. NaN marks a missing value. A pandas or polars DataFrame works.
    batch : array_like, shape (n_samples,)
        Batch label of each sample, in column order.
    covariates : array_like, dict or DataFrame, optional
        Variables whose effects are kept, with one row per sample. Numeric
        columns enter the model as they are. Other columns are one-hot
        encoded. Columns of ones are ignored, so an R-style model matrix
        works too.
    parametric : bool, default True
        Use parametric priors. Non-parametric priors take time quadratic in
        the number of features.
    mean_only : bool, default False
        Adjust batch means only and keep batch variances.
    ref_batch : optional
        Label of a batch to keep unchanged. The other batches are adjusted
        toward it.

    Returns
    -------
    numpy.ndarray or DataFrame
        Adjusted values with the shape of `data`. A pandas or polars DataFrame
        comes back as the same type with the same labels.

    Raises
    ------
    ValueError
        If `data` is not 2-D or `batch` does not have one label per column.
        If there are fewer than two batches or `ref_batch` is not one of them.
        If `covariates` has missing values or the wrong number of rows.
        If `covariates` is confounded with `batch`.

    Warns
    -----
    UserWarning
        If a batch has a single sample, which forces `mean_only`. Also if
        some features cannot be estimated, for example because they are
        constant within a batch or have too few observed values in one.
        Those features come back unchanged.

    References
    ----------
    .. [1] W. E. Johnson, C. Li and A. Rabinovic, "Adjusting batch effects in
       microarray expression data using empirical Bayes methods",
       Biostatistics 8, 118-127 (2007).

    Examples
    --------
    >>> import numpy as np
    >>> import combatpy
    >>> rng = np.random.default_rng(0)
    >>> batch = np.repeat(["a", "b"], 4)
    >>> data = rng.normal(size=(100, 8)) + np.where(batch == "b", 3.0, 0.0)
    >>> corrected = combatpy.combat(data, batch)
    >>> corrected.shape
    (100, 8)
    >>> shift = corrected[:, batch == "b"].mean() - corrected[:, batch == "a"].mean()
    >>> bool(abs(shift) < 0.1)
    True
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D array of features × samples")
    levels, batch_index = np.unique(np.ravel(batch), return_inverse=True)
    if batch_index.size != values.shape[1]:
        raise ValueError(
            f"batch has {batch_index.size} labels but data has {values.shape[1]} "
            "columns; data must be features × samples"
        )
    if levels.size < 2:
        raise ValueError("ComBat needs at least two batches")
    if not mean_only and (np.bincount(batch_index) == 1).any():
        warnings.warn(
            "a batch has a single sample, so only batch means are adjusted",
            stacklevel=2,
        )
        mean_only = True

    batch_design = np.eye(levels.size)[batch_index]
    ref = None
    if ref_batch is not None:
        matches = np.flatnonzero(levels == ref_batch)
        if matches.size == 0:
            raise ValueError(f"ref_batch {ref_batch!r} is not a batch label")
        ref = matches[0]
        batch_design[:, ref] = 1
    design = np.hstack([batch_design, _covariate_design(covariates, values.shape[1])])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("covariates are confounded with batch or with each other")

    keep = _estimable(values, design, batch_index, 1 if mean_only else 2)
    if not keep.all():
        warnings.warn(
            f"{keep.size - keep.sum()} of {keep.size} features cannot be estimated "
            "and are returned unadjusted",
            stacklevel=2,
        )
    adjusted = values.copy()
    if keep.any():
        adjusted[keep] = _adjust(
            values[keep], design, batch_index, ref, parametric, mean_only
        )
    if hasattr(data, "index") and hasattr(data, "columns"):  # pandas
        return type(data)(adjusted, index=data.index, columns=data.columns)
    if hasattr(data, "schema"):  # polars
        return type(data)(adjusted, schema=data.columns, orient="row")
    return adjusted


def _covariate_design(covariates, n_samples):
    """Encode covariates as model-matrix columns with one row per sample."""
    if covariates is None:
        return np.empty((n_samples, 0))
    if hasattr(covariates, "columns"):
        columns = [covariates[name] for name in covariates.columns]
    elif isinstance(covariates, Mapping):
        columns = list(covariates.values())
    else:
        array = np.asarray(covariates)
        columns = list(array.T) if array.ndim == 2 else [array]
    blocks = []
    for column in map(np.asarray, columns):
        if column.dtype.kind in "biuf":
            blocks.append(column.reshape(-1, 1).astype(float))
        else:
            levels, codes = np.unique(column, return_inverse=True)
            blocks.append(np.eye(levels.size)[codes, 1:])
    design = np.hstack(blocks)
    if design.shape[0] != n_samples:
        raise ValueError(
            f"covariates have {design.shape[0]} rows but data has {n_samples} samples"
        )
    if np.isnan(design).any():
        raise ValueError("covariates must not have missing values")
    return design[:, ~(design == 1).all(axis=0)]


def _gram(design, observed):
    """Return each feature's Gram matrix over its observed samples."""
    p = design.shape[1]
    outer = (design[:, :, None] * design[:, None, :]).reshape(len(design), p * p)
    return (observed @ outer).reshape(-1, p, p)


def _fit(values, design, observed):
    """Return each feature's least-squares coefficients over its observed samples."""
    rhs = np.where(observed, values, 0) @ design
    return np.linalg.solve(_gram(design, observed), rhs[..., None])[..., 0]


def _estimable(values, design, batch_index, min_observed):
    """Flag features whose batch and covariate effects can be estimated."""
    observed = ~np.isnan(values)
    indicator = np.eye(batch_index.max() + 1)[batch_index]
    enough = ((observed @ indicator) >= min_observed).all(axis=1)
    constant = np.zeros(len(values), dtype=bool)
    for b in np.flatnonzero(indicator.sum(axis=0) > 1):
        constant |= values[:, batch_index == b].var(axis=1) == 0
    rank = np.linalg.matrix_rank(_gram(design, observed))
    return enough & ~constant & (rank == design.shape[1])


def _adjust(values, design, batch_index, ref, parametric, mean_only):
    """Run ComBat on features that are all estimable."""
    observed = ~np.isnan(values)
    sizes = np.bincount(batch_index)
    n_batch = sizes.size
    beta = _fit(values, design, observed)
    if ref is None:
        grand_mean = beta[:, :n_batch] @ (sizes / sizes.sum())
        pooled = np.ones(batch_index.size, dtype=bool)
    else:
        grand_mean = beta[:, ref]
        pooled = batch_index == ref
    residuals = (values - beta @ design.T)[:, pooled]
    if observed.all():
        var_pooled = np.mean(residuals**2, axis=1)
    else:
        # sva switches to the n - 1 variance whenever any value is missing.
        var_pooled = np.nanvar(residuals, axis=1, ddof=1)
    stand_mean = grand_mean[:, None] + beta[:, n_batch:] @ design[:, n_batch:].T
    scale = np.sqrt(var_pooled)[:, None]
    standardized = (values - stand_mean) / scale

    batch_design = design[:, :n_batch]
    gamma_hat = _fit(standardized, batch_design, observed)
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.ones_like(gamma_hat)
    for b in range(n_batch):
        s = standardized[:, batch_index == b]
        g = gamma_hat[:, b]
        delta_hat = np.ones(len(s)) if mean_only else np.nanvar(s, axis=1, ddof=1)
        if not parametric:
            gamma_star[:, b], delta_star[:, b] = _nonparametric(s, g, delta_hat)
        elif mean_only:
            t2 = g.var(ddof=1)
            gamma_star[:, b] = (t2 * g + g.mean()) / (t2 + 1)
        else:
            gamma_star[:, b], delta_star[:, b] = _parametric(s, g, delta_hat)
    if ref is not None:
        gamma_star[:, ref] = 0
        delta_star[:, ref] = 1

    adjusted = standardized - gamma_star @ batch_design.T
    adjusted = adjusted / np.sqrt(delta_star[:, batch_index]) * scale + stand_mean
    if ref is not None:
        adjusted[:, pooled] = values[:, pooled]
    return adjusted


def _parametric(s, gamma_hat, delta_hat):
    """Return posterior batch effects under normal and inverse-gamma priors."""
    gamma_bar, t2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    m, v = delta_hat.mean(), delta_hat.var(ddof=1)
    a, b = (2 * v + m**2) / v, (m * v + m**3) / v
    n = np.count_nonzero(~np.isnan(s), axis=1)
    gamma_old, delta_old = gamma_hat, delta_hat
    change = 1.0
    while change > 1e-4:
        gamma_new = (t2 * n * gamma_hat + delta_old * gamma_bar) / (t2 * n + delta_old)
        sum2 = np.nansum((s - gamma_new[:, None]) ** 2, axis=1)
        delta_new = (sum2 / 2 + b) / (n / 2 + a - 1)
        # sva divides by signed values, so negative effects never block convergence.
        with np.errstate(divide="ignore", invalid="ignore"):
            change = max(
                np.max(np.abs(gamma_new - gamma_old) / gamma_old),
                np.max(np.abs(delta_new - delta_old) / delta_old),
            )
        gamma_old, delta_old = gamma_new, delta_new
    return gamma_new, delta_new


def _nonparametric(s, gamma_hat, delta_hat):
    """Return posterior batch effects weighted by the other features' likelihoods."""
    n = np.count_nonzero(~np.isnan(s), axis=1)
    mean = np.nanmean(s, axis=1)
    ss = np.nansum((s - mean[:, None]) ** 2, axis=1)
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(delta_hat)
    step = max(1, 2**20 // len(s))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_norm = np.log(2 * np.pi * delta_hat)
        for start in range(0, len(s), step):
            rows = np.arange(start, min(start + step, len(s)))
            sum2 = ss[rows, None] + n[rows, None] * (mean[rows, None] - gamma_hat) ** 2
            log_lik = -(n[rows, None] * log_norm + sum2 / delta_hat) / 2
            # Like sva, leave each feature out of its own prior and zero the NaN
            # likelihoods that zero variances produce.
            log_lik[np.arange(rows.size), rows] = -np.inf
            log_lik[np.isnan(log_lik)] = -np.inf
            weight = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))
            total = weight.sum(axis=1)
            gamma_star[rows] = weight @ gamma_hat / total
            delta_star[rows] = weight @ delta_hat / total
    return gamma_star, delta_star
