"""ComBat batch-effect correction, ported from R's ``sva::ComBat``."""

import math
import warnings
from collections.abc import Mapping
from typing import Any

import numba
import numpy as np
from numba.core import types
from numba.extending import intrinsic
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

    # The kernels run on samples x features, a view of an F-ordered input.
    samples = np.ascontiguousarray(values.T)
    observed = ~np.isnan(samples)
    patterns, pattern = _patterns(observed.T)
    gram = _gram(design, patterns)
    keep, factor, factored = _estimable(
        samples, batch_index, 1 if mean_only else 2, patterns, pattern, gram
    )
    if keep.all():
        adjusted = np.empty_like(samples)
        _adjust(
            samples,
            observed,
            gram,
            factor,
            factored,
            pattern,
            design,
            batch_index,
            ref,
            parametric,
            mean_only,
            adjusted,
            None,
        )
    else:
        warnings.warn(
            f"{keep.size - keep.sum()} of {keep.size} features cannot be estimated "
            "and are returned unadjusted",
            stacklevel=2,
        )
        adjusted = samples.copy()
        if keep.any():
            _adjust(
                samples.compress(keep, axis=1),
                observed.compress(keep, axis=1),
                gram,
                factor,
                factored,
                pattern[keep],
                design,
                batch_index,
                ref,
                parametric,
                mean_only,
                adjusted,
                np.flatnonzero(keep),
            )
    adjusted = adjusted.T
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


_BITS = 1 << np.arange(7, -1, -1, dtype=np.uint8)


def _patterns(observed):
    """Return the distinct rows of a mask and each row's index among them."""
    if observed.all():
        return observed[:1], np.zeros(len(observed), np.intp)
    # np.packbits reads along the packed axis, the strided one here, so pack
    # whole sample rows of `observed.T` instead: the same bits, 12x faster.
    rows = observed.T.view(np.uint8)
    if len(rows) % 8:
        rows = np.vstack([rows, np.zeros((-len(rows) % 8, rows.shape[1]), np.uint8)])
    packed = np.einsum("bjf,j->bf", rows.reshape(-1, 8, rows.shape[1]), _BITS)
    packed = np.ascontiguousarray(packed.T)
    _, first, pattern = np.unique(
        packed.view(f"V{packed.shape[1]}")[:, 0], return_index=True, return_inverse=True
    )
    return observed[first], pattern


def _gram(design, observed):
    """Return each mask row's Gram matrix over its observed samples."""
    p = design.shape[1]
    outer = (design[:, :, None] * design[:, None, :]).reshape(len(design), p * p)
    return (observed @ outer).reshape(-1, p, p)


def _aligned_zeros(shape, dtype=float):
    """Return a zeroed C array whose data start on a 64-byte boundary.

    Kernels take their buffers from here. numba aligns the arrays it
    allocates, while numpy's often start 16 bytes off a 32-byte boundary,
    which slowed `_weigh`'s vector loads by 8%.
    """
    shape = shape if isinstance(shape, tuple) else (shape,)
    nbytes = math.prod(shape) * np.dtype(dtype).itemsize
    raw = np.zeros(nbytes + 64, np.uint8)
    start = -raw.ctypes.data % 64
    return raw[start : start + nbytes].view(dtype).reshape(shape)


def _fit(values, design, observed, gram, factor, factored, pattern):
    """Return each feature's least-squares coefficients over its observed samples.

    `values` is samples x features. `factor` and `factored` come from
    `_cholesky` on the pattern Grams `gram`.
    """
    filled = values if observed.all() else np.where(observed, values, 0)
    rhs = filled.T @ design
    beta = _aligned_zeros(rhs.shape)
    _cholesky_solve(factor, pattern, rhs, beta)
    solved = factored[pattern]
    if not solved.all():
        beta[~solved] = np.linalg.solve(gram[pattern[~solved]], rhs[~solved, :, None])[
            ..., 0
        ]
    return beta


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _cholesky(gram, which, factor, factored, floor, x):
    """Factor the Grams flagged in `which` and bound their smallest eigenvalues.

    Writes the lower Cholesky factors into `factor`, sets `factored` where a
    Gram factored, and writes ``1 / ||L^-1||_F^2``, a lower bound on the
    smallest eigenvalue of each one that did, into `floor`. `factored` must
    start False. Factors and bounds of the others are undefined. `x` is
    scratch of length p.
    """
    n, p = gram.shape[0], gram.shape[1]
    for g in range(n):
        if not which[g]:
            continue
        lower = factor[g]
        factored[g] = True
        for j in range(p):
            pivot = gram[g, j, j]
            for m in range(j):
                pivot -= lower[j, m] * lower[j, m]
            if not pivot > 0:
                factored[g] = False
                break
            lower[j, j] = np.sqrt(pivot)
            for r in range(j + 1, p):
                t = gram[g, r, j]
                for m in range(j):
                    t -= lower[r, m] * lower[j, m]
                lower[r, j] = t / lower[j, j]
        if not factored[g]:
            continue
        total = 0.0
        for k in range(p):
            for j in range(k, p):
                t = 1.0 if j == k else 0.0
                for m in range(k, j):
                    t -= lower[j, m] * x[m]
                x[j] = t / lower[j, j]
                total += x[j] * x[j]
        floor[g] = 1 / total


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _cholesky_solve(factor, pattern, rhs, beta):
    """Solve each feature's normal equations with its pattern's Cholesky
    factor, into `beta`."""
    n_features, p = rhs.shape
    for i in range(n_features):
        lower = factor[pattern[i]]
        x = beta[i]
        for j in range(p):
            t = rhs[i, j]
            for m in range(j):
                t -= lower[j, m] * x[m]
            x[j] = t / lower[j, j]
        for j in range(p - 1, -1, -1):
            t = x[j]
            for m in range(j + 1, p):
                t -= lower[m, j] * x[m]
            x[j] = t / lower[j, j]


def _batch_fit(means, ref):
    """Return `_fit` coefficients for a design of batch columns only.

    Each batch's coefficient is its observed mean, given in `means`. With
    `ref`, the ref column is all ones, so it takes the ref batch's mean and
    the others are offsets from it.
    """
    if ref is None:
        return means
    coef = means - means[:, [ref]]
    coef[:, ref] = means[:, ref]
    return coef


def _estimable(values, batch_index, min_observed, patterns, pattern, gram):
    """Flag features whose batch and covariate effects can be estimated.

    The counts and the Gram depend only on which samples are observed, so
    they are checked once per distinct pattern. Also returns `_cholesky` of
    the Grams that pass the counts, which `_fit` reuses.
    """
    indicator = np.eye(batch_index.max() + 1)[batch_index]
    enough = ((patterns @ indicator) >= min_observed).all(axis=1)
    p = gram.shape[-1]
    factor = _aligned_zeros(gram.shape)
    factored = _aligned_zeros(len(gram), bool)
    floor = _aligned_zeros(len(gram))
    _cholesky(gram, enough, factor, factored, floor, _aligned_zeros(p))
    # matrix_rank's tolerance is p * eps times the largest eigenvalue, at most
    # the trace. A floor 1e3 times above that bound leaves eigvalsh's rounding
    # no room to report a deficient rank, so only the other Grams are ranked.
    certain = factored & (
        floor > 1e3 * p * np.finfo(float).eps * np.trace(gram, axis1=1, axis2=2)
    )
    unsure = enough & ~certain
    if unsure.any():
        enough[unsure] = np.linalg.matrix_rank(gram[unsure], hermitian=True) == p
    order = np.argsort(batch_index, kind="stable")
    starts = np.concatenate(([0], np.cumsum(np.bincount(batch_index))))
    constant = _aligned_zeros(values.shape[1], bool)
    _constant(values, order, starts, constant)
    return enough[pattern] & ~constant, factor, factored


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _constant(values, order, starts, constant):
    """Flag features whose variance is zero within some batch of two or more
    samples, in `constant`, which must start False.

    `values` is samples x features, and batch b's samples are
    ``order[starts[b]:starts[b + 1]]``. The decision matches numpy's
    ``var(axis=1) == 0`` on each batch's column copy of the features x samples
    matrix. Values more than 1e-150 apart give a squared deviation above the
    underflow range, so only features within that margin repeat numpy's sums
    in its order.
    """
    for b in range(starts.size - 1):
        start, stop = starts[b], starts[b + 1]
        if stop - start < 2:
            continue
        for i in range(values.shape[1]):
            if constant[i]:
                continue
            first = values[order[start], i]
            near = True
            for k in range(start + 1, stop):
                # NaN and inf fail this test, and give numpy a NaN variance.
                if not abs(values[order[k], i] - first) <= 1e-150:
                    near = False
                    break
            if not near:
                continue
            total = 0.0
            for k in range(start, stop):
                total += values[order[k], i]
            mean = total / (stop - start)
            squares = 0.0
            for k in range(start, stop):
                gap = values[order[k], i] - mean
                squares += gap * gap
            constant[i] = squares / (stop - start) == 0


def _adjust(
    values,
    observed,
    gram,
    factor,
    factored,
    pattern,
    design,
    batch_index,
    ref,
    parametric,
    mean_only,
    adjusted,
    columns,
):
    """Run ComBat on features that are all estimable, given samples x features.

    Writes the result into `adjusted`, which serves as scratch when `columns`
    is None, and otherwise into its `columns`, one per feature of `values`.
    """
    sizes = np.bincount(batch_index)
    n_batch = sizes.size
    if design.shape[1] == n_batch:
        indicator = np.eye(n_batch)[batch_index]
        means = (np.where(observed, values, 0).T @ indicator) / (observed.T @ indicator)
        beta = _batch_fit(means, ref)
    else:
        beta = _fit(values, design, observed, gram, factor, factored, pattern)
    if ref is None:
        grand_mean = beta[:, :n_batch] @ (sizes / sizes.sum())
        pooled = np.ones(batch_index.size, dtype=bool)
    else:
        grand_mean = np.ascontiguousarray(beta[:, ref])
        pooled = batch_index == ref
    beta_t = np.ascontiguousarray(beta.T)
    n_features = values.shape[1]
    standardized = adjusted if columns is None else np.empty_like(values)
    scale = _aligned_zeros(n_features)
    _standardize(
        values,
        beta_t,
        design,
        n_batch,
        grand_mean,
        pooled,
        not observed.all(),
        standardized,
        _aligned_zeros(n_features, np.int64),
        _aligned_zeros(n_features),
        _aligned_zeros(n_features),
        scale,
        _aligned_zeros(_BLOCK),
    )

    n = _aligned_zeros((n_batch, n_features), np.int64)
    mean = _aligned_zeros((n_batch, n_features))
    ss = _aligned_zeros((n_batch, n_features))
    _moments(standardized, np.argsort(batch_index, kind="stable"), sizes, n, mean, ss)
    gamma_hat = _batch_fit(mean.T, ref).T
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.ones_like(gamma_hat)
    for b in range(n_batch):
        g = gamma_hat[b]
        delta_hat = np.ones(len(g)) if mean_only else ss[b] / (n[b] - 1)
        if not parametric:
            gamma_star[b], delta_star[b] = _nonparametric(
                n[b], mean[b], ss[b], g, delta_hat
            )
        elif mean_only:
            t2 = g.var(ddof=1)
            gamma_star[b] = (t2 * g + g.mean()) / (t2 + 1)
        else:
            gamma_star[b], delta_star[b] = _parametric(
                n[b], mean[b], ss[b], g, delta_hat
            )
    if ref is not None:
        gamma_star[ref] = 0
        delta_star[ref] = 1

    # sva subtracts gamma_star @ t(batch.design), whose zero terms spread a
    # non-finite gamma to every sample of its feature.
    nonfinite = ~np.isfinite(gamma_star)
    spread = (nonfinite.sum(axis=0) > nonfinite) | np.isnan(gamma_star)
    gamma_star[spread] = np.nan
    _remove(
        standardized,
        gamma_star,
        np.sqrt(delta_star),
        batch_index,
        scale,
        grand_mean,
        beta_t,
        design,
        _aligned_zeros(_BLOCK),
        adjusted,
        np.empty(0, np.intp) if columns is None else columns,
    )
    if ref is None:
        return
    if columns is None:
        adjusted[pooled] = values[pooled]
    else:
        adjusted[np.ix_(pooled, columns)] = values[pooled]


# Features per block: one block of every sample stays in cache between the
# passes over it, about 200 KB for 100 samples. 1024 ran as fast, 64 slower.
_BLOCK = 256


@numba.njit(
    error_model="numpy", no_cpython_wrapper=True, no_cfunc_wrapper=True, _nrt=False
)
def _covariate_terms(j, beta, design, n_batch, stop, lo, out):
    """Write sample j's fitted effect on features lo, lo + 1, ... into `out`.

    `beta` is p coefficients x features. Adds coefficients n_batch, ...,
    stop - 1 in that order, where index p + k stands for coefficient k: `stop`
    = p gives the covariate terms, p + n_batch the covariates, then the
    batches.
    """
    p = beta.shape[0]
    for i in range(out.size):
        out[i] = 0.0
    for kk in range(n_batch, stop):
        k = kk - p if kk >= p else kk
        coef = beta[k, lo : lo + out.size]
        x = design[j, k]
        for i in range(out.size):
            out[i] += coef[i] * x


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _standardize(
    values,
    beta,
    design,
    n_batch,
    grand_mean,
    pooled,
    missing,
    standardized,
    count,
    total,
    squares,
    scale,
    covariate,
):
    """Standardize each feature by its fit and its pooled residual variance.

    `values` and `standardized` are samples x features and `beta` is
    coefficients x features. Writes ``(values - stand_mean) / scale`` into
    `standardized`, where ``stand_mean`` is `grand_mean` plus the covariate
    terms, and each feature's scale into `scale`: the root of its residual
    variance over the `pooled` samples, the mean square or, with `missing`, the
    n - 1 variance of the observed ones. The first `n_batch` design columns are
    the batches. Each feature's sums run in sample order. `count`, `total` and
    `squares` are per-feature scratch that must start at zero, and `covariate`
    is scratch of length `_BLOCK`.
    """
    n_samples, n_features = values.shape
    p = beta.shape[0]
    # Zero-based row views: loops over range(lo, hi) ran 40% slower.
    for lo in range(0, n_features, _BLOCK):
        hi = n_features if n_features < lo + _BLOCK else lo + _BLOCK  # noqa: FURB136
        n_i, sum_i, sq_i = count[lo:hi], total[lo:hi], squares[lo:hi]
        # standardized holds the pooled samples' residuals until the last pass.
        for j in range(n_samples):
            if not pooled[j]:
                continue
            row = values[j, lo:hi]
            residual = standardized[j, lo:hi]
            _covariate_terms(j, beta, design, n_batch, p + n_batch, lo, residual)
            for i in range(row.size):
                r = row[i] - residual[i]
                residual[i] = r
                ok = not np.isnan(r)
                n_i[i] += ok
                # The sums start at +0.0 and so never reach -0.0, which adding
                # +0.0 would change.
                sum_i[i] += r if ok else 0.0
                sq_i[i] += r * r if ok else 0.0
        # sva switches to the n - 1 variance whenever any value is missing.
        if missing:
            for i in range(sum_i.size):
                sum_i[i] /= n_i[i]
                sq_i[i] = 0.0
            for j in range(n_samples):
                if not pooled[j]:
                    continue
                residual = standardized[j, lo:hi]
                for i in range(residual.size):
                    r = residual[i]
                    gap = r - sum_i[i]
                    sq_i[i] += gap * gap if not np.isnan(r) else 0.0
        scale_i, mean_i = scale[lo:hi], grand_mean[lo:hi]
        for i in range(scale_i.size):
            scale_i[i] = np.sqrt(sq_i[i] / (n_i[i] - 1 if missing else n_i[i]))
        terms = covariate[: hi - lo]
        for j in range(n_samples):
            _covariate_terms(j, beta, design, n_batch, p, lo, terms)
            row = values[j, lo:hi]
            out = standardized[j, lo:hi]
            for i in range(row.size):
                stand_mean = mean_i[i] + terms[i]
                out[i] = (row[i] - stand_mean) / scale_i[i]


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _remove(
    standardized,
    gamma_star,
    delta_root,
    batch_index,
    scale,
    grand_mean,
    beta,
    design,
    covariate,
    adjusted,
    columns,
):
    """Remove each sample's batch effects and return to the data's scale.

    `standardized` is samples x features; `gamma_star` and `delta_root` are
    batches x features. `covariate` is scratch of length `_BLOCK`. Writes in
    place, then copies feature i into column `columns[i]` of `adjusted`. An
    empty `columns` copies nothing.
    """
    n_samples, n_features = standardized.shape
    n_batch, p = gamma_star.shape[0], beta.shape[0]
    for lo in range(0, n_features, _BLOCK):
        hi = n_features if n_features < lo + _BLOCK else lo + _BLOCK  # noqa: FURB136
        scale_i, mean_i = scale[lo:hi], grand_mean[lo:hi]
        terms = covariate[: hi - lo]
        target = columns[lo:hi]
        for j in range(n_samples):
            _covariate_terms(j, beta, design, n_batch, p, lo, terms)
            b = batch_index[j]
            row = standardized[j, lo:hi]
            gamma, root = gamma_star[b, lo:hi], delta_root[b, lo:hi]
            for i in range(row.size):
                effect = row[i] - gamma[i]
                stand_mean = mean_i[i] + terms[i]
                row[i] = effect / root[i] * scale_i[i] + stand_mean
            for i in range(target.size):
                adjusted[j, target[i]] = row[i]


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _moments(standardized, order, sizes, n, mean, ss):
    """Write each batch's per-feature observed count, mean and sum of squared
    deviations into `n`, `mean` and `ss`, which must start at zero.

    `standardized` is samples x features. `order` lists the samples batch by
    batch, ascending within each batch, and `sizes` gives each batch's sample
    count. Outputs have one row per batch. NaN values are skipped.
    """
    n_features = standardized.shape[1]
    for lo in range(0, n_features, _BLOCK):
        hi = n_features if n_features < lo + _BLOCK else lo + _BLOCK  # noqa: FURB136
        start = 0
        for b in range(sizes.size):
            stop = start + sizes[b]
            # Summing in sample order from +0.0 matches numpy's nan-reductions
            # over a batch's columns, which it gathers into an F-ordered copy.
            count, total, squares = n[b, lo:hi], mean[b, lo:hi], ss[b, lo:hi]
            for k in range(start, stop):
                row = standardized[order[k], lo:hi]
                for i in range(row.size):
                    x = row[i]
                    ok = not np.isnan(x)
                    count[i] += ok
                    total[i] += x if ok else 0.0
            for i in range(total.size):
                total[i] /= count[i]
            for k in range(start, stop):
                row = standardized[order[k], lo:hi]
                for i in range(row.size):
                    d = row[i] - total[i]
                    squares[i] += d * d if not np.isnan(d) else 0.0
            start = stop


def _parametric(n, mean, ss, gamma_hat, delta_hat):
    """Return posterior batch effects under normal and inverse-gamma priors.

    `n`, `mean` and `ss` are each feature's `_moments` in the batch.
    """
    gamma_bar, t2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    m, v = delta_hat.mean(), delta_hat.var(ddof=1)
    a, b = (2 * v + m**2) / v, (m * v + m**3) / v
    count = n.astype(float)
    weight = t2 * count
    gamma, delta = _aligned_zeros(n.size), _aligned_zeros(n.size)
    gamma[:], delta[:] = gamma_hat, delta_hat
    gamma_new, delta_new = _aligned_zeros(n.size), _aligned_zeros(n.size)
    swapped = _iterate(
        mean,
        ss,
        count,
        weight,
        weight * gamma_hat,
        count / 2 + a - 1,
        gamma_bar,
        b,
        gamma,
        delta,
        gamma_new,
        delta_new,
    )
    return (gamma_new, delta_new) if swapped else (gamma, delta)


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _iterate(
    mean,
    ss,
    count,
    weight,
    pull,
    shape,
    gamma_bar,
    b,
    gamma_old,
    delta_old,
    gamma_new,
    delta_new,
):
    """Iterate the posterior means and variances to convergence, as sva does.

    `gamma_old` and `delta_old` start as the estimates; `gamma_new` and
    `delta_new` are scratch. The pairs swap roles each pass. Returns whether
    the result ends in the scratch pair.
    """
    swapped = False
    while True:
        gamma_big = delta_big = gamma_nan = delta_nan = False
        for i in range(count.size):
            gamma = (pull[i] + delta_old[i] * gamma_bar) / (weight[i] + delta_old[i])
            gap = mean[i] - gamma
            delta = ((ss[i] + count[i] * (gap * gap)) / 2 + b) / shape[i]
            # sva divides by signed values, so negative effects never block
            # convergence.
            gamma_change = abs(gamma - gamma_old[i]) / gamma_old[i]
            delta_change = abs(delta - delta_old[i]) / delta_old[i]
            gamma_nan |= np.isnan(gamma_change)
            delta_nan |= np.isnan(delta_change)
            gamma_big |= gamma_change > 1e-4
            delta_big |= delta_change > 1e-4
            gamma_new[i], delta_new[i] = gamma, delta
        gamma_old, gamma_new = gamma_new, gamma_old
        delta_old, delta_new = delta_new, delta_old
        swapped = not swapped
        # sva stops on a NaN gamma change and ignores a NaN delta change.
        if gamma_nan or not (gamma_big or (delta_big and not delta_nan)):
            break
    return swapped


def _nonparametric(n, mean, ss, gamma_hat, delta_hat):
    """Return posterior batch effects weighted by the other features' likelihoods.

    `n`, `mean` and `ss` are each feature's `_moments` in the batch.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_norm = np.log(2 * np.pi * delta_hat)
    gamma_star, delta_star = _aligned_zeros(n.size), _aligned_zeros(n.size)
    _weigh(
        n,
        mean,
        ss,
        gamma_hat,
        delta_hat,
        log_norm,
        _aligned_zeros(n.size),
        _aligned_zeros(n.size),
        _aligned_zeros(n.size),
        gamma_star,
        delta_star,
    )
    return gamma_star, delta_star


_FLOOR = 1e-280


@numba.njit(cache=True, error_model="numpy", no_cfunc_wrapper=True, _nrt=False)
def _weigh(
    n, mean, ss, gamma_hat, delta_hat, log_norm, c, u, w, gamma_star, delta_star
):
    """Average gamma_hat and delta_hat over the other features, by likelihood,
    into `gamma_star` and `delta_star`. `c`, `u` and `w` are per-feature scratch.
    """
    low_norm = low_inv = np.inf
    # Row i's log-likelihood under feature j's prior is
    # n[i] * c[j] + q[i] * u[j] + r[i] * w[j], which needs no division per pair.
    for j in range(n.size):
        if delta_hat[j] > 0:
            # The builtin min compiles an overload per argument type.
            low_norm = log_norm[j] if log_norm[j] < low_norm else low_norm  # noqa: FURB136
            inv = 1 / delta_hat[j]
            low_inv = inv if inv < low_inv else low_inv  # noqa: FURB136
        c[j] = -(log_norm[j] + gamma_hat[j] ** 2 / delta_hat[j]) / 2
        u[j] = -0.5 / delta_hat[j]
        w[j] = gamma_hat[j] / delta_hat[j]
    for i in range(n.size):
        q = ss[i] + n[i] * mean[i] ** 2
        r = n[i] * mean[i]
        # An upper bound on row i's log-likelihoods: every weight shares the
        # factor it adds, which cancels. A loose bound underflows the weights,
        # so such a row is weighed again relative to its maximum.
        bound = -(n[i] * low_norm + ss[i] * low_inv) / 2
        # Like sva, leave each feature out of its own prior: with c[i] = -inf
        # its log-likelihood is -inf or NaN, which every comparison drops.
        own = c[i]
        c[i] = -np.inf
        total, gamma, delta = _sums(bound, n[i], q, r, c, u, w, gamma_hat, delta_hat)
        if not total >= _FLOOR:
            top = -np.inf
            for j in range(n.size):
                log_lik = _log_lik(j, 0.0, n[i], q, r, c, u, w)
                if log_lik > top:  # noqa: PLR1730
                    top = log_lik
            total, gamma, delta = _sums(top, n[i], q, r, c, u, w, gamma_hat, delta_hat)
        c[i] = own
        gamma_star[i] = gamma / total
        delta_star[i] = delta / total


@numba.njit(
    error_model="numpy",
    fastmath={"contract"},
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
    _nrt=False,
)
def _log_lik(j, shift, n, q, r, c, u, w):
    """Return a row's log-likelihood under feature j's prior less `shift`,
    from `_weigh`'s per-row q, r and per-feature c, u, w."""
    # Subtracting the shift first lets contract fuse every step into one of
    # three FMAs, a shorter head for `_exp`'s dependent chain.
    return n * c[j] + (q * u[j] + (r * w[j] - shift))


# Reassociation lets the sums vectorize; it does not reach the callees.
@numba.njit(
    error_model="numpy",
    fastmath={"reassoc"},
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
    _nrt=False,
)
def _sums(shift, n, q, r, c, u, w, gamma_hat, delta_hat):
    """Return the sums of exp(log_lik - shift) over a row's prior, and of its
    products with gamma_hat and delta_hat."""
    total = gamma = delta = 0.0
    for j in range(c.size):
        x = _log_lik(j, shift, n, q, r, c, u, w)
        # Drop the NaN likelihoods that zero variances produce. Below -708 a
        # weight is under 1e-27 of any total `_weigh` accepts, so it is dropped
        # too.
        keep = x >= -708.0
        weight = _exp(x)
        total += weight if keep else 0.0
        gamma += weight * gamma_hat[j] if keep else 0.0
        delta += weight * delta_hat[j] if keep else 0.0
    return total, gamma, delta


# Remez minimax for exp's relative error on [-ln2/2, ln2/2] with the two
# lowest coefficients held at exactly 1: 1.6e-14 at degree 9. The weights only
# enter ratios of sums, so this need not be 1 ulp. These are the coefficients
# of r**2 to r**9.
_MINIMAX = (
    0.4999999999966452,
    0.166666666672648,
    0.041666666962103076,
    0.008333333095214026,
    0.0013888811169238944,
    0.00019841538993727509,
    2.4880797289976457e-05,
    2.7493078688247787e-06,
)
_SHIFTER = 1.5 * 2**52
_SHIFTER_BITS = 0x4338000000000000


@numba.njit(
    fastmath={"contract"}, no_cpython_wrapper=True, no_cfunc_wrapper=True, _nrt=False
)
def _exp(x):
    """Return exp(x), within 4.1e-14 relative for x in [-708, 0] with FMA and
    1e-13 without.

    Outside that range the result is undefined, NaN included: `_sums` discards
    those lanes.
    """
    # numba's np.exp is a scalar libm call; this polynomial vectorizes. Adding
    # _SHIFTER rounds x / ln 2 to an integer k held in the low bits, and the
    # result is scaled by 2**k built from bits. With one ln 2 constant, r errs
    # by |k| * 2.3e-17 under FMA.
    shifted = x * 1.4426950408889634 + _SHIFTER
    r = x - (shifted - _SHIFTER) * 0.6931471805599453
    # Estrin's scheme shortens the dependent FMA chain that bounds `_sums`;
    # the two outer steps stay Horner, as at degree 11, where Estrin's final
    # additions cost an ulp.
    c = _MINIMAX
    r2 = r * r
    low = (c[0] + c[1] * r) + (c[2] + c[3] * r) * r2
    high = (c[4] + c[5] * r) + (c[6] + c[7] * r) * r2
    p = 1.0 + r * (1.0 + r * (low + high * (r2 * r2)))
    return p * _as_float((_as_int(shifted) - _SHIFTER_BITS + 1023) << 52)


def _bitcast(source, target):
    """Return a numba intrinsic that reinterprets source's bits as target."""

    def codegen(context, builder, signature, args):
        return builder.bitcast(args[0], context.get_value_type(target))

    return intrinsic(lambda typingctx, x: (target(source), codegen))


_as_int = _bitcast(types.float64, types.int64)
_as_float = _bitcast(types.int64, types.float64)
