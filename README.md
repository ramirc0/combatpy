# combatpy

ComBat batch-effect correction for NumPy, with results that match R's `sva::ComBat`.

## Description

combatpy removes known batch effects from a features × samples matrix of normalized log-scale data, such as protein or gene expression. It implements the empirical Bayes method of Johnson, Li and Rabinovic (2007) the way R's `sva::ComBat` does. It supports parametric and non-parametric priors, mean-only adjustment, a reference batch and covariates whose effects are kept. On the simulated data of sva's own unit tests, its output stays within a relative difference of 4e-13 of sva's.

sva fails or returns NA for a feature that is missing from a whole batch, which is common in proteomics. combatpy keeps missing values in place. It returns features it cannot estimate unchanged and warns about them.

The test suite reruns the protein-level ComBat analysis of Chen et al. (2025) on their Quartet and simulated proteomics data. It reproduces the paper's ComBat outputs and published SNR and CV values. It also pins baseline metrics for ComBat on those data, so later changes have a reference.

## Getting Started

### Dependencies

* Python 3.12 or later
* NumPy 2 or later
* pandas or polars, optional, to pass in and get back DataFrames

### Installing

* Install from a local checkout:

```sh
uv add /path/to/combatpy
```

### Executing program

* Pass a features × samples matrix, one batch label per sample and, optionally, the covariates to keep:

```python
import combatpy

corrected = combatpy.combat(log_intensities, batch, covariates=group)
```

* NaN marks a missing value.
* Numeric covariates enter the model as they are. Other covariates are one-hot encoded.
* `parametric=False` uses non-parametric priors.
* `mean_only=True` adjusts batch means only.
* `ref_batch=label` keeps one batch unchanged and adjusts the others toward it.
* A pandas or polars DataFrame comes back as the same type with the same labels.
* Run the tests. The first run downloads about 14 MB of the paper's data from Figshare into `~/.cache/combatpy`:

```sh
uv run pytest
uv run pytest -m "not paper"   # skip the tests that need the paper's data
```

* Regenerate the sva reference outputs in `tests/data/sva` from an [sva-devel](https://github.com/jtleek/sva-devel) checkout. This needs base R only:

```sh
Rscript scripts/sva_reference.R /path/to/sva-devel tests/data/sva
```

## Help

* "batch has N labels but data has M columns" means the matrix is likely transposed. Rows must be features and columns samples.
* "N of M features cannot be estimated" lists features that are constant within a batch, have too few observed values in a batch, or cannot separate batch from covariate effects on their observed samples. They come back unchanged.
* ComBat removes batch-level effects. Normalize per-sample loading differences first.
* Integer-coded group labels count as numeric covariates. Pass strings to one-hot encode them.

```python
help(combatpy.combat)
```

## Authors

Christian Ramirez (christian.ramirez1@umassmed.edu)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* W. E. Johnson, C. Li and A. Rabinovic, "Adjusting batch effects in microarray expression data using empirical Bayes methods", Biostatistics 8, 118–127 (2007).
* The [sva](https://bioconductor.org/packages/sva) package, whose `ComBat.R` this port follows.
* Q. Chen et al., "Protein-level batch-effect correction enhances robustness in MS-based proteomics", Nat Commun 16, 9735 (2025). Their [code](https://github.com/QiaochuChen/proteomics-batch-effect-correction-benchmarking) and Figshare data (CC BY 4.0) back the real-world tests.
* The pycombat module of [inmoose](https://github.com/epigenelabs/inmoose), an earlier Python port.
