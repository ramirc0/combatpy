# Writes the sva::ComBat reference outputs that tests/test_sva.py compares against.
# Usage: Rscript scripts/sva_reference.R <sva-devel checkout> <output directory>
# Needs base R only: the BiocParallel and genefilter functions that ComBat calls are defined here.

args <- commandArgs(trailingOnly = TRUE)

bpparam <- function(...) NULL
bplapply <- function(X, FUN, ..., BPPARAM) lapply(X, FUN, ...)
rowVars <- function(x, ...) {
  n <- rowSums(!is.na(x))
  n[n <= 1] <- NA
  rowSums((x - rowMeans(x, ...))^2, ...) / (n - 1)
}
for (file in c("helper.R", "ComBat.R")) {
  for (expr in parse(file.path(args[1], "R", file))) {
    if (!(is.call(expr) && identical(expr[[1]], as.name("library")))) eval(expr, globalenv())
  }
}

# The simulated data of sva's tests/testthat/test_combat_sim.R.
nb <- 3
nw <- 5
n <- nb * nw
genes <- 100
batch <- rep(1:nb, each = nw)
set.seed(0)
treat <- rbinom(n, 1, c(.5, .5))
X <- model.matrix(~ -1 + as.factor(batch) + treat)
set.seed(0)
y <- matrix(rnorm(n * genes, X %*% (1:(nb + 1))), nrow = genes, byrow = TRUE)
set.seed(1)
y_missing <- y
y_missing[matrix(rbinom(n * genes, 1, .9), nrow = genes, byrow = TRUE) == 0] <- NA

text <- function(x) {
  values <- ifelse(is.na(x), "", sprintf("%.17g", x))
  colnames(values) <- paste0("s", seq_len(ncol(x)))
  values
}
write_values <- function(x, name) {
  write.table(text(x), file.path(args[2], name), sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
write_values(y, "y.csv")
write_values(y_missing, "y_missing.csv")
write.csv(data.frame(batch, treat), file.path(args[2], "samples.csv"), row.names = FALSE)

cases <- expand.grid(
  parametric = c(TRUE, FALSE), covariates = c(FALSE, TRUE), mean_only = c(FALSE, TRUE),
  ref_batch = c(NA, 2), missing = c(FALSE, TRUE)
)
adjusted <- lapply(seq_len(nrow(cases)), function(i) {
  case <- cases[i, ]
  result <- suppressMessages(ComBat(
    if (case$missing) y_missing else y, batch,
    mod = if (case$covariates) treat,
    par.prior = case$parametric, mean.only = case$mean_only,
    ref.batch = if (!is.na(case$ref_batch)) case$ref_batch
  ))
  cbind(case[rep(1, genes), ], text(result))
})
reference <- do.call(rbind, adjusted)
reference[] <- lapply(reference, function(x) if (is.logical(x)) tolower(x) else x)
write.csv(reference, gzfile(file.path(args[2], "combat.csv.gz")), row.names = FALSE, quote = FALSE, na = "")
