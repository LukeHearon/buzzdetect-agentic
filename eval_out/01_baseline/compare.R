#!/usr/bin/env Rscript
# Compare grid search runs by analysis rate.
# Decodes settings from dirname; uses buzzr::evaluate_log() for rate metrics.
# Run from repo root or this file's directory.

library(buzzr)
library(stringr)
library(dplyr)

tuning_dir <- "eval_out/01_baseline/tuning"

rds_path <- "eval_out/01_baseline/compare.rds"

existing <- if (file.exists(rds_path)) readRDS(rds_path) else tibble()

run_dirs <- list.dirs(tuning_dir, full.names = TRUE, recursive = FALSE)

new_results <- lapply(run_dirs, function(run_path) {
  run_name <- basename(run_path)

  if (nrow(existing) > 0 && run_name %in% existing$run) return(NULL)

  log_file <- list.files(file.path(run_path, "files"), pattern = "\\.log$",
                         full.names = TRUE)
  if (length(log_file) == 0) return(NULL)
  if (length(log_file) > 1) log_file <- log_file[which.max(file.mtime(log_file))]

  metrics <- tryCatch(buzzr::evaluate_log(log_file), error = function(e) {
    message("evaluate_log() failed for ", run_name, ": ", e$message)
    return(NULL)
  })
  if (is.null(metrics)) return(NULL)

  tibble(
    run        = run_name,
    model      = str_extract(run_name, "^[^_]+(?:_[^_]+)*?(?=_\\d+s)"),
    chunklength = as.integer(str_extract(run_name, "(?<=_)\\d+(?=s)")),
    n_streamers = as.integer(str_extract(run_name, "(?<=_)\\d+(?=str)")),
    n_gpu       = as.integer(str_extract(run_name, "(?<=_)\\d+(?=gpu)")),
    buffer_depth = as.integer(str_extract(run_name, "(?<=_depth)\\d+")),
    as_tibble(as.list(metrics))
  )
}) |>
  bind_rows()

results <- bind_rows(existing, new_results)

saveRDS(results, rds_path)

