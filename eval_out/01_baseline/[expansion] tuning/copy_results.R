#!/usr/bin/env Rscript
# Copy log and profile files from tuning run subdirs to a destination directory,
# renaming each file after the run name (e.g., v3_100s_6str.log, v3_100s_6str.csv).

# --- Configuration ---
tuning_dir <- "~/projects/buzzdetect-agentic/eval_out/01_baseline/tuning"
dest_dir   <- "/media/luke/USB DISK/buzzdetect-tuning"
# ---------------------

if (!dir.exists(dest_dir)) {
  dir.create(dest_dir, recursive = TRUE)
  message("Created destination directory: ", dest_dir)
}

run_dirs <- list.dirs(tuning_dir, full.names = TRUE, recursive = FALSE)
copied <- 0

for (run_path in run_dirs) {
  run_name <- basename(run_path)

  # --- Log file ---
  # Check directly in run dir first, then inside files/
  log_candidates <- c(
    list.files(run_path,                    pattern = "\\.log$", full.names = TRUE),
    list.files(file.path(run_path, "files"), pattern = "\\.log$", full.names = TRUE)
  )
  log_candidates <- log_candidates[file.exists(log_candidates)]

  if (length(log_candidates) > 0) {
    if (length(log_candidates) > 1) {
      log_candidates <- log_candidates[which.max(file.mtime(log_candidates))]
    }
    dest_log <- file.path(dest_dir, paste0(run_name, ".log"))
    file.copy(log_candidates, dest_log, overwrite = TRUE)
    message("Copied log:     ", log_candidates, " -> ", dest_log)
    copied <- copied + 1
  } else {
    message("No log found in: ", run_path)
  }

  # --- Profile CSV ---
  profile_candidates <- c(
    list.files(run_path,                    pattern = "profile\\.csv$", full.names = TRUE),
    list.files(file.path(run_path, "files"), pattern = "profile\\.csv$", full.names = TRUE)
  )
  profile_candidates <- profile_candidates[file.exists(profile_candidates)]

  if (length(profile_candidates) > 0) {
    if (length(profile_candidates) > 1) {
      profile_candidates <- profile_candidates[which.max(file.mtime(profile_candidates))]
    }
    dest_csv <- file.path(dest_dir, paste0(run_name, ".csv"))
    file.copy(profile_candidates, dest_csv, overwrite = TRUE)
    message("Copied profile: ", profile_candidates, " -> ", dest_csv)
    copied <- copied + 1
  } else {
    message("No profile.csv in: ", run_path)
  }
}

message("\nDone. ", copied, " file(s) copied to ", dest_dir)
