library(readr)
library(dplyr)
library(lmerTest)
library(jtools)
library(tidyr)
library(ggplot2)
library(interactions)
library(gridExtra)
library(car)
library(mgcv)
library(nlme)

# ----------------- DATASETS ----------------- #

# LSTM results for English and Hindi
df_lstm <- read_csv("lstm_hi_en.csv")
# Transformer results for English and Hindi
df_transformer <- read_csv("transformer_hi_en.csv")

# ----------------- SPLIT DATA ----------------- #

# Split data into English and Hindi subsets for both the LSTM and the Transformer datasets
df_lstm_en <- df_lstm %>% filter(lang_y == "EN")
df_lstm_hi <- df_lstm %>% filter(lang_y == "HI")
df_transformer_en <- df_transformer %>% filter(lang_y == "EN")
df_transformer_hi <- df_transformer %>% filter(lang_y == "HI")

# ----------------- KEEP NEEDED COLUMNS ----------------- #

# Function to subset dataframe columns
subset_columns <- function(df, cols) {
  df %>%
    select(all_of(cols))
}

# Column mappings for different data types
col_map <- list(
  english_lstm = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", 
                   "actual_word", "word_len", "firstfix.dur", "dur", "firstrun.dur", "lstm_entropy", 
                   "lstm_surprisal", "lstm_perplexity"),
  hindi_lstm = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", 
                 "actual_word", "word_len", "FFD", "TFT", "FPRT", "lstm_entropy", 
                 "lstm_surprisal","lstm_perplexity"),
  english_transformer = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", 
                          "actual_word", "word_len", "firstfix.dur", "dur", "firstrun.dur", "tf_entropy", 
                          "tf_surprisal", "tf_perplexity"),
  hindi_transformer = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", 
                        "actual_word", "word_len", "FFD", "TFT", "FPRT", "tf_entropy", 
                        "tf_surprisal", "tf_perplexity")
)

df_lstm_en <- subset_columns(df_lstm_en, col_map$english_lstm)
df_lstm_hi <- subset_columns(df_lstm_hi, col_map$hindi_lstm)
df_transformer_en <- subset_columns(df_transformer_en, col_map$english_transformer)
df_transformer_hi <- subset_columns(df_transformer_hi, col_map$hindi_transformer)

# ----------------- RENAMING COLUMNS ----------------- #

# Renaming columns for consistency in both datasets
rename_vars <- function(df) {
  df %>%
    rename(FFD = firstfix.dur, TFT = dur, FPRT = firstrun.dur) %>%
    mutate(lang_y = as.factor(lang_y)) 
}

df_lstm_en <- rename_vars(df_lstm_en)
df_transformer_en <- rename_vars(df_transformer_en)

# ----------------- CONVERT FACTOR DATA TYPES ----------------- #

convert_factors <- function(df) {
  df %>%
    mutate(participant = as.factor(participant),
           word = as.factor(word))
}

df_lstm_en <- convert_factors(df_lstm_en)
df_lstm_hi <- convert_factors(df_lstm_hi)
df_transformer_en <- convert_factors(df_transformer_en)
df_transformer_hi <- convert_factors(df_transformer_hi)

# ----------------- REMOVE SKIPPED WORDS ----------------- #

dependent_vars <- c("FFD", "TFT", "FPRT")

replace_zeros <- function(df) {
  df[dependent_vars] <- lapply(df[dependent_vars], function(x) na_if(x, 0))
  df
}

df_lstm_en <- replace_zeros(df_lstm_en)
df_lstm_hi <- replace_zeros(df_lstm_hi)
df_transformer_en <- replace_zeros(df_transformer_en)
df_transformer_hi <- replace_zeros(df_transformer_hi)

# ----------------- LOG TRANSFORMER DEPENDET VARIABLES ----------------- #

log_transform <- function(df) {
  df[dependent_vars] <- lapply(df[dependent_vars], function(x) log(x))
  df
}

df_lstm_en <- log_transform(df_lstm_en)
df_lstm_hi <- log_transform(df_lstm_hi)
df_transformer_en <- log_transform(df_transformer_en)
df_transformer_hi <- log_transform(df_transformer_hi)

# ----------------- CORRELATIONS ----------------- #

correlation_lstm_en <- cor(df_lstm_en$lstm_entropy, df_lstm_en$lstm_surprisal, use = "complete.obs")
correlation_lstm_hi <- cor(df_lstm_hi$lstm_entropy, df_lstm_hi$lstm_surprisal, use = "complete.obs")
correlation_tf_en <- cor(df_transformer_en$tf_entropy, df_transformer_en$tf_surprisal, use = "complete.obs")
correlation_tf_hi <- cor(df_transformer_hi$tf_entropy, df_transformer_hi$tf_surprisal, use = "complete.obs")

print(c("Correlation LSTM English: ", correlation_lstm_en))
print(c("Correlation LSTM Hindi: ", correlation_lstm_hi))
print(c("Correlation Transformer English: ", correlation_tf_en))
print(c("Correlation Transformer Hindi: ", correlation_tf_hi))

# ----------------- LMER ----------------- #

# Function to preprocess, fit baseline model and fit extended models for LSTM and Transformer
fit_models_seperate <- function(df_lstm, df_transformer, dependent_vars) {
  results <- list()
  
  for (y in dependent_vars) {
    cat("Processing dependent variable:", y, "\n\n")
    
    # Preprocess and prepare merged dataset
    df_lstm_test <- df_lstm %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "lstm_surprisal", "lstm_entropy")))
    df_transformer_test <- df_transformer %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "tf_surprisal", "tf_entropy")))
    
    # Combine datasets
    merged_df <- df_lstm_test %>%
      inner_join(df_transformer_test, by = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", "actual_word", "word_len", "FFD", "TFT", "FPRT"))
    
    # Fit baseline model
    baseline_formula <- paste(y, "~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word)")
    baseline_model <- lmer(baseline_formula, data = merged_df)
    
    # Fit LSTM surprisal and entropy
    lstm_model <- update(baseline_model, . ~ . + lstm_surprisal + lstm_entropy, data = merged_df)
    
    # Fit Transformer surprisal and entropy
    transformer_model <- update(baseline_model, . ~ . + tf_surprisal + tf_entropy, data = merged_df)
    
    # Store results
    results[[paste("baseline", y, sep = "_")]] <- baseline_model
    results[[paste("lstm", y, sep = "_")]] <- lstm_model
    results[[paste("transformer", y, sep = "_")]] <- transformer_model
    
    # Print ANOVA comparisons
    cat("ANOVA Comparisons for LSTM", y, ":\n")
    print(anova(baseline_model, lstm_model))
    cat("ANOVA Comparisons for Transformer", y, ":\n")
    print(anova(baseline_model, transformer_model))
  }
  
  return(results)
}

dependent_vars <- c("FFD", "TFT", "FPRT")

models_en <- fit_models_seperate(df_lstm_en, df_transformer_en, dependent_vars)
models_hi <- fit_models_seperate(df_lstm_hi, df_transformer_hi, dependent_vars)


# ----------------- CHI-SQUARE ANALYSIS ----------------- #


# Chi-sqaure and p-values filtered from ANOVA for analysis 
fit_models_chi <- function(df_lstm, df_transformer, dependent_vars) {
  results <- list()
  summary_results <- data.frame()
  
  for (y in dependent_vars) {
    cat("Processing dependent variable:", y, "\n\n")
    
    # Preprocess and prepare merged dataset
    df_lstm_test <- df_lstm %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "lstm_surprisal", "lstm_entropy")))
    df_transformer_test <- df_transformer %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "tf_surprisal", "tf_entropy")))
    
    # Combine datasets
    merged_df <- df_lstm_test %>%
      inner_join(df_transformer_test, by = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", "actual_word", "word_len", "FFD", "TFT", "FPRT"))
    
    # Fit baseline model
    baseline_formula <- paste(y, "~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word)")
    baseline_model <- lmer(baseline_formula, data = merged_df)
    
    # Fit LSTM surprisal and entropy
    lstm_model <- update(baseline_model, . ~ . + lstm_surprisal + lstm_entropy)
    
    # Fit Transformer surprisal and entropy
    transformer_model <- update(baseline_model, . ~ . + tf_surprisal + tf_entropy)
    
    # Store model results 
    results[[paste("baseline", y, sep = "_")]] <- baseline_model
    results[[paste("lstm", y, sep = "_")]] <- lstm_model
    results[[paste("transformer", y, sep = "_")]] <- transformer_model
    
    # ANOVA comparisons and getting chi-square and p-values
    anova_results_lstm <- anova(baseline_model, lstm_model)
    anova_results_tf <- anova(baseline_model, transformer_model)
    summary_row <- data.frame(
      Measure = y,
      Model = c("LSTM", "Transformer"),
      Chi_square = c(anova_results_lstm$"Chisq"[2], anova_results_tf$"Chisq"[2]),
      P_value = c(anova_results_lstm$"Pr(>Chisq)"[2], anova_results_tf$"Pr(>Chisq)"[2])
    )
    summary_results <- rbind(summary_results, summary_row)
    
    # Printing ANOVA comparisons
    cat("ANOVA Comparisons for LSTM", y, ":\n")
    print(anova_results_lstm)
    cat("ANOVA Comparisons for Transformer", y, ":\n")
    print(anova_results_tf)
  }
  
  return(list(models = results, summary = summary_results))
}

results_en_chi <- fit_models_chi(df_lstm_en, df_transformer_en, dependent_vars)
results_hi_chi <- fit_models_chi(df_lstm_hi, df_transformer_hi, dependent_vars)

# View Chi-square results
print(results_en_chi$summary)
print(results_hi_chi$summary)

# Save results to files
write.table(results_en_chi$summary, "Chi_results_english.txt", row.names = FALSE, sep = "\t")
write.table(results_hi_chi$summary, "Chi_results_hindi.txt", row.names = FALSE, sep = "\t")


# ----------------- SAVE RESULTS ----------------- #

# Store ANOVA results for LSTM and Transformer in a .txt file
fit_models_save <- function(df_lstm, df_transformer, dependent_vars, file_name) {
  results <- list()
  
  sink(file_name)
  
  for (y in dependent_vars) {
    cat("Processing dependent variable:", y, "\n\n")
    
    # Preprocess and prepare datasets
    df_lstm_test <- df_lstm %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "lstm_surprisal", "lstm_entropy")))
    df_transformer_test <- df_transformer %>%
      drop_na(all_of(c(y, "word_len", "sent_id_and_idx", "word_idx", "participant", "word", "tf_surprisal", "tf_entropy")))
    
    # Combine datasets
    merged_df <- df_lstm_test %>%
      inner_join(df_transformer_test, by = c("lang_y", "participant", "sent_id_and_idx", "word_idx", "word", "actual_word", "word_len", "FFD", "TFT", "FPRT"))
    
    # Define model formulas
    baseline_formula <- paste(y, "~ word_len + sent_id_and_idx + word_idx + (1 | participant) + (1 | word)")
    lstm_formula <- paste(baseline_formula, "+ lstm_surprisal + lstm_entropy")
    transformer_formula <- paste(baseline_formula, "+ tf_surprisal + tf_entropy")
    
    # Fit models
    baseline_model <- lmer(as.formula(baseline_formula), data = merged_df)
    lstm_model <- lmer(as.formula(lstm_formula), data = merged_df)
    transformer_model <- lmer(as.formula(transformer_formula), data = merged_df)
    
    # Output model formulas and ANOVA results
    cat("Model Formulas for", y, ":\n")
    cat("Baseline Model: ", baseline_formula, "\n")
    cat("LSTM Model: ", lstm_formula, "\n")
    cat("Transformer Model: ", transformer_formula, "\n\n")
    cat("ANOVA Comparisons for LSTM:\n")
    print(anova(baseline_model, lstm_model))
    cat("\nANOVA Comparisons for Transformer:\n")
    print(anova(baseline_model, transformer_model))
    cat("\n\n--------------------------------------------\n\n")
  }
  
  sink()
  
  return(file_name)
}

dependent_vars <- c("FFD", "TFT", "FPRT")

file_en <- "anova_results_english.txt"
file_hi <- "anova_results_hindi.txt"

models_en <- fit_models_save(df_lstm_en, df_transformer_en, dependent_vars, file_en)
models_hi <- fit_models_save(df_lstm_hi, df_transformer_hi, dependent_vars, file_hi)



# ----------------- EXTRA NOT USED FUNCTIONS ----------------- #
#
# # ----------------- VIF CHECK ----------------- #
# 
# # Variance Inflation Factor - under 5 acceptable, above 10 bad
# 
# # Function to calculate VIF for all models
# calculate_vif_for_all_models <- function(models_grouped, models_incremental, language) {
#   vif_results <- list()  # Create an empty list to store VIF results
#   all_models <- list(grouped = models_grouped, incremental = models_incremental)  # Combine both strategies into one list
#   
#   for (strategy in names(all_models)) {
#     strategy_models = all_models[[strategy]]
#     dep_vars <- names(strategy_models)  # Get the names of dependent variables
#     
#     for (dep_var in dep_vars) {
#       model_list <- strategy_models[[dep_var]]  # Get the list of models for each dependent variable
#       
#       for (i in seq_along(model_list)) {
#         model <- model_list[[i]]
#         vif_result <- vif(model)
#         vif_results[[paste(language, strategy, dep_var, i, sep = "_")]] <- vif_result
#         
#         # cat(paste("VIF for", language, strategy, dep_var, "Model", i, ":\n"))
#         # print(vif_result)
#         # cat("\n")
#       }
#     }
#   }
#   
#   return(vif_results)
# }
# 
#
# all_vif_results_en <- calculate_vif_for_all_models(models_grouped_en, models_incremental_en, "English")
# all_vif_results_hi <- calculate_vif_for_all_models(models_grouped_hi, models_incremental_hi, "Hindi")
# 
# # ----------------- RESIDUAL DIAGNOSTICS ----------------- #
# 
# # Function to perform residual diagnostics and save plots to files
# perform_residual_diagnostics <- function(model, model_name) {
#   # Create a directory to save the plots if it doesn't exist
#   if (!dir.exists("diagnostic_plots")) {
#     dir.create("diagnostic_plots")
#   }
#   
#   # Residuals vs. Fitted Values Plot
#   png(filename = paste0("diagnostic_plots/", model_name, "_residuals_vs_fitted.png"))
#   plot(fitted(model), resid(model), pch='.')
#   abline(0, 0)
#   title(main = paste("Residuals vs Fitted for", model_name))
#   dev.off()
#   
#   # Q-Q Plot for Normality of Residuals
#   png(filename = paste0("diagnostic_plots/", model_name, "_qqplot.png"))
#   qqnorm(resid(model))
#   qqline(resid(model))
#   title(main = paste("Q-Q Plot for", model_name))
#   dev.off()
#   
#   # Density Plot of Residuals
#   png(filename = paste0("diagnostic_plots/", model_name, "_density.png"))
#   plot(density(resid(model)))
#   title(main = paste("Density of Residuals for", model_name))
#   dev.off()
# }
# 

