# ================================
# 1. Environment and data loading
# ================================
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)      # AUC-ROC
library(e1071)     # SVM
library(xgboost)
library(Matrix)
library(fastshap)

set.seed(42)

raw_data <- read.csv("compas-scores-two-years.csv", stringsAsFactors = FALSE)
cat("Number of rows in raw data:", nrow(raw_data), "\n")

# ================================
# 2. Data cleaning
# ================================
df_clean <- raw_data %>%
  filter(days_b_screening_arrest <= 30) %>%
  filter(days_b_screening_arrest >= -30) %>%
  filter(is_recid != -1) %>%
  filter(c_charge_degree %in% c("F", "M")) %>%
  filter(score_text != "N/A")

cat("Number of rows after cleaning:", nrow(df_clean), "\n")

# ================================
# 3. Feature engineering
# ================================
df_clean$two_year_recid  <- as.factor(df_clean$two_year_recid)
df_clean$race            <- as.factor(df_clean$race)
df_clean$sex             <- as.factor(df_clean$sex)
df_clean$c_charge_degree <- as.factor(df_clean$c_charge_degree)
df_clean$age_cat         <- as.factor(df_clean$age_cat)

# Binary COMPAS decision (Low = 0, Medium/High = 1)
df_clean$compas_decision <- ifelse(df_clean$score_text == "Low", 0, 1)
df_clean$compas_decision <- as.factor(df_clean$compas_decision)

# ================================
# 4. Train/test split (shared by all models)
# ================================
trainIndex <- createDataPartition(df_clean$two_year_recid, p = 0.7,
                                  list = FALSE, times = 1)
data_train <- df_clean[ trainIndex, ]
data_test  <- df_clean[-trainIndex, ]

y_test_factor  <- data_test$two_year_recid
y_test_numeric <- as.numeric(as.character(y_test_factor))  # for AUC

# ================================
# 5. Helper functions: metrics + d' / b
# ================================
# Signal detection metrics from 2x2 confusion table
sdt_from_table <- function(tbl) {
  # Rows: predicted 0/1, columns: actual 0/1
  TN <- tbl[1, 1]; FN <- tbl[1, 2]
  FP <- tbl[2, 1]; TP <- tbl[2, 2]
  
  hit_rate <- TP / (TP + FN)
  fa_rate  <- FP / (FP + TN)
  
  # Avoid 0 or 1 probabilities for qnorm
  eps <- 1e-5
  hit_rate <- min(max(hit_rate, eps), 1 - eps)
  fa_rate  <- min(max(fa_rate,  eps), 1 - eps)
  
  zH <- qnorm(hit_rate)
  zF <- qnorm(fa_rate)
  
  d_prime <- zH - zF
  beta    <- exp((zF^2 - zH^2) / 2)
  
  c(d_prime = d_prime, beta = beta)
}

# Unified evaluation: Accuracy / Precision / Recall / F1 / AUC / d' / b
evaluate_model <- function(model_name, pred_factor, pred_score, y_true_factor) {
  cm <- confusionMatrix(
    pred_factor,
    y_true_factor,
    positive = "1",
    mode = "everything"
  )
  acc  <- as.numeric(cm$overall["Accuracy"])
  prec <- as.numeric(cm$byClass["Precision"])
  rec  <- as.numeric(cm$byClass["Recall"])
  f1   <- as.numeric(cm$byClass["F1"])
  
  roc_obj <- roc(response = y_true_factor,
                 predictor = as.numeric(pred_score),
                 levels = rev(levels(y_true_factor)))
  auc_val <- as.numeric(auc(roc_obj))
  
  sdt <- sdt_from_table(cm$table)
  
  list(
    name   = model_name,
    acc    = acc,
    prec   = prec,
    rec    = rec,
    f1     = f1,
    auc    = auc_val,
    dprime = sdt["d_prime"],
    beta   = sdt["beta"]
  )
}

print_eval <- function(res) {
  cat(res$name, "\n")
  cat("  Accuracy:", round(res$acc, 4),
      " Precision:", round(res$prec, 4),
      " Recall:",    round(res$rec, 4),
      " F1:",        round(res$f1, 4),
      " AUC:",       round(res$auc, 4), "\n")
  cat("  d':", round(res$dprime, 3),
      "  b:", round(res$beta, 3), "\n\n")
}

# Fairness metrics per race
evaluate_fairness <- function(model_name, pred_factor, y_true_factor, race_factor) {
  # Align vectors
  df_eval <- data.frame(
    obs = y_true_factor,
    pred = pred_factor,
    race = race_factor
  )
  
  # Filter for African-American and Caucasian
  df_aa <- df_eval[df_eval$race == "African-American", ]
  df_cc <- df_eval[df_eval$race == "Caucasian", ]
  
  # Function to get FPR/FNR from confusion matrix
  get_rates <- function(preds, obs) {
    tbl <- table(Predicted = preds, Actual = obs)
    # Ensure 2x2 table structure even if some classes missing
    # Actual 0 (TN+FP), Actual 1 (FN+TP)
    # Predicted 0 (TN+FN), Predicted 1 (FP+TP)
    
    # Check if we have enough data points to form table
    if(nrow(tbl) < 2 || ncol(tbl) < 2) return(c(FPR=NA, FNR=NA))
    
    TN <- tbl[1,1]; FN <- tbl[1,2]
    FP <- tbl[2,1]; TP <- tbl[2,2]
    
    FPR <- FP / (FP + TN)
    FNR <- FN / (TP + FN)
    return(c(FPR=FPR, FNR=FNR))
  }
  
  rates_aa <- get_rates(df_aa$pred, df_aa$obs)
  rates_cc <- get_rates(df_cc$pred, df_cc$obs)
  
  cat("Fairness Metrics for:", model_name, "\n")
  cat("  [African-American] FPR:", round(rates_aa["FPR"], 3), 
      " FNR:", round(rates_aa["FNR"], 3), "\n")
  cat("  [Caucasian]        FPR:", round(rates_cc["FPR"], 3), 
      " FNR:", round(rates_cc["FNR"], 3), "\n\n")
}

# ================================
# 6. Model 1: Logistic Regression (7 features)
# ================================
formula_lr7 <- two_year_recid ~ sex + age + race + priors_count +
  c_charge_degree + juv_fel_count + juv_misd_count

model_lr7 <- glm(formula_lr7,
                 data = data_train,
                 family = "binomial")

lr7_prob <- predict(model_lr7, data_test, type = "response")
lr7_pred <- ifelse(lr7_prob > 0.5, 1, 0)
lr7_pred <- as.factor(lr7_pred)

res_lr7 <- evaluate_model("Logistic (7 features)",
                          pred_factor  = lr7_pred,
                          pred_score   = lr7_prob,
                          y_true_factor = y_test_factor)

# ================================
# 7. Model 2: Logistic Regression (age + priors)
# ================================
formula_lr2 <- two_year_recid ~ age + priors_count

model_lr2 <- glm(formula_lr2,
                 data = data_train,
                 family = "binomial")

lr2_prob <- predict(model_lr2, data_test, type = "response")
lr2_pred <- ifelse(lr2_prob > 0.5, 1, 0)
lr2_pred <- as.factor(lr2_pred)

res_lr2 <- evaluate_model("Logistic (age + priors)",
                          pred_factor  = lr2_pred,
                          pred_score   = lr2_prob,
                          y_true_factor = y_test_factor)

# ================================
# 8. Model 3: Nonlinear SVM (RBF)
# ================================
features_svm <- c("sex", "age", "race", "priors_count",
                  "c_charge_degree", "juv_fel_count", "juv_misd_count")

formula_svm <- as.formula(
  paste("~", paste(features_svm, collapse = " + "), "- 1")
)

X_train_svm <- model.matrix(formula_svm, data = data_train)
X_test_svm  <- model.matrix(formula_svm, data = data_test)

y_train_svm <- data_train$two_year_recid

svm_model <- svm(
  x = X_train_svm,
  y = y_train_svm,
  kernel = "radial",
  probability = TRUE,
  scale = TRUE
)

svm_pred_prob <- attr(
  predict(svm_model, X_test_svm, probability = TRUE),
  "probabilities"
)[, "1"]

svm_pred <- ifelse(svm_pred_prob > 0.5, 1, 0)
svm_pred <- as.factor(svm_pred)

res_svm <- evaluate_model("Nonlinear SVM (RBF)",
                          pred_factor  = svm_pred,
                          pred_score   = svm_pred_prob,
                          y_true_factor = y_test_factor)

# ================================
# 9. Black-box baseline: COMPAS
# ================================
compas_score <- data_test$decile_score    # continuous 1–10
compas_pred  <- data_test$compas_decision # binary 0/1 factor

res_compas <- evaluate_model(
  "COMPAS (decile score, threshold=Low vs Med/High)",
  pred_factor  = compas_pred,
  pred_score   = compas_score,
  y_true_factor = y_test_factor
)

# ================================
# 10. Model 4: XGBoost (full features, with explicit contrasts)
# ================================
features_xgb <- c("sex", "age", "race", "priors_count",
                  "c_charge_degree", "juv_fel_count", "juv_misd_count")
formula_xgb <- as.formula(
  paste("~", paste(features_xgb, collapse = " + "), "- 1")
)

# Explicit contrasts for all factor levels
contrasts_list <- list(
  race            = contrasts(data_train$race, contrasts = FALSE),
  sex             = contrasts(data_train$sex, contrasts = FALSE),
  c_charge_degree = contrasts(data_train$c_charge_degree, contrasts = FALSE)
)

X_train_mat <- model.matrix(formula_xgb, data = data_train,
                            contrasts.arg = contrasts_list)
X_test_mat  <- model.matrix(formula_xgb, data = data_test,
                            contrasts.arg = contrasts_list)

y_train_xgb <- as.numeric(as.character(data_train$two_year_recid))

dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train_xgb)
dtest  <- xgb.DMatrix(data = X_test_mat,  label = y_test_numeric)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",
  eta = 0.1,
  max_depth = 4,
  subsample = 0.7,
  colsample_bytree = 0.7
)

xgb_model <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_prob > 0.5, 1, 0)
xgb_pred <- as.factor(xgb_pred)

res_xgb <- evaluate_model("XGBoost (Full Features)",
                          pred_factor  = xgb_pred,
                          pred_score   = xgb_prob,
                          y_true_factor = y_test_factor)

# ================================
# 11. Summary of model performance
# ================================
cat("\n================== MODEL PERFORMANCE ==================\n")
print_eval(res_lr7)
print_eval(res_lr2)
print_eval(res_svm)
print_eval(res_compas)
print_eval(res_xgb)
cat("=======================================================\n")

cat("\n================== FAIRNESS ANALYSIS ==================\n")
# Prepare test race vector (ensure alignment)
test_race <- data_test$race

evaluate_fairness("COMPAS", compas_pred, y_test_factor, test_race)
evaluate_fairness("LR-7", lr7_pred, y_test_factor, test_race)
evaluate_fairness("SVM", svm_pred, y_test_factor, test_race)
evaluate_fairness("XGBoost", xgb_pred, y_test_factor, test_race)
cat("=======================================================\n")

# ================================
# 12. SHAP analysis for XGBoost
# ================================
pfun <- function(object, newdata) {
  predict(object, as.matrix(newdata))
}

cat("Computing SHAP values (first 200 test observations)...\n")
shap_res <- explain(
  xgb_model,
  X            = X_train_mat,
  pred_wrapper = pfun,
  nsim         = 10,
  newdata      = X_test_mat[1:200, ]
)

if (is.null(colnames(shap_res))) {
  colnames(shap_res) <- colnames(X_train_mat)
}
feature_names <- colnames(shap_res)

# Global feature importance
shap_imp <- data.frame(
  Variable  = feature_names,
  Importance = apply(shap_res, 2, function(x) mean(abs(x)))
) %>%
  arrange(Importance) %>%
  mutate(Variable = factor(Variable, levels = Variable))

p1 <- ggplot(shap_imp, aes(x = Importance, y = Variable)) +
  geom_col(fill = "steelblue") +
  labs(title = "Global Feature Importance (SHAP)",
       subtitle = "Including all racial groups explicitly",
       x = "Mean |SHAP Value|", y = "Features") +
  theme_minimal()

print(p1)
ggsave("shap_importance_full.png", plot = p1, width = 6, height = 4)

# SHAP dependence plot for priors_count
target_feat <- "priors_count"
if (target_feat %in% feature_names) {
  plot_data <- data.frame(
    Feature_Value = X_test_mat[1:200, target_feat],
    SHAP_Value    = as.data.frame(shap_res)[[target_feat]]
  )
  
  p2 <- ggplot(plot_data, aes(x = Feature_Value, y = SHAP_Value)) +
    geom_point(color = "darkorange", alpha = 0.6) +
    geom_smooth(method = "loess", color = "blue", se = FALSE) +
    labs(title = paste("SHAP Dependence:", target_feat),
         x = "Number of Prior Crimes", y = "SHAP Value") +
    theme_minimal()
  
  print(p2)
  ggsave("shap_dependence_full.png", plot = p2, width = 6, height = 4)
}


# ================================
# X. Distribution of COMPAS scores by race
# ================================
plot_data <- df_clean %>%
  filter(race %in% c("African-American", "Caucasian"))

p_compas <- ggplot(plot_data,
                   aes(x = decile_score,
                       fill = race)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("African-American" = "#E69F00",
                               "Caucasian"        = "#56B4E9")) +
  labs(title = "Distribution of COMPAS Scores by Race",
       x = "Decile Score (1=Low, 10=High)",
       y = "Count of Defendants",
       fill = "race") +
  theme_minimal()

print(p_compas)

ggsave("compas_score_dist.png",
       plot   = p_compas,
       width  = 6,
       height = 4,
       dpi    = 300)

