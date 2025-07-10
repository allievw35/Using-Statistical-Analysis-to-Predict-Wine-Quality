
# 0. Install packages if needed
# install.packages(c("glmnet","caret","Metrics","reshape2","ggplot2"))

# 1. Load libraries and set seed
library(glmnet)
library(caret)
library(Metrics)
library(reshape2)
library(ggplot2)
set.seed(123)

# 2. Load data
wine <- read.csv("wine-quality-white-and-red.csv", header = TRUE, sep = ",")
# Quick inspect
str(wine)
summary(wine)

# 3. Train/test split
idx   <- createDataPartition(wine$quality, p = 0.8, list = FALSE)
train <- wine[idx, ]
test  <- wine[-idx, ]

# 4. Prepare matrices
X_train <- model.matrix(quality ~ ., train)[, -1]
y_train <- train$quality
X_test  <- model.matrix(quality ~ ., test)[, -1]
y_test  <- test$quality

# 5. 10-fold CV Ridge & Lasso
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)

best_lambda_ridge <- cv_ridge$lambda.min
best_lambda_lasso <- cv_lasso$lambda.min

# 6. Extract & rank coefficients
ridge_coef <- as.matrix(coef(cv_ridge, s = best_lambda_ridge))
lasso_coef <- as.matrix(coef(cv_lasso, s = best_lambda_lasso))
ridge_vals <- ridge_coef[-1, , drop = TRUE]
names(ridge_vals) <- rownames(ridge_coef)[-1]
lasso_vals <- lasso_coef[-1, , drop = TRUE]
names(lasso_vals) <- rownames(lasso_coef)[-1]
ridge_ranked <- sort(abs(ridge_vals), decreasing = TRUE)
lasso_ranked <- sort(abs(lasso_vals[lasso_vals != 0]), decreasing = TRUE)

# 7. Test-set performance
pred_ridge <- predict(cv_ridge, X_test, s = best_lambda_ridge)
pred_lasso <- predict(cv_lasso, X_test, s = best_lambda_lasso)
rmse_ridge <- rmse(y_test, pred_ridge)
r2_ridge   <- cor(y_test, pred_ridge)^2
rmse_lasso <- rmse(y_test, pred_lasso)
r2_lasso   <- cor(y_test, pred_lasso)^2

cat("Ridge → λ =", round(best_lambda_ridge, 4),
    "| RMSE =", round(rmse_ridge, 3),
    "| R² =", round(r2_ridge, 3), "\n")
cat("Lasso → λ =", round(best_lambda_lasso, 4),
    "| RMSE =", round(rmse_lasso, 3),
    "| R² =", round(r2_lasso, 3), "\n")

# 8. Reporting
# Performance table
perf_tbl <- data.frame(
  Model  = c("Ridge", "Lasso"),
  Lambda = c(best_lambda_ridge, best_lambda_lasso),
  RMSE   = c(rmse_ridge, rmse_lasso),
  R2     = c(r2_ridge, r2_lasso)
)
print(perf_tbl)

# Top 10 predictors table
coef_tbl <- data.frame(
  Predictor = names(ridge_ranked)[1:10],
  Ridge     = round(ridge_ranked[1:10], 3),
  Lasso     = round(lasso_ranked[1:10], 3)
)
print(coef_tbl)

# Bar-plot of absolute coefficients
coef_long <- melt(coef_tbl, id.vars = "Predictor",
                  variable.name = "Model", value.name = "AbsCoef")
ggplot(coef_long, aes(x = reorder(Predictor, AbsCoef), y = AbsCoef, fill = Model)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Top 10 Predictors by Absolute Coefficient",
       x = NULL, y = "Absolute Coefficient") +
  theme_minimal()
