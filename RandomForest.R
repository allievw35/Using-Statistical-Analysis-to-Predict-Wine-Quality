# Install required package (run once)
install.packages("randomForest", dependencies=TRUE)

# Load libraries
library(caret)
library(randomForest)

# 1. Grab the already-imported combined dataset
wine <- wine_quality_white_and_red

# 2. Clean column names and types
colnames(wine) <- make.names(colnames(wine))
wine$type      <- factor(wine$type)

# 3. Split into training (80%) and test (20%)
set.seed(123)
train_idx  <- createDataPartition(wine$quality, p = 0.8, list = FALSE)
train_data <- wine[train_idx, ]
test_data  <- wine[-train_idx, ]

# 4. Fit a Random Forest to predict quality
set.seed(123)
rf_model <- randomForest(
  formula    = quality ~ . - type,
  data       = train_data,
  ntree      = 500,
  importance = TRUE
)

# 5. Inspect OOB error and variable importance
print(rf_model)            # OOB MSE and % variance explained
plot(rf_model, main="OOB Error vs. Trees")
varImpPlot(rf_model, main="Variable Importance")

# 6. Evaluate on the test set
preds       <- predict(rf_model, newdata = test_data)
test_mse    <- mean((test_data$quality - preds)^2)
test_rmse   <- sqrt(test_mse)
ss_res      <- sum((test_data$quality - preds)^2)
ss_tot      <- sum((test_data$quality - mean(test_data$quality))^2)
test_r2     <- 1 - ss_res/ss_tot

cat("Test MSE:   ", round(test_mse, 4), "\n")
cat("Test RMSE:  ", round(test_rmse, 4), "\n")
cat("Test R²:    ", round(test_r2,   4), "\n")

