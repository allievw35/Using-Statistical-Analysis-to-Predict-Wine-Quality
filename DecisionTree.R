# Install required packages (run once)
install.packages(c("caret", "rpart", "rpart.plot"), dependencies=TRUE)

# Load libraries
library(caret)
library(rpart)
library(rpart.plot)

# 1. Load the imported data
wine <- wine_quality_white_and_red

# 2. Check it loaded correctly
str(wine)
head(wine)


# 2. Clean column names and types
colnames(wine) <- make.names(colnames(wine))
wine$type      <- factor(wine$type)

# 3. Split into training (80%) and test (20%) with seed 123
set.seed(123)
train_idx  <- createDataPartition(wine$quality, p = 0.8, list = FALSE)
train_data <- wine[train_idx, ]
test_data  <- wine[-train_idx, ]

# 4. Fit full (unpruned) regression tree
tree_full <- rpart(
  quality ~ . - type,
  data    = train_data,
  method  = "anova",
  control = rpart.control(cp = 0)
)

# 5. Prune to optimal complexity parameter
opt_cp       <- tree_full$cptable[ which.min(tree_full$cptable[,"xerror"]), "CP" ]
tree_pruned  <- prune(tree_full, cp = opt_cp)

# 6. Plot the pruned tree
rpart.plot(
  tree_pruned,
  type          = 4,
  extra         = 101,
  fallen.leaves = TRUE,
  main          = paste("Pruned Regression Tree (CP =", round(opt_cp,4), ")")
)

# 7. Evaluate on the test set
preds      <- predict(tree_pruned, newdata = test_data)
test_mse   <- mean((test_data$quality - preds)^2)
test_rmse  <- sqrt(test_mse)

cat("Test MSE: ",  round(test_mse,  4), "\n")
cat("Test RMSE:", round(test_rmse, 4), "\n")
