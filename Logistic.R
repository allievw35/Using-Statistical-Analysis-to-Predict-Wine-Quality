
library(ggplot2)
library(dplyr)
library(corrplot)
library(readr)
library(reshape2)
library(gridExtra)
library(caret)

wine <- read.csv("wine-quality-white-and-red.csv")
str(wine)


# Convert quality to factor for some plots
wine$quality_factor <- as.factor(wine$quality)


#statistics
summary(wine)


# 1. Quality distribution
ggplot(wine, aes(x = quality)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Wine Quality Distribution", x = "Quality", y = "Count") +
  theme_minimal()


# Reshape data for boxplot grid

wine_long <- melt(wine, id.vars = "quality", 
                  measure.vars = c("alcohol", "volatile.acidity", "residual.sugar", 
                                   "sulphates", "citric.acid", "pH", "chlorides", "density"))

# Plot
ggplot(wine_long, aes(x = as.factor(quality), y = value, fill = as.factor(quality))) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  labs(title = "Boxplots of Features by Wine Quality", x = "Wine Quality", y = "Feature Value") +
  theme_minimal()


cor_matrix <- cor(numeric_data)

# Create heatmap with correlation values
corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black", # add correlation values
         number.cex = 0.7,       # size of text
         tl.cex = 0.8,           # size of labels
         col = colorRampPalette(c("darkred", "white", "darkgreen"))(200),
         tl.col = "black", diag = FALSE)





library(nnet)
library(dplyr)

# Create 3-category target variable
wine$class <- cut(wine$quality, 
                  breaks = c(0, 5, 6, 10), 
                  labels = c("Low", "Medium", "High"),
                  right = TRUE)
wine$class <- as.factor(wine$class)

# Check class balance
table(wine$class)

# Split into train/test sets
set.seed(123)
sample_index <- sample(1:nrow(wine), 0.8 * nrow(wine))
train <- wine[sample_index, ]
test  <- wine[-sample_index, ]

# Fit multinomial logistic regression
multi_logit <- multinom(class ~ alcohol + volatile.acidity + sulphates, data = train)

# Predict on test set
preds <- predict(multi_logit, newdata = test)

# Accuracy
acc <- mean(preds == test$class)
cat("Multinomial Logistic Regression Accuracy:", round(acc, 3), "\n")

# Confusion matrix
table(Predicted = preds, Actual = test$class)




set.seed(123)
cv_control <- trainControl(method = "cv", number = 10)

# -------------------------------
# Ridge Regression (alpha = 0)
ridge_model <- train(
  quality ~ ., 
  data = wine,
  method = "glmnet",
  trControl = cv_control,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4, 1, length = 100))
)
print(ridge_model)

# -------------------------------
# Lasso Regression (alpha = 1)
lasso_model <- train(
  quality ~ ., 
  data = wine,
  method = "glmnet",
  trControl = cv_control,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, -1, length = 100))
)
print(lasso_model)

# -------------------------------
# Regression Tree
# Clean data (remove non-numeric columns if present)
wine_tree <- wine %>%
  select(-class, -quality_factor, -type)  # remove only if these exist

cp_grid <- expand.grid(cp = seq(0.001, 0.05, length.out = 10))

tree_model <- train(
  quality ~ ., 
  data = wine_tree,
  method = "rpart",
  trControl = cv_control,
  tuneGrid = cp_grid
)
print(tree_model)

# -------------------------------
# Random Forest
rf_model <- train(
  quality ~ ., 
  data = wine,
  method = "rf",
  trControl = cv_control,
  ntree = 10
)
print(rf_model)

# -------------------------------
# Compare all models
results <- resamples(list(
  Ridge = ridge_model,
  Lasso = lasso_model,
  Tree = tree_model,
  RF = rf_model
))

# Summarize cross-validated RMSE & R²
summary(results)

# Boxplot to compare RMSE visually
bwplot(results, metric = "RMSE")


