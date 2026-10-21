## =========================================================
## Load data and define feature names
## =========================================================

water <- read.csv("data/water.csv")

feature_names <- c("ph", "hardness", "solids", "chloramines",
                   "sulfate", "conductivity", "carbon",
                   "trihalomethanes", "turbidity")


## =========================================================
## Q1 – How many missing values does the dataset contain?
## =========================================================

q1 <- sum(is.na(water))
q1


## =========================================================
## Q2 – Train/test split + mean imputation + mean ph in test
## =========================================================

## Split
train <- water[1:3000, ]
test  <- water[3001:nrow(water), ]

## Means of features from training set (ignoring NAs)
train_means <- sapply(train[ , feature_names], function(x) mean(x, na.rm = TRUE))

## Impute missing values in train and test with training means
for (name in feature_names) {
  m <- train_means[name]
  train[is.na(train[ , name]), name] <- m
  test[is.na(test[ , name]),  name]  <- m
}

## Mean ph in test after imputation
q2 <- mean(test$ph)
q2


## =========================================================
## Q3 – Gaussian parameters for ph | potability
## =========================================================

## Split training data by class
train_pos <- subset(train, potability == 1)
train_neg <- subset(train, potability == 0)

## Means
mean_ph_pos <- mean(train_pos$ph)
mean_ph_neg <- mean(train_neg$ph)

## Standard deviations
sd_ph_pos <- sd(train_pos$ph)
sd_ph_neg <- sd(train_neg$ph)

## Table for report
q3_table <- data.frame(
  Statistic    = c("Mean", "Standard Deviation"),
  Potability_1 = c(mean_ph_pos, sd_ph_pos),
  Potability_0 = c(mean_ph_neg, sd_ph_neg)
)
q3_table

## SD for potable water
q3 <- sd_ph_pos
q3


## =========================================================
## Q4 – Plot fitted Gaussians for ph
## =========================================================

x <- seq(0, 14, length.out = 500)

dens_pos <- dnorm(x, mean = mean_ph_pos, sd = sd_ph_pos)
dens_neg <- dnorm(x, mean = mean_ph_neg, sd = sd_ph_neg)

plot(x, dens_pos, type = "l",
     col = "darkgreen", lwd = 2,
     xlab = "ph",
     ylab = "Density",
     xlim = c(0, 14),
     main = "Density plot of pH by Potability")
lines(x, dens_neg, col = "red", lwd = 2)
legend("topright",
       legend = c("Potable (1)", "Non-potable (0)"),
       col = c("darkgreen", "red"),
       lwd = 2, bty = "n")


## =========================================================
## Q5 – Interpretation (in report, not in code)
## =========================================================
## Answer written in Word/report.


## =========================================================
## Q6 – Gaussian parameters for chloramines | potability
## =========================================================

mean_chl_pos <- mean(train_pos$chloramines)
mean_chl_neg <- mean(train_neg$chloramines)

sd_chl_pos <- sd(train_pos$chloramines)
sd_chl_neg <- sd(train_neg$chloramines)

q6_table <- data.frame(
  Statistic    = c("Mean", "Standard Deviation"),
  Potability_1 = c(mean_chl_pos, sd_chl_pos),
  Potability_0 = c(mean_chl_neg, sd_chl_neg)
)
q6_table

q6 <- sd_chl_pos
q6


## =========================================================
## Q7 – GNB confusion matrix (probs) + P(false positive)
## =========================================================

## Class-conditional parameters for ALL features
means_pos_all <- sapply(train_pos[ , feature_names], mean)
sds_pos_all   <- sapply(train_pos[ , feature_names], sd)
means_neg_all <- sapply(train_neg[ , feature_names], mean)
sds_neg_all   <- sapply(train_neg[ , feature_names], sd)

## Priors
prior_pos <- mean(train$potability == 1)
prior_neg <- 1 - prior_pos

## 1D Gaussian log-density
gaussian_log_density <- function(x, mean, sd) {
  -0.5 * log(2 * pi) - log(sd) - 0.5 * ((x - mean)/sd)^2
}

## Posterior P(y=1 | x) under GNB
gnb_proba_pos <- function(x_vec) {
  log_lik_pos <- 0
  log_lik_neg <- 0
  
  for (j in seq_along(feature_names)) {
    m_pos <- means_pos_all[j]
    s_pos <- sds_pos_all[j]
    m_neg <- means_neg_all[j]
    s_neg <- sds_neg_all[j]
    
    log_lik_pos <- log_lik_pos + gaussian_log_density(x_vec[j], m_pos, s_pos)
    log_lik_neg <- log_lik_neg + gaussian_log_density(x_vec[j], m_neg, s_neg)
  }
  
  log_post_pos <- log(prior_pos) + log_lik_pos
  log_post_neg <- log(prior_neg) + log_lik_neg
  
  max_log <- max(log_post_pos, log_post_neg)
  num_pos <- exp(log_post_pos - max_log)
  num_neg <- exp(log_post_neg - max_log)
  
  p_pos <- num_pos / (num_pos + num_neg)
  return(p_pos)
}

## Confusion matrix on training data
X_train <- as.matrix(train[ , feature_names])
y_train <- train$potability

probs_train_gnb <- apply(X_train, 1, gnb_proba_pos)
yhat_train_gnb  <- ifelse(probs_train_gnb >= 0.5, 1, 0)

TP <- sum(y_train == 1 & yhat_train_gnb == 1)
FP <- sum(y_train == 0 & yhat_train_gnb == 1)
TN <- sum(y_train == 0 & yhat_train_gnb == 0)
FN <- sum(y_train == 1 & yhat_train_gnb == 0)

## Counts table (internal)
train_conf_gnb_counts <- matrix(c(TP, FP, FN, TN),
                                nrow = 2, byrow = TRUE,
                                dimnames = list(
                                  "True class" = c("1", "0"),
                                  "Predicted"  = c("1", "0")
                                ))
train_conf_gnb_counts

## Probability table (for report)
N_train <- length(y_train)
train_conf_gnb_probs <- train_conf_gnb_counts / N_train
train_conf_gnb_probs

## Probability of a false positive: P(pred=1 | true=0)
q7 <- FP / (FP + TN)
q7


## =========================================================
## Q8 – Z-normalised pH for first test point
## =========================================================

norm_means <- sapply(train[ , feature_names], mean)
norm_sds   <- sapply(train[ , feature_names], sd)

train_norm <- train
test_norm  <- test

for (name in feature_names) {
  m <- norm_means[name]
  s <- norm_sds[name]
  train_norm[ , name] <- (train[ , name] - m) / s
  test_norm[ , name]  <- (test[ , name]  - m) / s
}

q8 <- test_norm$ph[1]
q8


## =========================================================
## Q9 – Train MLP and get weight sulfate -> hidden neuron 4
## =========================================================
## a) Helper functions

weight_matrix <- function(m, n){
  c <- sqrt(2 / n)
  matrix(seq(from = -c, to = c, length.out = m * n),
         nrow = m, ncol = n)
}

relu <- function(z){
  out <- pmax(0, z)
  dim(out) <- dim(z)
  out
}

sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

## b) MLP function – same structure as in the assignment

mlp <- function(x, y, m2, m3, epochs = 4000, lr = 0.1){
  
  x <- t(x)                # x: (features x samples)
  y <- matrix(y, nrow = 1) # y: (1 x samples)
  
  m1 <- nrow(x)            # number of input features
  m4 <- nrow(y)            # number of outputs (1)
  n  <- ncol(x)            # number of samples
  
  # w2: hidden layer weights (hidden_units x input_features)
  # w3: output layer weights (output_units x hidden_units)
  w2 <- weight_matrix(m2, m1)
  w3 <- weight_matrix(m3, m2)
  
  b2 <- numeric(m2)        # bias hidden
  b3 <- numeric(m3)        # bias output
  
  losses <- numeric(epochs)
  
  for(i in 1:epochs){
    # ----- forward -----
    z2 <- w2 %*% x + b2          # (m2 x n)
    a2 <- relu(z2)               # (m2 x n)
    
    z3 <- w3 %*% a2 + b3         # (m3 x n)
    a3 <- sigmoid(z3)            # (m3 x n)
    
    # cross-entropy loss
    epsilon <- 1e-15
    a3_clip <- pmin(pmax(a3, epsilon), 1 - epsilon)
    loss <- -mean(y * log(a3_clip) + (1 - y) * log(1 - a3_clip))
    losses[i] <- loss
    
    # ----- backward -----
    delta3 <- a3 - y                     # (m3 x n)
    dz2    <- (z2 > 0)                   # (m2 x n), ReLU derivative
    delta2 <- (t(w3) %*% delta3) * dz2   # (m2 x n)
    
    dw3 <- (1 / n) * delta3 %*% t(a2)    # (m3 x m2)
    dw2 <- (1 / n) * delta2 %*% t(x)     # (m2 x m1)
    db3 <- (1 / n) * rowSums(delta3)     # length m3
    db2 <- (1 / n) * rowSums(delta2)     # length m2
    
    # gradient step
    w3 <- w3 - lr * dw3
    w2 <- w2 - lr * dw2
    b3 <- b3 - lr * db3
    b2 <- b2 - lr * db2
  }
  
  list(w2 = w2, w3 = w3, b2 = b2, b3 = b3, losses = losses)
}

## c) Prepare inputs for the MLP using YOUR vars

X_train_mlp <- as.matrix(train_norm[ , feature_names])
y_mlp       <- as.numeric(train$potability)

## d) Train the MLP: 10 hidden units, 1 output neuron

mlp_model <- mlp(X_train_mlp, y_mlp, m2 = 10, m3 = 1,
                 epochs = 4000, lr = 0.1)

## e) Weight from 'sulfate' (input) to hidden neuron 4

idx_sulfate <- which(feature_names == "sulfate")
q9 <- mlp_model$w2[4, idx_sulfate]
q9



## =========================================================
## Q10 – Loss curve of MLP
## =========================================================

loss_history <- mlp_model$losses
epochs <- 1:length(loss_history)

old_par <- par(mar = c(5, 5, 4, 2) + 0.1)

plot(
  x = epochs,
  y = loss_history,
  type = "l",
  lwd  = 2.5,
  xlab = "Epoch",
  ylab = "Average Cross Entropy Loss",
  main = "Loss Curve During Gradient Descent",
  ylim = c(min(loss_history) - 0.01,
           max(loss_history) + 0.01),
  
  # >>> Make axis labels bigger <<<
  cex.axis = 1.3,   # axis numbers (y-axis + x-axis)
  cex.lab  = 1.3,   # axis titles
  cex.main = 1.4    # plot title
)

grid(col = "grey85", lty = 1)

par(old_par)
## =========================================================
## Q11 – MLP train confusion matrix + P(false negative)

# 1. Forward pass on the TRAIN SET using mlp_model
X_train_mlp <- as.matrix(train_norm[ , feature_names, drop = FALSE])  # N x 9
y_train_mlp <- train$potability                                      # length N
N <- nrow(X_train_mlp)

# unpack trained parameters from Q9
w2 <- mlp_model$w2   # 10 x 9
w3 <- mlp_model$w3   # 1 x 10
b2 <- mlp_model$b2   # length 10
b3 <- mlp_model$b3   # length 1

x  <- t(X_train_mlp)               # 9 x N
z2 <- w2 %*% x + b2                # 10 x N
a2 <- relu(z2)                     # 10 x N
z3 <- w3 %*% a2 + b3               # 1 x N
yhat_prob <- as.numeric(sigmoid(z3))   # length N

yhat_class <- ifelse(yhat_prob >= 0.5, 1, 0)

# 2. Confusion matrix (counts) – Predicted = rows, True = columns
TP_mlp <- sum(y_train_mlp == 1 & yhat_class == 1)
FP_mlp <- sum(y_train_mlp == 0 & yhat_class == 1)
FN_mlp <- sum(y_train_mlp == 1 & yhat_class == 0)
TN_mlp <- sum(y_train_mlp == 0 & yhat_class == 0)

train_conf_mlp_counts <- matrix(
  c(TP_mlp, FP_mlp,   # row: Pred = 1
    FN_mlp, TN_mlp),  # row: Pred = 0
  nrow = 2, byrow = TRUE,
  dimnames = list(
    "Predicted class" = c("1", "0"),
    "True class"      = c("1", "0")
  )
)
train_conf_mlp_counts

# 3. Confusion matrix (probabilities, joint)
total_mlp <- TP_mlp + FP_mlp + FN_mlp + TN_mlp
train_conf_mlp_probs <- train_conf_mlp_counts / total_mlp
train_conf_mlp_probs

# 4. q11 = probability of a false negative (Pred = 0, True = 1)
q11 <- train_conf_mlp_probs["0", "1"]
q11


## =========================================================
## Q12 – Bar plot of weights of hidden neuron 10
## 
weights_h10 <- mlp_model$w2[10, ]
names(weights_h10) <- feature_names

# Give more bottom space so BOTH labels fit nicely
old_par <- par(mar = c(11, 6, 4, 2))   # bottom, left, top, right

bp <- barplot(weights_h10,
              main = "Weights of Hidden Neuron 10",
              xlab = "",            # add manually after
              ylab = "Weight",
              las  = 2,
              col  = "lightgray",
              border = NA,
              cex.names = 0.9)      # makes names fit cleanly

# Place x-axis label centered and at the right vertical position
mtext("Input feature", side = 1, line = 8, cex = 1.1)

par(old_par)


## =========================================================
## Q14 – New measurement: probabilities (GNB & MLP) + graph
## =========================================================
# New measurement (raw scale)
new_x <- c(
  ph              = 8,
  hardness        = 187,
  solids          = 21751,
  chloramines     = 9,
  sulfate         = 318,
  conductivity    = 412,
  carbon          = 13,
  trihalomethanes = 61,
  turbidity       = 5
)

# Ensure same order as feature_names
new_x <- new_x[feature_names]

## --- GNB probability ---
p_pos_gnb <- gnb_proba_pos(new_x)

## --- MLP probability (using mlp_model) ---

# Normalise with training stats from Q8
new_x_norm <- (new_x - norm_means) / norm_sds

# Shape: features x 1 (same orientation as in mlp())
x_new <- matrix(new_x_norm, nrow = length(feature_names), ncol = 1)

# Unpack parameters
w2 <- mlp_model$w2
w3 <- mlp_model$w3
b2 <- mlp_model$b2
b3 <- mlp_model$b3

# Forward pass
z2_new <- w2 %*% x_new + b2      # 10 x 1
a2_new <- relu(z2_new)           # 10 x 1
z3_new <- w3 %*% a2_new + b3     # 1 x 1
p_pos_mlp <- as.numeric(sigmoid(z3_new))

# Store MLP probability
q14 <- p_pos_mlp

# Table for the report
Q14_table <- data.frame(
  Model       = c("GNB", "MLP"),
  Probability = c(p_pos_gnb, p_pos_mlp)
)
Q14_table

## =========================================================
## Q15 – Test accuracy (GNB vs MLP) + table
## =========================================================
## --- GNB on test set (raw, not normalised) ---
X_test <- as.matrix(test[ , feature_names])
y_test <- test$potability

probs_test_gnb <- apply(X_test, 1, gnb_proba_pos)
yhat_test_gnb  <- ifelse(probs_test_gnb >= 0.5, 1, 0)

acc_gnb <- mean(yhat_test_gnb == y_test)

## --- MLP on test set (normalised, using mlp_model) ---

# Normalised test features (you already have test_norm from Q8)
X_test_mlp <- as.matrix(test_norm[ , feature_names, drop = FALSE])  # N_test x 9

# Unpack parameters from trained MLP
w2 <- mlp_model$w2   # 10 x 9
w3 <- mlp_model$w3   # 1 x 10
b2 <- mlp_model$b2   # length 10
b3 <- mlp_model$b3   # length 1

# Forward pass on test set
x_test <- t(X_test_mlp)             # 9 x N_test
z2_tst <- w2 %*% x_test + b2        # 10 x N_test
a2_tst <- relu(z2_tst)              # 10 x N_test
z3_tst <- w3 %*% a2_tst + b3        # 1 x N_test
prob_test_mlp <- as.numeric(sigmoid(z3_tst))  # length N_test

yhat_test_mlp <- ifelse(prob_test_mlp >= 0.5, 1, 0)

acc_mlp <- mean(yhat_test_mlp == y_test)

# Store MLP test accuracy
q15 <- acc_mlp
q15

## Accuracy table for report
Q15_table <- data.frame(
  Model    = c("GNB", "MLP"),
  Accuracy = c(acc_gnb, acc_mlp)
)
Q15_table
