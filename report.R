# classification summative
# library download
install.packages(c("readr", "DataExplorer", "skimr", "mlr3verse"))
install.packages("psych")
install.packages("data.table")
install.packages("mlr3verse")
install.packages("randomForestSRC")
install.packages("mlr3extralearners")
install.packages("ranger")
install.packages("xgboost")
install.packages("rpart.plot")

# data preprocessing
# read data
bank_data <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")

# the summary of data
library("skimr")
skim(bank_data)

# remove invalid values
invalid_exp_index <- c(which(bank_data["Experience"] < 0))
#bank_data[c(which(bank_data["Experience"] < 0)),]
bank_data_copy <- bank_data
bank_data_copy <- as.matrix(bank_data_copy)
invalid_exp_index
new_data <- bank_data_copy[-invalid_exp_index,]
invalid_zipcode_index <- c(which(new_data[,"ZIP.Code"] < 90000))
new_data <- new_data[-invalid_zipcode_index,]

# the summary of new data
skim(new_data)

# loan pie chart
loan <- c(length(which(new_data[,"Personal.Loan"] == 1)), length(which(new_data[,"Personal.Loan"] == 0)))
lable_loan <- c("accept loan", "not accept loan")
piepercent<- round(100*loan/sum(loan), 1)
pie(loan, labels = piepercent, main = "Loan Pie Chart",col = rainbow(length(loan)))
legend("topright", c("accept loan", "not accept loan"), cex = 0.8, fill = rainbow(length(loan)))

# EDA
# boxplot
library("DataExplorer")
DataExplorer::plot_boxplot(new_data, by = "Personal.Loan")

# correlation plot
library("psych")
corPlot(new_data)

#model fitting
library("data.table")
library("mlr3verse")
library("randomForestSRC")
library("ranger")
library("mlr3extralearners")
library("xgboost")

# change number to a factor
df_data <- as.data.frame(new_data)
df_data$Education = as.factor(df_data$Education)
df_data$Personal.Loan = as.factor(df_data$Personal.Loan)
df_data$Securities.Account = as.factor(df_data$Securities.Account)
df_data$CD.Account = as.factor(df_data$CD.Account)
df_data$Online = as.factor(df_data$Online)
df_data$CreditCard = as.factor(df_data$CreditCard)

# set seed for reproducibility
set.seed(21) 
# set a new task
loan_task <- TaskClassif$new(id = "BankLoan",
                             backend = df_data, 
                             target = "Personal.Loan",
                             positive = "1")

# 5 fold cross validation
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loan_task)

#build models
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_rf <- lrn("classif.rfsrc", predict_type = "prob")
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
lrn_naive_bayes <- lrn("classif.naive_bayes", predict_type = "prob")
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

# cart cv and cp
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10) # xval is number of cross-validation

res_cart_cv <- resample(loan_task, lrn_cart_cv, cv5, store_models = TRUE)
for (i in 1:5) {
  rpart::plotcp(res_cart_cv$learners[[i]]$model)
}

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.013)

# model set
res <- benchmark(data.table(
  task       = list(loan_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_rf,
                    lrn_cart_cp,
                    lrn_log_reg,
                    lrn_naive_bayes,
                    pl_xgb),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

# random forest
lrn_rf <- lrn("classif.rfsrc", predict_type = "prob", importance = 'TRUE')
res_rf <- resample(loan_task, lrn_rf, cv5, store_models = TRUE)
# ntree for rf
rf_model <- res_rf$learners[[2]]$model
rf_model
plot(rf_model)

# plot importance of features
important_scores <- res_rf$learners[[1]]$importance()
for (i in 2:5) {
  important_scores <- important_scores + res_rf$learners[[i]]$importance()
}
important_scores <- important_scores / 5
lables <- c("Income", "Education", "Family", "CCAvg", "CD.Account", "Mortgage", "Age", "Experience", "ZIP.Code", "CreditCard", "Securities.Account", "Online")
important_scores["others"] <- important_scores_1["Age"] + important_scores_1["Experience"] + important_scores_1["ZIP.Code"] + 
  important_scores_1["CreditCard"] + important_scores_1["Securities.Account"] + important_scores_1["Online"]
scores_sort <- sort(important_scores_1, decreasing = TRUE)
piepercent<- round(100*scores_sort, 1)
pie(scores_sort[1:7], labels = piepercent, main = "Importance of Different Features",col = rainbow(7))
legend("topright", c("Income", "Education", "Family", "CCAvg", "CD.Account", "Mortgage", "others"), cex = 0.8, fill = rainbow(7))
pie(scores_sort[1:7], main = "importance of different features")

# tree plot
library("rpart.plot")
trees <- res$resample_result(2)
tree1 <- trees$learners[[3]]
tree1_rpart <- tree1$model
rpart.plot(tree1_rpart)

# tuning random forest
tune_ps_ranger <- ps(
  mtry = p_int(lower = 1, upper = 6)
)
evals_trm = trm("evals", n_evals = 25)

instance_ranger <- TuningInstanceSingleCrit$new(
  task = loan_task,
  learner = lrn_rf,
  resampling = cv5,
  measure = msr("classif.ce"),
  search_space = tune_ps_ranger,
  terminator = evals_trm
)
tuner <- tnr("grid_search", resolution = 5)
# suppress output
lgr::get_logger("bbotk")$set_threshold("warn")
lgr::get_logger("mlr3")$set_threshold("warn")

tuner$optimize(instance_ranger) 

# turn output back on
lgr::get_logger("bbotk")$set_threshold("info")
lgr::get_logger("mlr3")$set_threshold("info")

instance_ranger$result_learner_param_vals

#refit
set.seed(12)
lrn_rf <- lrn("classif.rfsrc", predict_type = "prob", importance = 'TRUE')
lrn_rf$param_set$values = list(mtry = 3, ntree = 24)

res_rf <- resample(loan_task, lrn_rf, cv5, store_models = TRUE)

res_rf$aggregate(list(msr("classif.ce"),
                      msr("classif.acc"),
                      msr("classif.auc"),
                      msr("classif.fpr"),
                      msr("classif.fnr")))

