
library(mlr3verse)
library(mlr3proba)
library(mlr3pipelines)
library(tidyverse)
rm(list = ls())
load("./multi_surv_simple/1data/FeatureSelect/xgb_top_tasks_lasso.rdata")
path = "./multi_surv_simple/3result/08top_res_lasso_cv/"
if(!file.exists(path)){
  dir.create(path)
}
trains = xgb_tasks_train
tests = xgb_tasks_test

lrns = c("coxnet","surv_gbm","rsf","surv_xgb")

# wks = 4
# # 加速
# library(future)
# plan("multisession",workers = wks)

file0 = "./multi_surv_simple/3result/07hpo_top_res/"
# file1 = paste0(file0,"coxnet_",lrns,"_random_top")
# file1 = paste0(file0,"gbm_",lrns,"_random_top")
file1 = paste0(file0,"xgb_",lrns,"_random_top")

pre_f <- function(surv_lrn, train_task, test_task) {
  set.seed(180615)
  # 定义5折交叉验证的重采样方案
  rcv = rsmp("repeated_cv", repeats = 5, folds = 5)
  # 使用交叉验证训练模型
  rr = resample(train_task, surv_lrn, rcv)
  # rr$aggregate(msr("surv.cindex"))
  rrp_train = data.table(id = rr$prediction()$row_ids,
                         crank = rr$prediction()$crank) %>%
    group_by(id) %>% summarise(crank = mean(crank))
  
  set.seed(180615)
  surv_lrn$train(train_task)
  class(surv_lrn) <- c(class(surv_lrn), "LearnerSurv")
  # rrp_train = surv_lrn$predict(train_task)
  rrp_test = surv_lrn$predict(test_task)
  
  return(list(rrp_train,rrp_test))
}

# n = 500
for(i in 1:10){
  # i = 2 # 2 - 5
  
  if(i == 1){
    gbm = openxlsx::read.xlsx(paste0(file1[2],i,"_lasso.xlsx")) %>% 
      filter(surv.cindex == max(surv.cindex))
    # rsf_para = openxlsx::read.xlsx(paste0(file1[3],i,"_lasso.xlsx")) %>% 
    #   filter(surv.cindex == max(surv.cindex))
    xgb = openxlsx::read.xlsx(paste0(file1[4],i,"_lasso.xlsx")) %>% 
      filter(surv.cindex == max(surv.cindex))
  } else {
    coxnet_para = openxlsx::read.xlsx(paste0(file1[1],i,"_lasso.xlsx")) %>% 
      filter(surv.cindex == max(surv.cindex))
    coxnet_para = coxnet_para[1,]
    gbm = openxlsx::read.xlsx(paste0(file1[2],i,"_lasso.xlsx")) %>% 
      filter(surv.cindex == max(surv.cindex))
    # rsf_para = openxlsx::read.xlsx(paste0(file1[3],i,"_lasso.xlsx")) %>% 
    #   filter(surv.cindex == max(surv.cindex))
    xgb = openxlsx::read.xlsx(paste0(file1[4],i,"_lasso.xlsx")) %>% 
      filter(surv.cindex == max(surv.cindex))
  }
  
  library(paradox)
  # 创建一个预处理操作
  task_scale <- po("scale", center = T, scale = TRUE)
  # 使用预处理操作拟合训练任务并返回预处理后的任务
  train_task <- task_scale$train(list(trains[[i]]))[[1]]
  # 使用相同的预处理操作转换测试任务
  test_task <- task_scale$predict(list(tests[[i]]))[[1]]
  ## 这里需要注意的是，应该用predict 才能将train的均数标准差移植到test
  ## 如果test还是用train进行标准化，那就是test自己的，此时就会出现范围错位。
  # test_task_1 <- task_scale$train(list(xgb_tasks_test[[1]]))[[1]]
  
  
  surv_gbm = lrn("surv.gbm", id = paste0("surv_gbm_top",i),
                 interaction.depth = gbm$interaction.depth,
                 n.trees = gbm$n.trees,
                 shrinkage = gbm$shrinkage)
  surv_gbm_stack <- as_learner(ppl("distrcompositor", 
                                   learner = surv_gbm, 
                                   estimator = "kaplan",
                                   form = "ph"))
  gbm_pre = pre_f(surv_gbm_stack, train_task, test_task)
  gbm_train = gbm_pre[[1]]
  gbm_test = gbm_pre[[2]]
  
  if(i != 1){
    coxnet   = lrn("surv.glmnet", id = paste0("coxnet_top",i),
                   alpha = coxnet_para$alpha,
                   nlambda = coxnet_para$nlambda)
    coxnet_stack = coxnet$clone()
    coxnet_pre = pre_f(coxnet_stack, train_task, test_task)
    coxnet_train = coxnet_pre[[1]]
    coxnet_test = coxnet_pre[[2]]
  }
  
  # rsf = lrn("surv.rfsrc", id = paste0("rsf_top",i),
  #                ntree = rsf_para$ntree,
  #                mtry = rsf_para$mtry,
  #                nodesize = rsf_para$nodesize,
  #                nsplit = rsf_para$nsplit)
  rsf = lrn("surv.rfsrc", id = paste0("rsf_top",i))
  
  surv_xgb = lrn("surv.xgboost", id = paste0("surv_xgb_top",i),
                 max_depth = xgb$max_depth, 
                 nrounds = xgb$nrounds, 
                 eta = xgb$eta,
                 colsample_bytree = xgb$colsample_bytree,
                 alpha = xgb$alpha,
                 gamma = xgb$gamma,
                 lambda = xgb$lambda,
                 subsample = xgb$subsample)
  surv_xgb_stack <- as_learner(ppl("distrcompositor", 
                                   learner = surv_xgb, 
                                   estimator = "kaplan",
                                   form = "ph"))
  xgb_pre = pre_f(surv_xgb_stack, train_task, test_task)
  xgb_train = xgb_pre[[1]]
  xgb_test = xgb_pre[[2]]
  
  if(i != 1){
    surv_list = list(
      surv_gbm,
      coxnet,
      rsf,
      surv_xgb,
      coxph = lrn("surv.coxph")
    )
    train_stack = cbind(survival_time = train_task$data()$survival_time,
                        status = train_task$data()$status,
                        gbm = gbm_train$crank,
                        coxnet = coxnet_train$crank,
                        # coxph = coxph_train$crank,
                        xgb = xgb_train$crank) %>% 
      as.data.table() %>%  as_task_surv(time = "survival_time",
                                        event = "status",
                                        id = "ensemble")
    test_stack = cbind(survival_time = test_task$data()$survival_time,
                       status = test_task$data()$status,
                       gbm = gbm_test$crank,
                       coxnet = coxnet_test$crank,
                       # coxph = coxph_test$crank,
                       xgb = xgb_test$crank) %>% 
      as.data.table() %>%  as_task_surv(time = "survival_time",
                                        event = "status",
                                        id = "ensemble")
  } else {
    surv_list = list(
      surv_gbm,
      rsf,
      surv_xgb,
      coxph = lrn("surv.coxph", id = "coxph")
    )
    train_stack = cbind(survival_time = train_task$data()$survival_time,
                        status = train_task$data()$status,
                        gbm = gbm_train$crank,
                        xgb = xgb_train$crank) %>% 
      as.data.table() %>%  as_task_surv(time = "survival_time",
                                        event = "status",
                                        id = "ensemble")
    test_stack = cbind(survival_time = test_task$data()$survival_time,
                       status = test_task$data()$status,
                       gbm = gbm_test$crank,
                       xgb = xgb_test$crank) %>% 
      as.data.table() %>%  as_task_surv(time = "survival_time",
                                        event = "status",
                                        id = "ensemble")
  }
  
  library(survival)
  library(survminer)
  library(survex)
  
  surv_total <- vector("list")
  msr_surv <- c("surv.cindex","surv.graf","surv.dcalib",
                "surv.intlogloss","surv.rcll","surv.schmid",
                "surv.calib_alpha","surv.logloss")
  
  
  for(k in 1:length(surv_list)){
    
    # k = 1
    print(k)
    surv_lrn = surv_list[[k]]
    
    if(substr(surv_list[[k]]$id,1,8) == "surv_gbm" | 
       substr(surv_list[[k]]$id,1,8) == "surv_xgb"){
      surv_lrn <- as_learner(ppl(
        "distrcompositor",
        learner = surv_lrn,
        estimator = "kaplan",
        form = "ph"
      ))
    } 
    
    set.seed(180615)
    surv_lrn$train(train_task)
    print(surv_lrn)
    # important!
    class(surv_lrn) <- c(class(surv_lrn), "LearnerSurv") 
    
    
    time1 = Sys.time()
    
    
    set.seed(180615)
    surv_pre = surv_lrn$predict(test_task)
    surv_res = surv_pre$score(msrs(msr_surv)) %>% round(4)
    print(surv_res)
    
    set.seed(180615)
    surv_exp_test <- explain(surv_lrn, 
                             data = test_task$data(),
                             y = Surv(test_task$data()$survival_time, 
                                      test_task$data()$status),
                             label = surv_list[[k]]$id)
    # 计算C-inde
    cindex = c_index(y_true = surv_exp_test$y,
                     risk = surv_exp_test$predict_function(surv_exp_test$model,
                                                           surv_exp_test$data))
    print(cindex)
    time2 = Sys.time()
    print(time2 - time1)
    # 计算integrated Brier score
    surv = surv_exp_test$predict_survival_function(surv_lrn,
                                                   surv_exp_test$data,
                                                   surv_exp_test$times)
    ibs = integrated_brier_score(surv_exp_test$y,
                                 surv = surv,
                                 times = surv_exp_test$times)
    time3 = Sys.time()
    print(time3 - time2)
    # 计算integrated AUC
    auc = integrated_cd_auc(surv_exp_test$y,
                            surv = surv,
                            times = surv_exp_test$times)
    print(ibs)
    print(auc)
    time4 = Sys.time()
    print(time4 - time3)
    
    surv_total[[surv_list[[k]]$id]] = c(lrn = surv_list[[k]]$id,
                                        surv_res, 
                                        cindex = round(cindex,4), 
                                        ibs = round(ibs,4), 
                                        iauc = round(auc,4))
    
  }
  
  coxph_stack = lrn("surv.coxph")
  
  set.seed(180615)
  
  coxph_stack$train(train_stack)
  class(coxph_stack) <- c(class(coxph_stack), "LearnerSurv")
  coxph_pre_stack = coxph_stack$predict(test_stack)
  
  surv_res_stack = coxph_pre_stack$score(msrs(msr_surv)) %>% round(4)
  print(surv_res_stack)
  
  set.seed(180615)
  time1 = Sys.time()
  surv_exp_test_stack <- explain(coxph_stack,
                                 data = test_stack$data(),
                                 y = Surv(test_stack$data()$survival_time, 
                                          test_stack$data()$status),
                                 label = "coxph")
  # 计算C-inde
  cindex_stack = c_index(y_true = surv_exp_test_stack$y,
                         risk = surv_exp_test_stack$predict_function(surv_exp_test_stack$model,
                                                                     surv_exp_test_stack$data))
  print(cindex_stack)
  time2 = Sys.time()
  print(time2 - time1)
  # 计算integrated Brier score
  surv_stack = surv_exp_test_stack$predict_survival_function(coxph_stack,
                                                             surv_exp_test_stack$data,
                                                             surv_exp_test_stack$times)
  ibs_stack = integrated_brier_score(surv_exp_test_stack$y,
                                     surv = surv_stack,
                                     times = surv_exp_test_stack$times)
  time3 = Sys.time()
  print(time3 - time2)
  # 计算integrated AUC
  auc_stack = integrated_cd_auc(surv_exp_test_stack$y,
                                surv = surv_stack,
                                times = surv_exp_test_stack$times)
  
  print(cindex_stack)
  print(ibs_stack)
  print(auc_stack)
  
  surv_total[["stacking"]] = c(lrn = "Stacking",
                               surv_res_stack, 
                               cindex = round(cindex_stack,4), 
                               ibs = round(ibs_stack,4), 
                               iauc = round(auc_stack,4))
  
  surv_total_result  = do.call(rbind, surv_total) %>% as.data.frame()
  save(surv_total, surv_total_result,
       file = paste0(path,"xgb_surv_top",i,"_lasso.rdata"))
  openxlsx::write.xlsx(surv_total_result,
                       file = paste0(path,
                                     "xgb_surv_top",i,"_lasso.xlsx"))
}

