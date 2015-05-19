import xgboost as xgb

def prepare_xgb_matrixes(X_all, y_all, eval_size = 0.20):
    train_size = 1. - eval_size
    Xy_all_xgb = xgb.DMatrix(X_all, label=y_all)
    X_train_xgb = xgb.DMatrix(X_all[:int(train_size*len(X_all))], label=y_all[:int(train_size*len(X_all))])
    X_val_xgb   = xgb.DMatrix(X_all[int(train_size*len(X_all)):], label=y_all[int(train_size*len(X_all)):])
    return Xy_all_xgb, X_train_xgb, X_val_xgb