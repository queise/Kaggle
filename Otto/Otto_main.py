from __future__ import division
from load_write_data import *
from transform_data import *
from run_classifier import *
from functions import *

def main():
    # Examples of usage:
    #
    # 1) Train and submit a single model
    run_all(eta = 0.005, min_child_weight = 3, seednum=1)
    #
    # 2) CV score on a grid of parameters (scores printed on file 0results-xgb-grid):
    #for eta in [0.010, 0.005]:
    #   for max_depth in [10,11]:
    #       for min_child_weight in [3,4]:
    #           run_all(eta = eta, max_depth = max_depth, min_child_weight = min_child_weight,
    #                   SINGLE = False, CV_ONE = True, SUB = False)
    #



def run_all(# Parameters of xgb classifier:
            eta = 0.020, max_depth = 10, min_child_weight = 4, colsample_bytree = 0.8, gamma = 1,
            # Options for data transformation:
            NEWF=False, LOG=True, SCA=True, PCA=False, INTE=False,
            # Options for type of run:
            SINGLE = True, CV_ONE = False, GRID = False, CV4ENS = False, SUB = True, ALLTRAINSET = False, LOADM = False,
            best_n_iter = None, seednum=1):

    np.random.seed(seednum*1313)

    # loading train data
    X_all, y_all, ids_all, encoder = load_train_data(subsample=False)

    # transforming data
    X_all, str_opt, scaler = transform_data(X_all, NEWF, LOG, SCA, PCA, INTE)

    # preparing xgb classifier:
    Xy_all_xgb, X_train_xgb, X_val_xgb = prepare_xgb_matrixes(X_all, y_all, eval_size = 0.05)
    param = {'nthread':4, 'silent':1,
             'max_depth':max_depth, 'eta':eta, 'min_child_weight':min_child_weight, 	# eta = step_size, default 0.3
             'colsample_bytree':colsample_bytree, 'gamma':gamma,  		# gamma = min_loss_reduction, default 0
             'num_class': 9,
             'objective':'multi:softprob', "eval_metric": "mlogloss"}
    num_round = 1000000
    evallist  = [(X_val_xgb,'eval'), (X_train_xgb,'train')]

    # single run:
    if SINGLE: bst, best_n_iter = train_single(Xy_all_xgb, param, num_round, evallist, save_model=True)

    # cv score on 1 classifier (by default runs in parallel in 3 cores)
    if CV_ONE: cvscore_single(X_all, y_all, param, num_round, str_opt)

    # grid, does NOT work with xgb
    if GRID: run_grid()

    # predicts on a fraction of the train set (on which is not trained), to cv the ensemble
    if CV4ENS: train_and_write_sub_4ensemble(X_all, y_all, ids_all, param, num_round,str_opt, seednum, encoder)

    # submission
    if SUB: write_sub_file(NEWF, LOG, SCA, PCA, INTE, GRID, ALLTRAINSET, LOADM, param, best_n_iter, str_opt, seednum,
                           scaler, encoder, bst=bst)


if __name__ == '__main__':
    main()
