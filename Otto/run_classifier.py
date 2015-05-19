import sys
import xgboost as xgb
import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.cross_validation as skl_cv
import sklearn.metrics as skl_met
import sklearn.grid_search as skl_grid
from joblib import Parallel, delayed
import multiprocessing
from functions import *
from load_write_data import write_submission, load_test_data


def train_single(Xy_all_xgb, param, num_round, evallist, save_model=False):
    print("\n\n Training with all train set, with small eval set for early stopping...")
    bst = xgb.train(param, Xy_all_xgb, num_round, evallist, early_stopping_rounds=25)
    if hasattr(bst, 'best_iteration'):
        best_n_iter = bst.best_iteration
        print(" ... trained, optimal num of iterations = %i, best score = %.5f" % (best_n_iter, bst.best_score))
    else:
        best_n_iter = num_round
        print(" ... trained, until max rounds = %i" % best_n_iter)
    sys.stdout.flush()
    # saving model:
    if save_model:
        model_name = 'xbg-ss%.4f_d%i_c%i_s%.1f_i%i' % (param["eta"], param["max_depth"], param["min_child_weight"],
                                                       param["colsample_bytree"], best_n_iter)
        bst.save_model(model_name+'.model')
#           bst.dump_model(model_name+'.nice.txt',
#                         model_name+'.featmap.txt')
    return bst, best_n_iter

    """ # adaptative eta
    A_ETA = False
    # DONT WORK BECAUSE WHEN TRAIN AFTER LOADING DOES NOT REMEMBER ANYTHING
    if A_ETA:
        Etas = [0.030, 0.020, 0.010, 0.005, 0.001]
        for i,eta in enumerate(Etas):
            if i!=0:
                bst = xgb.Booster(model_file = 'xbg-a_eta.model')
            print("\n training with eta=%.3f" % eta)
            param["eta"]=eta
            bst = xgb.train(param, Xy_all_xgb, num_round, evallist, early_stopping_rounds=10)
            bst.save_model('xbg-a_eta.model') """

def cvscore_single(X_all, y_all, param, num_round, str_opt):
    n_folds = 3 ; scores = [] ; train_ind = [] ; test_ind = []
    kf = skl_cv.KFold(len(X_all), n_folds=n_folds, random_state=13)
    num_cores = n_folds
    for train, test in kf:
        train_ind.append(train)
        test_ind.append(test)
    scores = Parallel(n_jobs=num_cores)(delayed(calc_cv_score)(i,train_ind[i],test_ind[i],n_folds,X_all,y_all,
                                                               param,num_round) for i in xrange(n_folds))
    print '\n cv scores of the booster:', scores
    mean_cv_score = sum(scores)/len(scores)
    print '\n mean score = ',mean_cv_score
    sys.stdout.flush()
    with open('0results-xgb-grid', 'a') as f:
        f.write("%.4f\t%i\t%i\t%.2f\t%.2f\t%.5f\t\t%s\n" % (param["eta"], param["max_depth"], param["min_child_weight"],
                                                            param["colsample_bytree"], param["gamma"], mean_cv_score, str_opt))

def run_grid():
    print '\n\n ** GRID SEARCH ** \n\n'
    multiclass_log_loss = skl_met.make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
    grid_clf = skl_grid.GridSearchCV(clf,
                                 param_grid={'hidden1_num_units': [3000, 4000], #[100, 150, 200, 250, 300, 400, 600],
                                             'dropout1_p': [0.6, 0.8], #, 0.5, 0.7],
                                             'hidden2_num_units': [300, 500]},
                                 scoring=multiclass_log_loss, cv=3, n_jobs=3)
    print(' ... grid created')

    print('\n Fitting grid...') ; tfit = time() ; sys.stdout.flush()
    grid_clf.fit(X_all, y_all)
    print(' ... grid fitted in %.2fs' % (time()-tfit))
    print("\n Grid scores:")
    for params, mean_score, scores in grid_clf.grid_scores_:
        print("%0.5f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
    print("\n Best estimator:")
    print(grid_clf.best_estimator_)
    print("\n Best params:")
    print(grid_clf.best_params_)
    print("\n Best score:")
    print(grid_clf.best_score_)


def train_and_write_sub_4ensemble(X_all, y_all, ids_all, param, num_round, str_opt, seednum, encoder):
        X_train = X_all[:int(0.85*len(X_all))] ; y_train = y_all[:int(0.85*len(y_all))]
        X_cv4ens = X_all[int(0.85*len(X_all)):] ; y_cv4ens = y_all[int(0.85*len(y_all)):]
        ids_cv4ens = ids_all[int(0.85*len(X_all)):]
        Xy_cvtrain_xgb, X_train_xgb, X_val_xgb = prepare_xgb_matrixes(X_train, y_train, eval_size = 0.05)
        X_cvtest_xgb = xgb.DMatrix(X_cv4ens)
        print("\n  Training and obtaining optimal num_rounds...")
        evallist  = [(X_val_xgb,'eval'), (X_train_xgb,'train')]
        bst = xgb.train(param, Xy_cvtrain_xgb, num_round, evallist, early_stopping_rounds=15)
        if hasattr(bst, 'best_iteration'):
            best_n_iter = bst.best_iteration
        else:
            best_n_iter = num_round
        print(" ... trained, optimal num of iterations = %i" % best_n_iter)
        print("\n    now calc probs on cv4ens set")
        y_prob = bst.predict(X_cvtest_xgb, ntree_limit=best_n_iter) # ntree_limit is not necessary if bst was fitted with best_n_iter
        print("\n  ... probs done")
        subfile = 'xbg-S%i-%s_ss%.4f_d%i_c%i_s%.1f_i%i_CV4ENS.csv' % \
                  (seednum, str_opt, param["eta"], param["max_depth"], param["min_child_weight"],
                   param["colsample_bytree"], best_n_iter)
        write_submission(ids_cv4ens, encoder, y_prob, path=subfile)


def write_sub_file(NEWF, LOG, SCA, PCA, INTE, GRID, ALLTRAINSET, LOADM, param, best_n_iter, str_opt, seednum,
                   scaler, encoder, bst=None):
    print("\n\n  Starting submission process...")
    X_test, ids = load_test_data()
    if NEWF: X_test = F_addnewfeats(X_test)
    if LOG: X_test = np.log1p(X_test)
    if SCA: X_test = scaler.transform(X_test)
    if INTE:
        X_test_reduced = pcatemp.transform(X_test)
        X_test_inter = poly.transform(X_test_reduced)
        X_test_inter = X_test_inter[:,-numinterfeats:]
        X_test = np.concatenate((X_test, X_test_inter), axis=1)
    if PCA: X_test = pca.transform(X_test)
    X_test_xgb = xgb.DMatrix(X_test)

    if GRID:
        print('\n      Setting NN params to best values in the grid...')
        NNargs["hidden1_num_units"] = grid_clf.best_params_['hidden1_num_units']
        NNargs["dropout1_p"] = grid_clf.best_params_['dropout1_p']
        NNargs["hidden2_num_units"] = grid_clf.best_params_['hidden2_num_units']

    RECALCEPOC = False
    if RECALCEPOC:
        print '\n      Refitting to get the num epochs optimal (with smaller eval_size and more patience)...'
        NNargs["eval_size"] = 0.05 # just a small test set to derive optimal numepochs, closer to final training with all set
        if len(NNargs["on_epoch_finished"])==3: NNargs["on_epoch_finished"][-1] = EarlyStopping(patience=25) # more patience for final sub
        clf_final = NeuralNet(**NNargs)
        clf_final.fit(X_all, y_all)
        print '        ... done refitting to obtain GLOBrealnumepochs:',GLOBrealnumepochs

#        print("\n  Writing submission WITHOUT retraining with all train set")
#       subfile = 'xbg-NORETR-ss%.3f_i%i.csv' % (param['eta'], best_n_iter)
#      y_prob = bst.predict(X_test_xgb, ntree_limit=best_n_iter)
#     write_submission(ids, encoder, y_prob, path=subfile)

#        ALLTRAINSET = False
    if ALLTRAINSET:
        print("\n  re-fitting with all training set and optimal num of iterations...")
        sys.stdout.flush()
        bst = xgb.train(param, Xy_all_xgb, best_n_iter)
        print("\n  ... re-fitted with all training set and optimal num of iterations")
        # saving model:
        model_name = 'xbg-ss%.4f_d%i_c%i_s%.1f_i%i' % (param["eta"], param["max_depth"], param["min_child_weight"],
                                                       param["colsample_bytree"], best_n_iter)
        bst.save_model(model_name+'.model')
#         bst.dump_model('xbg-ss%.3f_i%i.nice.txt' % (param['eta'], best_n_iter),
#                    'xbg-ss%.3f_i%i.featmap.txt'% (param['eta'], best_n_iter))
        print("\n  calculating predictions...")
        y_prob = bst.predict(X_test_xgb) # ntree_limit is not necessary if bst was fitted with best_n_iter
                                     # otherwise it should be used, because train returns the model in last iteration, not the best
        str_opt += 'yAt'
    else:
        if LOADM:
            print("\n Loading model...")
            model_name = 'xbg-ss%.4f_d%i_c%i_s%.1f_i%i' % (param["eta"], param["max_depth"], param["min_child_weight"],
                                                           param["colsample_bytree"], best_n_iter)
            bst = xgb.Booster(model_file = model_name+'.model')
        print("\n  Directly predict with previously trained bst using best_n_iter=%i" % best_n_iter)
        y_prob = bst.predict(X_test_xgb, ntree_limit=best_n_iter)
        str_opt += 'nAt'

    print("\n  Writing submission file for best estimator")
    subfile = 'xbg-S%i-%s_ss%.4f_d%i_c%i_s%.1f_i%i.csv' % (seednum, str_opt, param["eta"], param["max_depth"],
                                                           param["min_child_weight"], param["colsample_bytree"], best_n_iter)
    print '\n name:',subfile,'\n'
    write_submission(ids, encoder, y_prob, path=subfile)
#   make_submission(X_test, ids, clf_final, encoder, path=subfile)


def calc_cv_score(i, train_ind, test_ind, n_folds, X_all, y_all, param, num_round):
    print("\n ***  K-fold CV, fold %i/%i" % ((i+1),n_folds))
    X_cvtrain, X_cvtest, y_cvtrain, y_cvtest = X_all[train_ind], X_all[test_ind], y_all[train_ind], y_all[test_ind]
    Xy_cvall_xgb, X_cvtrain_xgb, X_cvval_xgb = prepare_xgb_matrixes(X_cvtrain, y_cvtrain, eval_size = 0.05)
    evallist  = [(X_cvval_xgb,'eval'), (X_cvtrain_xgb,'train')]
    # I just train once, with eval subset and not retrain to best_iter, because I am only interested on choosing the best model,
    # later I will retrain the best model with allthe train set and with best_iter
    print("\n    training with cv train set and eval subset ...")
    bst = xgb.train(param, Xy_cvall_xgb, num_round, evallist, early_stopping_rounds=15)
    if hasattr(bst, 'best_iteration'):
        best_n_iter = bst.best_iteration
    else:
        best_n_iter = num_round
    print("\n  ... trained, now calc probs on cv test set with optimal num of iterations = %i ..." % best_n_iter)
    X_cvtest_xgb = xgb.DMatrix(X_cvtest)
    y_prob = bst.predict(X_cvtest_xgb, ntree_limit=best_n_iter) # ntree_limit is not necessary if bst was fitted to best_n_iter
    print("\n  ... probs done, now scoring...")
    score_i = logloss_mc(y_cvtest, y_prob)
    print 'score_i:',score_i
    print("\n    score of fold %i/%i: %.5f " % ((i+1), n_folds, score_i))
    return score_i


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss

    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    encoder = skl_pre.LabelEncoder()
    y_true = encoder.fit_transform(y_true)

    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None


    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            global GLOBrealnumepochs
            GLOBrealnumepochs = self.best_valid_epoch
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


def calc_prob_bag(i, best_max_epochs, NNargs, X_all, y_all, X_test):
    np.random.seed(111*(i+1)) # diffs random seed to diffs bagging
    print('\n - Bag: %i ' % (i+1))
    if best_max_epochs == 0:
        print('    First fit to get optimal num of epochs...')
        NNargs["max_epochs"] = 1000
        NNargs["eval_size"] = 0.05 # just a small test set to derive optimal numepochs
        NNargs["on_epoch_finished"][-1] = EarlyStopping(patience=25) # more patience
        clf_bag = NeuralNet(**NNargs)
        clf_bag.fit(X_all, y_all)


        print("\n\n Training with all train set...")
        bst = xgb.train(param, Xy_all_xgb, num_round, evallist, early_stopping_rounds=25)
        if hasattr(bst, 'best_iteration'):
            best_n_iter = bst.best_iteration
        else:
            best_n_iter = num_round
        print(" ... trained, optimal num of iterations = %i" % best_n_iter)



        global GLOBrealnumepochs
        best_max_epochs = GLOBrealnumepochs
    print('        we will refit now with max epochs = %i' % best_max_epochs)
    NNargs["max_epochs"] = best_max_epochs
    NNargs["eval_size"] = 0.0001
    NNargs["on_epoch_finished"][-1] = EarlyStopping(patience=1000) # kind of a infinite patience to let max epochs rule
    clf_bag = NeuralNet(**NNargs)
    clf_bag.fit(X_all, y_all)
    probs_bags = clf_bag.predict_proba(X_test)
    return probs_bags
