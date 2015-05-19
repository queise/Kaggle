from __future__ import division
import sys
from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import sklearn.preprocessing as skl_pre
import sklearn.decomposition as skl_dec
import sklearn.ensemble as skl_ens
import sklearn.linear_model as skl_lm
import sklearn.metrics as skl_met
import sklearn.cross_validation as skl_cv
import sklearn.grid_search as skl_grid
import theano
import lasagne.layers as las_lay
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from lasagne.objectives import Objective
from nolearn.lasagne import NeuralNet
# TODO:

# tune:
# http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

np.random.seed(1313)

def main():
    print(" - Start.") ; t0 = time()
    trainpath = "../data/train.csv"
    testpath  = "../data/test.csv"
    subpath   = "sub.csv"

    X_all, y_all, ids_all = load_shuf_all_data_ids(trainpath)
#    X_all = X_all[:20000]
#    y_all = y_all[:20000]
    encoder = skl_pre.LabelEncoder()
    y_all = encoder.fit_transform(y_all).astype(np.int32)
    print 'X_all:',X_all
    print 'y_all:',y_all

    # new features
#    f1=True ; f2=True ; f3=True ; f4=True ; str_opt = '_f1234'
    f1=False ; f2=False ; f3=False ; f4=False ; str_opt = '_'
    X_all = F_addnewfeats(X_all, f1,f2,f3,f4)

    # to log scale
    LOG = True
    if LOG:
        print '\n log(1+x) to all features...'
        X_all = np.log1p(X_all)
        str_opt = str_opt + 'l'

    # scaling
    SCA = True
    if SCA:
        print '\n Scaling features...'
        scaler = skl_pre.StandardScaler().fit(X_all)
        X_all = scaler.transform(X_all)
        str_opt = str_opt + 'st'

    # PCA
    PCA = False
    if PCA:
        print '\n PCA...'
        pca = skl_dec.PCA(n_components='mle').fit(X_all)
        print('   ... num components: %i , variance retained: %.2f' % (len(pca.components_),sum(pca.explained_variance_ratio_)))
        X_all = pca.transform(X_all)
        print '\n X_all[0]:',X_all[0]
        str_opt = str_opt + ('p%.2f' % sum(pca.explained_variance_ratio_))


    # Prepare neural network:
    num_classes = len(encoder.classes_)
    num_features = X_all.shape[1]

    layers0 = [('input', las_lay.InputLayer),
               ('dropout0', las_lay.DropoutLayer),
               ('hidden1', las_lay.DenseLayer),
               ('dropout1', las_lay.DropoutLayer),
               ('hidden2', las_lay.DenseLayer),
               ('dropout2', las_lay.DropoutLayer),
#               ('hidden3', las_lay.DenseLayer),
               ('output', las_lay.DenseLayer)]

    NNargs = dict(layers=layers0,
                  input_shape=(None, num_features),
                  dropout0_p=0.15,
                  hidden1_num_units=1000,
                  dropout1_p=0.25,
                  hidden2_num_units=500,
                  dropout2_p=0.25,
#                  hidden3_num_units=150,
#                  hidden*_nonlinearity=rectifier by default, try softmax
                  output_num_units=num_classes,
                  output_nonlinearity=softmax,
                  #
                  update= adagrad, # nesterov_momentum, # adagrad, rmsprop
                  update_learning_rate=theano.shared(float32(0.03)), # 0.01,
                  # update_momentum=theano.shared(float32(0.9)), # 0.9, ONLY USE WITH nesterov_momentum
                  on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
#                                     AdjustVariable('update_momentum', start=0.9, stop=0.999),  # ONLY USE WITH nesterov_momentum
                                     EarlyStopping(patience=10),],
		  #
                  eval_size=0.2, # this fraction of the set is used for validation (test)
                  verbose=1,
                  max_epochs=150) #,
#		  objective = MyObjective)
    global GLOBrealnumepochs
    GLOBrealnumepochs = NNargs["max_epochs"]
    clf = NeuralNet(**NNargs)

    multiclass_log_loss = skl_met.make_scorer(score_func=logloss_mc, greater_is_better=False, needs_proba=True)
    num_cores = multiprocessing.cpu_count()

    # single run:
    SINGLE = False
    if SINGLE:
        clf.fit(X_all, y_all)
        train_loss = np.array([i["train_loss"] for i in clf.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in clf.train_history_])
        pyplot.plot(train_loss, linewidth=3, label="train")
        pyplot.plot(valid_loss, linewidth=3, label="valid")
        pyplot.grid()
        pyplot.legend()
        pyplot.xlabel("epoch")
        pyplot.ylabel("loss")
        pyplot.ylim(0.4, 0.8)
        pyplot.yscale("log")
#        pyplot.show()
        pyplot.savefig('learningcurves.png')

    # cv score on 1 classifier
    CV = False
    if CV:
        scores = skl_cv.cross_val_score(clf, X_all, y_all, cv=3, scoring=multiclass_log_loss, n_jobs=3)
        print ('\n\n ... done in %.0fs\n\n' % (time() - t0))
        print '\n cv scores of 1 classifier:', scores
        print '\n mean score=',np.mean(scores)
        sys.stdout.flush()

    # grid
    GRID = False
    if GRID:
        print '\n\n ** GRID SEARCH ** \n\n'
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


    CV4ENS = True
    if CV4ENS:
        # writing the predictions of 5 CV in files for future testing of ensemble
        n_folds=5
        kf = skl_cv.KFold(len(X_all), n_folds=n_folds, random_state=13)
        scores = [] ; ids_total = np.array([]) ; y_prob_total = np.empty((1,num_classes))
        for train, test in kf:
            X_train, X_test, y_train, y_test = X_all[train], X_all[test], y_all[train], y_all[test]
            ids_total = np.append(ids_total, ids_all[test])
            BAG = True
            if BAG:
                print("\n\n Fit and predict for different realizations of same architecture (diff seeds)")
                num_bags = 5
                best_max_epochs = {}
                for i in xrange(num_bags): best_max_epochs[i]=0 
#                best_max_epochs = { 0: 1, 1:1, 2:1, 3:1, 4:1} #68, 1: 57, 2: 46} # zeros for bags you already dont know best n_epochs
                assert len(best_max_epochs)==num_bags
                probs_bags = Parallel(n_jobs=num_cores)(delayed(calc_prob_bag)(i,best_max_epochs[i],NNargs,
                                                        X_all,y_all,X_test) for i in xrange(num_bags))
                y_prob = sum(probs_bags) / num_bags
                print 'y_prob:',np.shape(y_prob),y_prob
                y_prob_total = np.concatenate((y_prob_total, y_prob), axis=0)
            else:
                clf = NeuralNet(**NNargs)
                clf.fit(X_train,y_train)
                y_prob = clf.predict_proba(X_test)
                y_prob_total = np.concatenate((y_prob_total, y_prob), axis=0)
        y_prob_total = np.delete(y_prob_total, 0, 0) # removes first row (created by np.empty)
#        subfile = 'LNN4' + str_opt + '_e%i_h%i_d%.1f_h%i_CVALL.csv' % (NNargs["max_epochs"],NNargs["hidden1_num_units"],
#                                                                       NNargs["dropout1_p"], NNargs["hidden2_num_units"])            
        subfile = 'LNN4sbag%i%s_d%.2f_h%i_d%.2f_h%i_d%.2f_CVALL.csv' % (num_bags, str_opt, NNargs["dropout0_p"],
                                                                   NNargs["hidden1_num_units"], NNargs["dropout1_p"],
                                                                   NNargs["hidden2_num_units"], NNargs["dropout2_p"])
#        subfile = 'LNN5' + str_opt + '_e%i_h%i_d%.1f_h%i_d%.1f_h%i_CVALL.csv' % (NNargs["max_epochs"], NNargs["hidden1_num_units"],
 #                                                                NNargs["dropout1_p"], NNargs["hidden2_num_units"],
  #                                                               NNargs["dropout2_p"], NNargs["hidden3_num_units"])

        write_submission(ids_total, encoder, y_prob_total, path=subfile)


    SUB = False
    if SUB:
        print("\n\n  Starting submission process...")
#        encoder = skl_pre.LabelEncoder()
#        y_true = encoder.fit_transform(y_all)
#        assert (encoder.classes_ == clf_final.classes_).all()
        X_test, ids = load_test_data(path=testpath)
        X_test = F_addnewfeats(X_test, f1,f2,f3,f4)
        if LOG: X_test = np.log1p(X_test)
        if SCA: X_test = scaler.transform(X_test)
        if PCA: X_test = pca.transform(X_test)

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

        BAG = True
        if BAG:
            # see https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13851/lasagne-with-2-hidden-layers
            print("\n\n Fit and predict for different realiztions of same architecture (diff seeds)")
            print("\n\n Fit and predict for different realizations of same architecture (diff seeds)")
            num_bags = 5
            best_max_epochs = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0} # { 0: 68, 1: 57, 2: 46} # zeros for bags you already dont know best n_epochs
            assert len(best_max_epochs)==num_bags
            probs_bags = Parallel(n_jobs=num_cores)(delayed(calc_prob_bag)(i,best_max_epochs[i],NNargs,
                                                    X_all,y_all,X_test) for i in xrange(num_bags))
            probs_final = sum(probs_bags) / num_bags

            subfile = 'LNN4bag%i%s_d%.2f_h%i_d%.2f_h%i_d%.2f.csv' % (num_bags, str_opt, NNargs["dropout0_p"],
                                                                     NNargs["hidden1_num_units"], NNargs["dropout1_p"],
                                                                     NNargs["hidden2_num_units"], NNargs["dropout2_p"])
            print("\n writing submission to file: %s" % subfile)
            write_submission(ids, encoder, probs_final, path=subfile)
            sys.exit()
          
        # re-set properties to train with all set
        NNargs["eval_size"] = 0.0001
        if len(NNargs["on_epoch_finished"])==3: del NNargs["on_epoch_finished"][-1] # removes the EarlyStopping
        NNargs["max_epochs"] = GLOBrealnumepochs

        clf_final = NeuralNet(**NNargs)

        print("\n  Writing submission file for best estimator")
#        subfile = 'LNN4' + str_opt + '_e%i_h%i_d%.1f_h%i.csv' % (NNargs["max_epochs"], NNargs["hidden1_num_units"],
 #                                                                NNargs["dropout1_p"], NNargs["hidden2_num_units"])
        subfile = 'LNN5' + str_opt + '_e%i_h%i_d%.1f_h%i_d%.1f_h%i.csv' % (NNargs["max_epochs"], NNargs["hidden1_num_units"],
                                                                 NNargs["dropout1_p"], NNargs["hidden2_num_units"],
							         NNargs["dropout2_p"], NNargs["hidden3_num_units"])
        print '\n name:',subfile,'\n'

        print("   re-fitting with all training set...")
        clf_final.fit(X_all, y_all)

        make_submission(X_test, ids, clf_final, encoder, path=subfile)


def F_addnewfeats(X, f1,f2,f3,f4):
    Xoriginal = X
    Xmeans = np.mean(X, axis=0)
    Xstds = np.std(X, axis=0)
    if f1:
        numnonzerosineachrow = (Xoriginal != 0).sum(1)
        X = np.c_[X, numnonzerosineachrow]
        print ' ... new feature numnonzerosineachrow added'
    if f2:
        numover1stdineachrow = (Xoriginal > (Xmeans+Xstds)).sum(1)
        X = np.c_[X, numover1stdineachrow]
        print ' ... new feature numover1stdineachrow added'
    if f3:
        numover2stdineachrow = (Xoriginal > (Xmeans+2*Xstds)).sum(1)
        X = np.c_[X, numover2stdineachrow]
        print ' ... new feature numover2stdineachrow added'
    if f4:
        numover3stdineachrow = (Xoriginal > (Xmeans+3*Xstds)).sum(1)
        X = np.c_[X, numover3stdineachrow]
        print ' ... new feature numover3stdineachrow added'
    return X


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


def load_shuf_all_data_ids(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    print 'example of row:',X[0]
    ids = X[:, 0].astype(str)
    y = X[:,-1].astype(str)
    X = X[:,1:-1].astype(float)
    print(" -- Loaded data.")
    return X, y, ids


def load_shuf_all_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    print 'example of row:',X[0]
    y = X[:,-1].astype(str)
    X = X[:,1:-1].astype(float)
    print(" -- Loaded data.")
    return X, y


def load_test_data(path=None):
    path = sys.argv[2] if len(sys.argv) > 2 else path
    df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def make_submission(X_test, ids, clf, encoder, path='my_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    write_submission(ids, encoder, y_prob, path)


def write_submission(ids, encoder, y_prob, path):
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("\n ... wrote submission to file {}".format(path))


class MyObjective(Objective):
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
	loss = super(MyObjective, self).get_loss(input=input, target=target,
                                                 deterministic=deterministic, **kwargs)
 	if not deterministic:
	    return loss + 0.01 * lasagne.regularization.l2(self.input_layer)
	else:
	    return loss

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


if __name__ == '__main__':
    main()
