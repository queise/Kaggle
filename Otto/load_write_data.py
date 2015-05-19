import sys
import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_pre


def load_train_data(trainpath = "../data/train.csv", subsample=False, subsample_size=0.1, verbose=False):
    print("\n   Loading train set from %s ..." % trainpath)
    X_all, y_all, ids_all = load_shuf_all_data_ids(trainpath)
    if subsample:
        X_all = X_all[:int(subsample_size*len(X_all))]
        y_all = y_all[:int(subsample_size*len(y_all))]
    encoder = skl_pre.LabelEncoder()
    y_all = encoder.fit_transform(y_all).astype(np.int32)
    num_classes = len(encoder.classes_)
    if verbose:
        print 'X_all:',X_all
        print 'y_all:',y_all

    return X_all, y_all, ids_all, encoder

def load_test_data(path  = "../data/test.csv"):
    path = sys.argv[2] if len(sys.argv) > 2 else path
    df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def load_shuf_all_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    print 'example of row:',X[0]
    y = X[:,-1].astype(str)
    X = X[:,1:-1].astype(float)
    print(" -- Loaded data.")
    return X, y


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

