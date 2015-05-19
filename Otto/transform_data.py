import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.decomposition as skl_dec


def transform_data(X_all, NEWF = False, LOG = True, SCA = True, PCA = False, INTE = False):

    # string that will contain info on data transformations:
    str_opt = '_'

    # new features
    if NEWF:
        X_all = F_addnewfeats(X_all)
        str_opt += 'Nf'

    # to log scale
#    LOG = True
    if LOG:
        print '\n log(1+x) to all features...'
        X_all = np.log1p(X_all)
        str_opt = str_opt + 'l'

    # scaling
#    SCA = True
    if SCA:
        print '\n Scaling features...'
        scaler = skl_pre.StandardScaler().fit(X_all)
        X_all = scaler.transform(X_all)
        str_opt = str_opt + 'st'

    # interactions among features
#    INTE = False
    if INTE:
        X_all = np.float16(X_all)
        #  first reduce to only those more important
        print '    Creating temp set with reduced number of features to calculate interactions...'
        pcatemp = skl_dec.PCA(n_components=0.6).fit(X_all)
        print('    ... num components: %i , variance retained: %.2f' % (len(pcatemp.components_),sum(pcatemp.explained_variance_ratio_)))
        X_all_reduced = pcatemp.transform(X_all)
        print '    Creating interactions among features...'
        poly = skl_pre.PolynomialFeatures(2, interaction_only=True).fit(X_all_reduced)
        X_all_inter = poly.transform(X_all_reduced)
        numpcacomp = len(pcatemp.components_)
        numinterfeats = int((numpcacomp-1)*numpcacomp/2) # equiv to (numpcacomp-1) + (numpcacomp-2) + (numpcacomp-3) + ... + 1
        print '    ... num new features created:',numinterfeats
        X_all_inter =X_all_inter[:,-numinterfeats:] # only the columns of interactions are selected
        X_all = np.concatenate((X_all, X_all_inter), axis=1) # adds interaction of pca to the original matrix
        scaler = skl_pre.StandardScaler().fit(X_all)
        X_all = scaler.transform(X_all)
        str_opt = str_opt + 'int'

    # PCA
#    PCA = False
    if PCA:
        print '\n PCA...'
        pca = skl_dec.PCA(n_components='mle').fit(X_all)
        print('   ... num components: %i , variance retained: %.2f' % (len(pca.components_),sum(pca.explained_variance_ratio_)))
        X_all = pca.transform(X_all)
        print '\n X_all[0]:',X_all[0]
        str_opt = str_opt + ('p%.2f' % sum(pca.explained_variance_ratio_))

    return X_all, str_opt, scaler


def F_addnewfeats(X, f1=True, f2=True, f3=True, f4=True):
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
