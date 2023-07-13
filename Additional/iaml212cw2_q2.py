
##########################################################
#  Python script template for Question 2 (IAML Level 11)
#  Note that
#  - You should not change the filename of this file, 'iaml212cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from iaml_cw2_helpers import *
# from iaml212cw2_my_helpers import *


# Q2.1
def iaml212cw2_q2_1():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean
    # X = []
    # for x in Xtrn:
    #     for xx in x:
    #         X.append(xx)
    # print(np.max(X))
    # print(np.min(X))
    # print(np.mean(X))
    # print(np.std(X))
    # X = list()
    # for x in Xtst:
    #     for xx in x:
    #         X.append(xx)
    # print(np.max(X))
    # print(np.min(X))
    # print(np.mean(X))
    # print(np.std(X))
    image = []
    for i in range(0, 28):
        line = []
        for j in range(0, 28):
            line.append(255 - Xtst_org[0][i * 28 + j])
        image.append(line)
    image = np.array(image).T
    print(Ytrn[0])
    plt.imshow(image, cmap="gray")
    plt.show()
    image = []
    for i in range(0, 28):
        line = []
        for j in range(0, 28):
            line.append(255 - Xtrn_org[1][i * 28 + j])
        image.append(line)
    image = np.array(image).T
    print(Ytrn[1])
    plt.imshow(image, cmap="gray")
    plt.show()

    return

# iaml212cw2_q2_1()   # comment this out when you run the function

# Q2.3
def iaml212cw2_q2_3():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean
    A = []
    F = []
    I = []
    for i in range(0, len(Xtrn)):
        if Ytrn[i] == 0:
            A.append(Xtrn[i])
        elif Ytrn[i] == 5:
            F.append(Xtrn[i])
        elif Ytrn[i] == 8:
            I.append(Xtrn[i])

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=0).fit(A)
    A3c = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=3, random_state=0).fit(F)
    F3c = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=3, random_state=0).fit(I)
    I3c = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=5, random_state=0).fit(A)
    A5c = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=5, random_state=0).fit(F)
    F5c = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=5, random_state=0).fit(I)
    I5c = kmeans.cluster_centers_

    k = 3
    f, ax = plt.subplots(3, k)
    for ii in range(0, 3):
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 -A3c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[0][ii].imshow(image, cmap="gray")
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 - F3c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[1][ii].imshow(image, cmap="gray")
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 - I3c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[2][ii].imshow(image, cmap="gray")
    plt.show()

    k = 5
    f, ax = plt.subplots(3, k)
    for ii in range(0, 5):
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 - A5c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[0][ii].imshow(image, cmap="gray")
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 - F5c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[1][ii].imshow(image, cmap="gray")
        image = []
        for i in range(0, 28):
            line = []
            for j in range(0, 28):
                line.append(255 - I5c[ii][i * 28 + j])
            image.append(line)
        image = np.array(image).T
        ax[2][ii].imshow(image, cmap="gray")
    plt.show()

    return


# iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.5
def iaml212cw2_q2_5():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean

    # Xtrn_m = np.vstack((Xtrn_m, Xtrn_m))
    # Ytrn = np.hstack((Ytrn, Ytrn))

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrn_m, Ytrn)

    pre = clf.predict(Xtrn_m)
    hit = [0 for i in range(0, 26)]
    total = [0 for i in range(0, 26)]
    for i in range(0, len(Xtrn_m)):
        total[Ytrn[i]] += 1
        if pre[i] == Ytrn[i]:
            hit[Ytrn[i]] += 1
    acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
    # print(hit)
    # print(total)
    # print(acc)
    print(np.mean(acc))

    pre = clf.predict(Xtst_m)
    hit = [0 for i in range(0, 26)]
    total = [0 for i in range(0, 26)]
    for i in range(0, len(Xtst_m)):
        total[Ytst[i]] += 1
        if pre[i] == Ytst[i]:
            hit[Ytst[i]] += 1
    acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
    print(hit)
    print(total)
    print(acc)
    print(np.mean(acc))

    # miss = [total[i] - hit[i] for i in range(0, 26)]
    # for i in range(0, 5):
    #     m = max(miss)
    #     index = miss.index(m)
    #     print(index, end=' ')
    #     print(m)
    #     miss[index] = 0

    return


# iaml212cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml212cw2_q2_6():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean

    # Xtrn_m = np.vstack((Xtrn_m, Xtrn_m))
    # Ytrn = np.hstack((Ytrn, Ytrn))

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(Xtrn_m, Ytrn)

    pre = clf.predict(Xtrn_m)
    hit = [0 for i in range(0, 26)]
    total = [0 for i in range(0, 26)]
    for i in range(0, len(Xtrn_m)):
        total[Ytrn[i]] += 1
        if pre[i] == Ytrn[i]:
            hit[Ytrn[i]] += 1
    acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
    # print(hit)
    # print(total)
    # print(acc)
    print(np.mean(acc))

    pre = clf.predict(Xtst_m)
    hit = [0 for i in range(0, 26)]
    total = [0 for i in range(0, 26)]
    for i in range(0, len(Xtst_m)):
        total[Ytst[i]] += 1
        if pre[i] == Ytst[i]:
            hit[Ytst[i]] += 1
    acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
    print(hit)
    print(total)
    print(acc)
    print(np.mean(acc))

    # miss = [total[i] - hit[i] for i in range(0, 26)]
    # for i in range(0, 5):
    #     m = max(miss)
    #     index = miss.index(m)
    #     print(index, end=' ')
    #     print(m)
    #     miss[index] = 0

    print()
    for c in range(0, 21):
        C = 0.01 + 0.01 * c
        clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(Xtrn_m, Ytrn)

        pre = clf.predict(Xtrn_m)
        hit = [0 for i in range(0, 26)]
        total = [0 for i in range(0, 26)]
        for i in range(0, len(Xtrn_m)):
            total[Ytrn[i]] += 1
            if pre[i] == Ytrn[i]:
                hit[Ytrn[i]] += 1
        acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
        # print(hit)
        # print(total)
        # print(acc)
        # print(np.mean(acc))

        pre = clf.predict(Xtst_m)
        hit = [0 for i in range(0, 26)]
        total = [0 for i in range(0, 26)]
        for i in range(0, len(Xtst_m)):
            total[Ytst[i]] += 1
            if pre[i] == Ytst[i]:
                hit[Ytst[i]] += 1
        acc = [float(hit[i]) / float(total[i]) for i in range(0, len(total))]
        # print(hit)
        # print(total)
        # print(acc)
        print(np.mean(acc), end=', ')

    return

# iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7 
def iaml212cw2_q2_7():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean
    X = []
    for i in range(0, len(Xtrn_m)):
        if Ytrn[i] == 0:
            X.append(Xtrn_m[i])
    X = np.array(X)
    mean_vec = []
    for i in range(0, X.shape[1]):
        mean_vec.append(np.mean(X[:, i]))
    cov = np.cov(X.T)
    # print(np.min(cov))
    # print(np.max(cov))
    # print(np.mean(cov))
    diag = [cov[i][i] for i in range(0, len(cov))]
    # print(np.min(diag))
    # print(np.max(diag))
    # print(np.mean(diag))
    # plt.hist(diag, bins=15)
    # plt.grid()
    # plt.show()

    from scipy.stats import multivariate_normal
    p = multivariate_normal.cdf(Xtst_m[0], mean_vec, cov)
    print(p)
    return


# iaml212cw2_q2_7()   # comment this out when you run the function

# Q2.8 
def iaml212cw2_q2_8():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean
    from sklearn.mixture import GaussianMixture
    X = [[] for i in range(0, 26)]
    for i in range(0, len(Xtrn_m)):
        X[Ytrn[i]].append(Xtrn_m[i])
    gm = [GaussianMixture(n_components=1, random_state=0).fit(X[i]) for i in range(0, len(X))]

    X = [[] for i in range(0, 26)]
    for i in range(0, len(Xtst_m)):
        X[Ytst[i]].append(Xtst_m[i])

    train_num = [0 for i in range(0, 26)]
    train_total = [0 for i in range(0, 26)]
    for i in range(0, len(X)):
        result = []
        XX = X[i]
        train_total[i] = len(X[i])
        for j in range(0, 26):
            r = gm[j].score_samples(XX)
            result.append(r)
        result = np.array(result).T
        for ii in range(0, len(result)):
            x = result[ii].tolist()
            r = x.index(max(x))
            if r == i:
                train_num[i] += 1

    print(train_num)
    print([float(train_num[i]) / float(train_total[i]) for i in range(0, len(train_num))])

    return


    # def find(XX, gm, ii):
    #     current_index = 0
    #     current = gm[ii].score_samples(XX)[0]
    #     for i in range(1, 26):
    #         cur = gm[i].score_samples(XX)[0]
    #         if cur > current:
    #             return -1
    #     return ii
    #
    # train_num = [0 for i in range(0, 26)]
    # train_total = [0 for i in range(0, 26)]
    # for i in range(0, len(X)):
    #     train_total[i] = len(X[i])
    #     for j in range(0, len(X[i])):
    #         XX = [X[i][j]]
    #         index = find(XX, gm, ii=i)
    #         if index == i:
    #             train_num[i] += 1
    #
    # print(train_num)
    # print(train_total)

    # gm = GaussianMixture(n_components=26, random_state=0).fit(Xtrn_m)
    # score = gm.predict(Xtrn_m)
    # train_num = [0 for i in range(0, 26)]
    # train_total = [0 for i in range(0, 26)]
    # for i in range(0, len(Xtrn_m)):
    #     train_total[Ytrn[i]] += 1
    #     if score[i] == Ytrn[i]:
    #         train_num[score[i]] += 1
    #
    # print(train_num)
    # print(train_total)


    # return

# iaml212cw2_q2_8()   # comment this out when you run the function

# Q2.10 
def iaml212cw2_q2_10():
    Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
    Xtrn = Xtrn_org / 255.0
    Xtst = Xtst_org / 255.0
    Ytrn = Ytrn_org - 1
    Ytst = Ytst_org - 1
    Xmean = np.mean(Xtrn, axis=0)
    Xtrn_m = Xtrn - Xmean
    Xtst_m = Xtst - Xmean
    from sklearn.mixture import GaussianMixture
    X = [[] for i in range(0, 26)]
    for i in range(0, len(Xtrn_m)):
        X[Ytrn[i]].append(Xtrn_m[i])
    K = 2
    L = [1, 2, 4, 8]
    best = -1
    val = -1
    for ij in range(0, len(L)):
        X = [[] for i in range(0, 26)]
        for i in range(0, len(Xtrn_m)):
            X[Ytrn[i]].append(Xtrn_m[i])
        gm = [GaussianMixture(n_components=K, random_state=0, reg_covar=L[ij]).fit(X[i]) for i in range(0, len(X))]

        # X = [[] for i in range(0, 26)]
        # for i in range(0, len(Xtrn_m)):
        #     X[Ytrn[i]].append(Xtrn_m[i])
        #
        # train_num = [0 for i in range(0, 26)]
        # train_total = [0 for i in range(0, 26)]
        # for i in range(0, len(X)):
        #     result = []
        #     XX = X[i]
        #     train_total[i] = len(X[i])
        #     for j in range(0, 26):
        #         r = gm[j].score_samples(XX)
        #         result.append(r)
        #     result = np.array(result).T
        #     for ii in range(0, len(result)):
        #         x = result[ii].tolist()
        #         r = x.index(max(x))
        #         if r == i:
        #             train_num[i] += 1
        #
        # print(train_num)
        # print([float(train_num[i]) / float(train_total[i]) for i in range(0, len(train_num))])
        #
        # print(sum(train_num) / sum(train_total))

        X = [[] for i in range(0, 26)]
        for i in range(0, len(Xtst_m)):
            X[Ytst[i]].append(Xtst_m[i])

        train_num = [0 for i in range(0, 26)]
        train_total = [0 for i in range(0, 26)]
        for i in range(0, len(X)):
            result = []
            XX = X[i]
            train_total[i] = len(X[i])
            for j in range(0, 26):
                r = gm[j].score_samples(XX)
                result.append(r)
            result = np.array(result).T
            for ii in range(0, len(result)):
                x = result[ii].tolist()
                r = x.index(max(x))
                if r == i:
                    train_num[i] += 1

        print(train_num)
        print([float(train_num[i]) / float(train_total[i]) for i in range(0, len(train_num))])

        cur = sum(train_num) / sum(train_total)
        if best == 0 and val == 0:
            best = L[ij]
            val = cur
        elif cur > val:
            best = L[ij]
            val = cur
        print(L[ij])
        print(cur)
    print()
    print(best)
    print(val)

    return

# iaml212cw2_q2_10()   # comment this out when you run the function
