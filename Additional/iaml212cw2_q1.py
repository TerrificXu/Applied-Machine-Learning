import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from iaml_cw2_helpers import *


# from iaml212cw2_my_helpers import *


# Q1.1
def iaml212cw2_q1_1():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        Xa = []
        Xb = []
        for j in range(0, len(Xtrn)):
            if Ytrn[j] == 0:
                Xa.append(Xtrn[j][i - 1])
            else:
                Xb.append(Xtrn[j][i - 1])
        plt.grid()
        plt.hist([Xa, Xb], bins=15)
    plt.show()
    return


# iaml212cw2_q1_1()  # comment this out when you run the function


# Q1.2
def iaml212cw2_q1_2():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    from scipy.stats import pearsonr
    correlation_list = []
    for i in range(0, 9):
        X = Xtrn[:, i]
        Y = Ytrn
        correlation, _ = pearsonr(X, Y)
        correlation_list.append(correlation)
    return


# iaml212cw2_q1_2()   # comment this out when you run the function

# Q1.4
def iaml212cw2_q1_4():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    var = []
    for i in range(0, 9):
        X = Xtrn[:, i]
        var.append(np.var(X, ddof=1))
    index = list()
    val = list()
    for i in range(0, 9):
        max_v = max(var)
        max_index = var.index(max_v)
        index.append(max_index)
        val.append(max_v)
        var[max_index] = -1
    plt.subplot(1, 2, 1)
    plt.grid()
    tick = index
    labels = [str(t) for t in tick]
    plt.xticks(ticks=[i for i in range(0, 9)], labels=labels)
    plt.scatter([i for i in range(0, 9)], val)
    plt.subplot(1, 2, 2)
    plt.grid()
    ratio = []
    for i in range(0, 9):
        current = 0
        for j in range(0, i + 1):
            current += val[j]
        ratio.append(current / sum(val))
    plt.xticks(ticks=[i for i in range(0, 9)], labels=labels)
    plt.ylim(0, 1)
    plt.scatter([i for i in range(0, 9)], ratio)
    plt.show()

    return


# iaml212cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml212cw2_q1_5():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(Xtrn)
    ratio = pca.explained_variance_ratio_
    var = pca.explained_variance_
    com = pca.components_


    plt.subplot(1, 2, 1)
    plt.grid()
    plt.scatter([i for i in range(0, 9)], var)
    plt.subplot(1, 2, 2)
    plt.grid()
    cumulative_ratio = []
    for i in range(0, 9):
        cumulative = 0
        for j in range(0, i + 1):
            cumulative += ratio[j]
        cumulative_ratio.append(cumulative)
    plt.ylim(0, 1)
    plt.scatter([i for i in range(0, 9)], cumulative_ratio)
    plt.show()

    vec0 = com[0]
    vec1 = com[1]
    X = []
    for x in Xtrn:
        x = np.array(x)
        X.append([x.dot(vec0), x.dot(vec1)])
    X0 = []
    X1 = []
    for i in range(0, len(X)):
        if Ytrn[i] == 1:
            X1.append(X[i])
        else:
            X0.append(X[i])
    plt.grid()
    for i in range(0, len(X0)):
        plt.scatter(X0[i][0], X0[i][1], c="blue")
    for i in range(0, len(X1)):
        plt.scatter(X1[i][0], X1[i][1], c="red")
    plt.show()
    first = [x[0] for x in X]
    second = [x[1] for x in X]
    first_cor, second_cor = [], []
    from scipy.stats import pearsonr
    for i in range(0, 9):
        x = Xtrn[:, i]
        correlation, _ = pearsonr(x, first)
        first_cor.append(correlation)
        correlation, _ = pearsonr(x, second)
        second_cor.append(correlation)
    return


# iaml212cw2_q1_5()   # comment this out when you run the function


# Q1.6
def iaml212cw2_q1_6():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)
    Xtrn = Xtrn_s
    Xtst = Xtst_s
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(Xtrn)
    ratio = pca.explained_variance_ratio_
    var = pca.explained_variance_
    com = pca.components_

    plt.subplot(1, 2, 1)
    plt.grid()
    plt.scatter([i for i in range(0, 9)], var)
    plt.subplot(1, 2, 2)
    plt.grid()
    cumulative_ratio = []
    for i in range(0, 9):
        cumulative = 0
        for j in range(0, i + 1):
            cumulative += ratio[j]
        cumulative_ratio.append(cumulative)
    plt.ylim(0, 1)
    plt.scatter([i for i in range(0, 9)], cumulative_ratio)
    plt.show()

    vec0 = com[0]
    vec1 = com[1]
    X = []
    for x in Xtrn:
        x = np.array(x)
        X.append([x.dot(vec0), x.dot(vec1)])
    X0 = []
    X1 = []
    for i in range(0, len(X)):
        if Ytrn[i] == 1:
            X1.append(X[i])
        else:
            X0.append(X[i])
    plt.grid()
    for i in range(0, len(X0)):
        plt.scatter(X0[i][0], X0[i][1], c="blue")
    for i in range(0, len(X1)):
        plt.scatter(X1[i][0], X1[i][1], c="red")
    plt.show()
    first = [x[0] for x in X]
    second = [x[1] for x in X]
    first_cor, second_cor = [], []
    from scipy.stats import pearsonr
    for i in range(0, 9):
        x = Xtrn[:, i]
        correlation, _ = pearsonr(x, first)
        first_cor.append(correlation)
        correlation, _ = pearsonr(x, second)
        second_cor.append(correlation)
    return


# iaml212cw2_q1_6()   # comment this out when you run the function

# Q1.8
def iaml212cw2_q1_8():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold, StratifiedKFold
    best_result = -1
    best_C = -2
    C_list = []
    mean_list = []
    std_list = []
    for i in range(0, 13):
        index = -2 + 1 / 3 * i
        exp = 10**index
        clf = svm.SVC(kernel="rbf", C=exp)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        C_list.append(exp)
        score = cross_val_score(clf, Xtrn_s, Ytrn, cv=5)
        mean_list.append(np.mean(score))
        std_list.append(np.std(score))
        result = np.mean(score)
        if best_result == -1:
            best_result = result
            best_C = exp
        elif result > best_result:
            best_result = result
            best_C = exp
    # print(best_result)
    # print(best_C)
    plt.grid()
    tick = C_list
    labels = [str(t)[0:4] for t in tick]
    plt.xticks(ticks=[i for i in range(0, 13)], labels=labels, rotation=30)
    # plt.scatter([i for i in range(0, 13)], mean_list)
    plt.errorbar([i for i in range(0, 13)], mean_list, yerr=std_list)
    plt.show()

    clf = svm.SVC(kernel="rbf", C=0.46415888336127775)
    clf.fit(Xtrn_s, Ytrn)
    result = clf.predict(Xtst_s)
    num = 0
    for i in range(0, len(result)):
        if result[i] == Ytst[i]:
            num += 1
    print(num)
    return


# iaml212cw2_q1_8()   # comment this out when you run the function

# Q1.9
def iaml212cw2_q1_9():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    Z = []
    for i in range(0, len(Xtrn)):
        if Ytrn[i] == 0 and Xtrn[i][4] >= 1:
            Z.append([Xtrn[i][0], Xtrn[i][4]])
    Zrn = np.array(Z)
    x = np.array(Zrn[:, 0])
    y = np.array(Zrn[:, 1])
    n = len(x)
    mean = [np.mean(x), np.mean(y)]
    cov = np.cov(x, y)
    mu1 = mean[0]
    mu2 = mean[1]
    sigma1 = np.sqrt(cov[0][0])
    sigma2 = np.sqrt(cov[1][1])
    rho = cov[0][1] / np.sqrt(cov[0][0] * cov[1][1])

    def f(x, y, mu1, mu2, sigma1, sigma2, rho):
        result = 1.0 / (2 * 3.1415926 * sigma1 * sigma2 * np.sqrt(1 - rho**2))
        result *= np.exp(
            -1 * (1.0 / (2 * (1 - rho**2))) *
            (
                    (x - mu1)**2 / sigma1**2
                    - 2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
                    + (y - mu2)**2 / sigma2**2
             )
        )
        return result

    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(0, len(X)):
        z = []
        for j in range(0, len(X)):
            z.append(f(X[i][j], Y[i][j], mu1, mu2, sigma1, sigma2, rho))
        Z.append(z)
    Z = np.array(Z)
    plt.contour(X, Y, Z)
    plt.grid()
    plt.show()

    return


# iaml212cw2_q1_9()   # comment this out when you run the function

# Q1.10
def iaml212cw2_q1_10():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    Z = []
    for i in range(0, len(Xtrn)):
        if Ytrn[i] == 0 and Xtrn[i][4] >= 1:
            Z.append([Xtrn[i][0], Xtrn[i][4]])
    Zrn = np.array(Z)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(Zrn, [1 for i in range(0, len(Zrn))])

    x = np.array(Zrn[:, 0])
    y = np.array(Zrn[:, 1])

    mean = clf.theta_
    mu1 = mean[0][0]
    mu2 = mean[0][1]
    var = clf.var_
    sigma1 = np.sqrt(var[0][0])
    sigma2 = np.sqrt(var[0][1])
    rho = 0

    def f(x, y, mu1, mu2, sigma1, sigma2, rho):
        result = 1.0 / (2 * 3.1415926 * sigma1 * sigma2 * np.sqrt(1 - rho**2))
        result *= np.exp(
            -1 * (1.0 / (2 * (1 - rho**2))) *
            (
                    (x - mu1)**2 / sigma1**2
                    - 2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
                    + (y - mu2)**2 / sigma2**2
             )
        )
        return result

    X, Y = np.meshgrid(x, y)
    Z = []
    for i in range(0, len(X)):
        z = []
        for j in range(0, len(X)):
            z.append(f(X[i][j], Y[i][j], mu1, mu2, sigma1, sigma2, rho))
        Z.append(z)
    Z = np.array(Z)
    plt.contour(X, Y, Z)
    plt.grid()
    plt.show()
    return


# iaml212cw2_q1_10()   # comment this out when you run the function

# Q1.11
def iaml212cw2_q1_11():
    X, Y = load_Q1_dataset()
    Xtrn = X[100:, :]
    Ytrn = Y[100:]
    Xtst = X[0:100, :]
    Ytst = Y[0:100]
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(Xtrn, Ytrn)
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(clf, Xtrn, Ytrn, cv=5)
    mean = np.mean(score)
    std = np.std(score)
    mean_list = []
    std_list = []
    for i in range(0, 9):
        X = np.delete(Xtrn, i, axis=1)
        clf = LogisticRegression(random_state=0, max_iter=1000)
        score = cross_val_score(clf, X, Ytrn, cv=5)
        mean = np.mean(score)
        std = np.std(score)
        mean_list.append(mean)
        std_list.append(std)
    return

# iaml212cw2_q1_11()   # comment this out when you run the function
