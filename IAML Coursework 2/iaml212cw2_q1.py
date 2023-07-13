
##########################################################
#  Python script template for Question 1 (IAML Level 11)
#  Note that
#  - You should not change the name of this file, 'iaml212cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import matplotlib.pyplot as plt
from iaml_cw2_helpers import *
# from iaml212cw2_my_helpers import *

import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression

X, Y = load_Q1_dataset()
print('X: ', X.shape, 'Y: ', Y.shape)
Xtrn = X[100:, :]
Ytrn = Y[100:]
Xtst = X[0:100, :]
Ytst = Y[0:100]
#<----

# Q1.1
def iaml212cw2_q1_1():
    index0 = []
    index1 = []
    for i in range(len(Ytrn)):
        if Ytrn[i] == 0:
            index0.append(i)
        else:
            index1.append(i)

    plt.figure()
    for i in range(9):
        x0 = Xtrn[index0, i]
        x1 = Xtrn[index1, i]
        plt.subplot(3, 3, i+1)
        plt.hist([x0, x1], bins=15, label=['class 0', 'class 1'])
        plt.legend(loc='upper right', fontsize=7)
        plt.title('Attribute A{}'.format(i), fontsize=8)
        plt.grid()
    plt.show()

iaml212cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml212cw2_q1_2():
    cc = []
    for i in range(9):
        cc.append(stats.pearsonr(Xtrn[:, i], Ytrn)[0])
    print(cc)

iaml212cw2_q1_2()   # comment this out when you run the function

# Q1.4
def iaml212cw2_q1_4():
    attribute_name = np.array(['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'])
    var = np.var(Xtrn, axis=0)
    sort_index = np.argsort(-var)

    var_sorted = var[sort_index]
    attribute_sorted = attribute_name[sort_index]

    var_sum = sum(var)
    print('Sum of variance: ', var_sum)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(9), var_sorted)
    plt.xticks(range(9), attribute_sorted)
    plt.xlabel('attributes')
    plt.ylabel('Explained_variance')
    plt.grid()

    attribute_ratio = var_sorted / var_sum
    cumulative_ratio = []
    cumu = 0
    for i in range(len(attribute_ratio)):
        cumu = cumu + attribute_ratio[i]
        cumulative_ratio.append(cumu)
    plt.subplot(1, 2, 2)
    plt.bar(range(len(attribute_ratio)), cumulative_ratio)
    plt.xticks(range(len(attribute_ratio)), np.arange(1, len(attribute_ratio) + 1))
    plt.xlabel('the number of attributes')
    plt.ylabel('cumulative variance ratio')
    plt.grid()
    plt.show()

iaml212cw2_q1_4()   # comment this out when you run the function

# Q1.5
def iaml212cw2_q1_5():
    model = PCA()
    model.fit(Xtrn)
    Principal_name = np.array(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'])

    explained_variance = model.explained_variance_
    total_explained_variance = sum(explained_variance)
    print('Total amount of unbiased sample variance explained by the whole set of principal components is: {}'
          .format(total_explained_variance))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(explained_variance)), explained_variance)
    plt.xticks(range(len(explained_variance)), Principal_name)
    plt.xlabel('Principal components')
    plt.ylabel('Explained_variance')
    plt.grid()

    explained_variance_ratio = model.explained_variance_ratio_
    cumulative_ratio = []
    cumu = 0
    for i in range(len(explained_variance_ratio)):
        cumu = cumu + explained_variance_ratio[i]
        cumulative_ratio.append(cumu)
    plt.subplot(1, 2, 2)
    plt.bar(range(len(explained_variance_ratio)), cumulative_ratio)
    plt.xticks(range(len(explained_variance_ratio)), np.arange(1, len(explained_variance_ratio) + 1))
    plt.xlabel('the number of principal components')
    plt.ylabel('Cumulative variance ratio')
    plt.grid()
    plt.show()

    eigen_vectors = model.components_[:2, :]
    Xtrn_features = Xtrn.dot(eigen_vectors.T)
    index0 = []
    index1 = []
    for i in range(len(Ytrn)):
        if Ytrn[i] == 0:
            index0.append(i)
        else:
            index1.append(i)
    Xtrn_features0 = Xtrn_features[index0]
    Xtrn_features1 = Xtrn_features[index1]
    plt.scatter(Xtrn_features0[:, 0], Xtrn_features0[:, 1], c='blue', label='class 0')
    plt.scatter(Xtrn_features1[:, 0], Xtrn_features1[:, 1], c='red', label='class 1')
    plt.xlabel('Principal 1')
    plt.ylabel('Principal 2')
    plt.grid()
    plt.legend()
    plt.show()

    cc = np.zeros([2, 9])
    for i in range(2):
        for j in range(9):
            cc[i, j] = stats.pearsonr(Xtrn_features[:, i], Xtrn[:, j])[0]
    print('The correlation coefficient between each attribute and '
          'each of the first and second principal components is:\n', cc)

iaml212cw2_q1_5()   # comment this out when you run the function

# Q1.6
def iaml212cw2_q1_6():
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    model = PCA()
    model.fit(Xtrn_s)
    Principal_name = np.array(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'])

    explained_variance = model.explained_variance_
    total_explained_variance = sum(explained_variance)
    print('Total amount of unbiased sample variance explained by the whole set of principal components is: {}'
          .format(total_explained_variance))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(explained_variance)), explained_variance)
    plt.xticks(range(len(explained_variance)), Principal_name)
    plt.xlabel('Principal components')
    plt.ylabel('Explained_variance')
    plt.grid()

    explained_variance_ratio = model.explained_variance_ratio_
    cumulative_ratio = []
    cumu = 0
    for i in range(len(explained_variance_ratio)):
        cumu = cumu + explained_variance_ratio[i]
        cumulative_ratio.append(cumu)
    plt.subplot(1, 2, 2)
    plt.bar(range(len(explained_variance_ratio)), cumulative_ratio)
    plt.xticks(range(len(explained_variance_ratio)), np.arange(1, len(explained_variance_ratio) + 1))
    plt.xlabel('the number of principal components')
    plt.ylabel('Cumulative variance ratio')
    plt.grid()
    plt.show()

    eigen_vectors = model.components_[:2, :]
    Xtrn_features = Xtrn_s.dot(eigen_vectors.T)
    index0 = []
    index1 = []
    for i in range(len(Ytrn)):
        if Ytrn[i] == 0:
            index0.append(i)
        else:
            index1.append(i)
    Xtrn_features0 = Xtrn_features[index0]
    Xtrn_features1 = Xtrn_features[index1]
    plt.scatter(Xtrn_features0[:, 0], Xtrn_features0[:, 1], c='blue', label='class 0')
    plt.scatter(Xtrn_features1[:, 0], Xtrn_features1[:, 1], c='red', label='class 1')
    plt.xlabel('Principal 1')
    plt.ylabel('Principal 2')
    plt.grid()
    plt.legend()
    plt.show()

    cc = np.zeros([2, 9])
    for i in range(2):
        for j in range(9):
            cc[i, j] = stats.pearsonr(Xtrn_features[:, i], Xtrn_s[:, j])[0]
    print('The correlation coefficient between each scaled attribute and '
          'each of the first and second principal components is:\n', cc)

iaml212cw2_q1_6()   # comment this out when you run the function

# Q1.8
def iaml212cw2_q1_8():
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    paras = [0.01, 0.1, 1, 10, 100]
    means = []
    stds = []
    for C in paras:
        skf = StratifiedKFold(n_splits=5)
        acc_log = []
        for train_index, test_index in skf.split(Xtrn_s, Ytrn):
            X_train, X_test = Xtrn_s[train_index], Xtrn_s[test_index]
            y_train, y_test = Ytrn[train_index], Ytrn[test_index]
            model = SVC(C=C)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            acc_log.append(acc)
        means.append(np.mean(acc_log))
        stds.append(np.std(acc_log))

    plt.errorbar(range(5), means, yerr=stds, fmt='--o', markersize=9, capsize=15, ecolor='red', elinewidth=5)
    plt.xticks(range(5), paras)
    plt.xlabel('parameter C (in log space)')
    plt.ylabel('Classification accuracy')
    plt.show()

    best_acc = max(means)
    best_C = paras[np.argsort(means)[-1]]
    print('The highest mean cross-validation accuracy is {}'.format(best_acc))
    print('The highest mean cross-validation accuracy is yield by C={}'.format(best_C))

    best_model = SVC(C=best_C)
    best_model.fit(Xtrn_s, Ytrn)
    acc = best_model.score(Xtst_s, Ytst)
    correct_number = len(Xtst) * acc
    print('The number of instances correctly classified: %d' % correct_number)
    print('Classification accuracy of the best SVC model: {}'.format(acc))

iaml212cw2_q1_8()   # comment this out when you run the function

# Q1.9
def iaml212cw2_q1_9():
    index = []
    for i in range(len(Ytrn)):
        if Ytrn[i] == 0:
            if Xtrn[i, 4] >= 1.0:
                index.append(i)
    Ztrn = np.column_stack((Xtrn[index][:, 4], Xtrn[index][:, 7]))
    mean_vector = np.mean(Ztrn, axis=0)
    cov_matrix = np.cov(Ztrn, rowvar=False)
    print('mean vector: ', mean_vector)
    print('covariance matrix: ', cov_matrix)

    xx, yy = np.mgrid[-10:70:.01, 0:60:.01]
    pos = np.dstack((xx, yy))
    gaussian = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    zz = gaussian.pdf(pos)

    plt.contourf(xx, yy, zz)
    plt.scatter(Ztrn[:, 0], Ztrn[:, 1], c='red', s=13)
    plt.axis('equal')
    plt.grid()
    plt.show()

iaml212cw2_q1_9()   # comment this out when you run the function

# Q1.10
def iaml212cw2_q1_10():
    index = []
    for i in range(len(Ytrn)):
        if Ytrn[i] == 0:
            if Xtrn[i, 4] >= 1.0:
                index.append(i)
    Ztrn = np.column_stack((Xtrn[index][:, 4], Xtrn[index][:, 7]))
    mean_vector = np.mean(Ztrn, axis=0)
    cov_matrix = np.cov(Ztrn, rowvar=False)
    print('mean vector: ', mean_vector)
    print('covariance matrix: ', cov_matrix)

    xx, yy = np.mgrid[-10:70:.01, 0:60:.01]
    pos = np.dstack((xx, yy))
    gaussian = multivariate_normal(mean=mean_vector, cov=cov_matrix)
    zz = gaussian.pdf(pos)

    plt.contourf(xx, yy, zz)
    plt.scatter(Ztrn[:, 0], Ztrn[:, 1], c='red', s=13)
    plt.axis('equal')
    plt.grid()
    plt.show()

iaml212cw2_q1_10()   # comment this out when you run the function

# Q1.11
def iaml212cw2_q1_11():
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    skf = StratifiedKFold(n_splits=5)
    acc_log = []
    for train_index, test_index in skf.split(Xtrn_s, Ytrn):
        X_train, X_test = Xtrn_s[train_index], Xtrn_s[test_index]
        y_train, y_test = Ytrn[train_index], Ytrn[test_index]
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        acc_log.append(acc)
    mean = np.mean(acc_log)
    std = np.std(acc_log)
    print('The mean of cross-validation accuracy: {}'.format(mean))
    print('The standard deviation of cross-validation accuracy: {}'.format(std))

    means = []
    stds = []
    for i in range(9):
        skf = StratifiedKFold(n_splits=5)
        acc_log = []
        Xtrn_s_drop = np.delete(Xtrn_s, i, axis=1)
        for train_index, test_index in skf.split(Xtrn_s_drop, Ytrn):
            X_train, X_test = Xtrn_s_drop[train_index], Xtrn_s_drop[test_index]
            y_train, y_test = Ytrn[train_index], Ytrn[test_index]
            model = LogisticRegression(max_iter=1000, random_state=0)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            acc_log.append(acc)
        means.append(np.mean(acc_log))
        stds.append(np.std(acc_log))

    print('mean:', means)
    print('std:', stds)

    fig, ax1 = plt.subplots()
    ax1.plot(range(9), means, 'o--', color="blue", label='Mean accuracy')
    ax1.set_xlabel("Dropped attribute")
    ax1.set_ylabel("Mean accuracy")

    ax2 = ax1.twinx()
    ax2.plot(range(9), stds, 'o--', color="red", label="Standard deviation")
    ax2.set_ylabel("Standard deviation")

    fig.legend(loc="upper right")
    plt.show()

iaml212cw2_q1_11()   # comment this out when you run the function





