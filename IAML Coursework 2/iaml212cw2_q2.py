
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
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

Xtrn_org, Ytrn_org, Xtst_org, Ytst_org = load_Q2_dataset()
Xtrn = Xtrn_org / 255.0
Xtst = Xtst_org / 255.0
Ytrn = Ytrn_org - 1
Ytst = Ytst_org - 1
Xmean = np.mean(Xtrn, axis=0)
Xtrn_m = Xtrn - Xmean
Xtst_m = Xtst - Xmean
#<----

# Q2.1
def iaml212cw2_q2_1():
    train_info = {}
    train_info['min'] = np.min(Xtrn)
    train_info['max'] = np.max(Xtrn)
    train_info['mean'] = np.mean(Xtrn)
    train_info['std'] = np.std(Xtrn)
    test_info = {}
    test_info['min'] = np.min(Xtst)
    test_info['max'] = np.max(Xtst)
    test_info['mean'] = np.mean(Xtst)
    test_info['std'] = np.std(Xtst)
    print('train:', train_info)
    print('test:', test_info)

    def show(image, label):
        image = image.reshape(28, 28)
        image = 1.0 - image
        plt.imshow(image, cmap='gray')
        plt.title('Class number: %d' % label)

    plt.figure()
    plt.subplot(121)
    show(Xtrn[0], Ytrn[0])

    plt.subplot(122)
    show(Xtrn[1], Ytrn[1])
    plt.show()

iaml212cw2_q2_1()   # comment this out when you run the function

# Q2.3
def iaml212cw2_q2_3():
    index0 = []
    index5 = []
    index8 = []
    for i in range(len(Xtrn)):
        if Ytrn[i] == 0:
            index0.append(i)
        if Ytrn[i] == 5:
            index5.append(i)
        if Ytrn[i] == 8:
            index8.append(i)

    Xtrn0 = Xtrn[index0]
    Xtrn5 = Xtrn[index5]
    Xtrn8 = Xtrn[index8]
    datas = [Xtrn0, Xtrn5, Xtrn8]

    title_names = ['class number: 0 (\'A\')',
                   'class number: 5 (\'F\')',
                   'class number: 8 (\'I\')']

    ks = [3, 5]  # parameter k
    font_sizes = [9, 8]  # font size

    for j, k in enumerate(ks):
        count = 1
        plt.figure()
        for i, cluster_data in enumerate(datas):
            model = KMeans(n_clusters=k, random_state=0)
            model.fit(cluster_data)
            centers = model.cluster_centers_
            for image in centers:
                image = 1.0 - image.reshape(28, 28)
                plt.subplot(3, k, count)
                plt.imshow(image, cmap='gray')
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                plt.title(title_names[i], fontsize=font_sizes[j])
                count += 1
        plt.show()

iaml212cw2_q2_3()   # comment this out when you run the function

# Q2.5
def iaml212cw2_q2_5():
    model = LogisticRegression(max_iter=1000, C=0.5, random_state=0)
    model.fit(Xtrn_m, Ytrn)
    train_acc = model.score(Xtrn_m, Ytrn)
    test_acc = model.score(Xtst_m, Ytst)
    print('train accuracy:', train_acc)
    print('test accuracy:', test_acc)

    label_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    test_pred = model.predict(Xtst_m)
    class_miscount = np.zeros(26)
    mis_index = []
    for i in range(len(test_pred)):
        if test_pred[i] != Ytst[i]:
            mis_index.append(i)
            class_miscount[Ytst[i]] += 1

    sorted_index = np.argsort(-class_miscount)

    sorted_class_miscount = class_miscount[sorted_index]

    top5class_index = sorted_index[:5]
    top5class_name = [label_name[i] for i in top5class_index]
    top5class_misnumber = sorted_class_miscount[:5]

    print('\nTop five classes that were misclassified most in the test set:')
    print('Class numbers:', top5class_index)
    print('Alphbet letters:', top5class_name)
    print('Numbers of misclassifications', top5class_misnumber)

    def show(image, label):
        image = image.reshape(28, 28)
        image = 1.0 - image
        plt.imshow(image, cmap='gray')
        plt.title('Class number: %d' % label)

iaml212cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml212cw2_q2_6():
    # Phase1. Grid search
    para1s = [100, 200, 500, 1000, 1500]
    para2s = [0.01, 0.05, 0.1, 0.7, 1.0]
    test_acc_grid = np.zeros((5, 5))
    for i, para1 in enumerate(para1s):
        for j, para2 in enumerate(para2s):
            model = LogisticRegression(max_iter=para1, C=para2, random_state=0)
            model.fit(Xtrn_m, Ytrn)
            test_acc_grid[i, j] = model.score(Xtst_m, Ytst)
    print('Grid search:\n', test_acc_grid)

    # Phase2. Construct the improved model
    # these two parameter is obtained by Grid search
    best_model = LogisticRegression(max_iter=100, C=0.05, random_state=0)
    best_model.fit(Xtrn_m, Ytrn)
    test_acc_improved = best_model.score(Xtst_m, Ytst)
    print('After improvement, the test accuracy is: ', test_acc_improved)

iaml212cw2_q2_6()   # comment this out when you run the function

# Q2.7 
def iaml212cw2_q2_7():
    index = []
    for i in range(len(Xtrn_m)):
        if Ytrn[i] == 0:
            index.append(i)
    Xtrn_A = Xtrn_m[index]

    cov = np.cov(Xtrn_A, rowvar=False)
    cov_min = np.min(cov)
    cov_max = np.max(cov)
    cov_mean = np.mean(cov)

    cov_diag = [cov[i, i] for i in range(len(cov))]
    cov_diag_min = np.min(cov_diag)
    cov_diag_max = np.max(cov_diag)
    cov_diag_mean = np.mean(cov_diag)

    plt.hist(cov_diag, bins=15)
    plt.xlabel('Diagonal values of the covariance matrix')
    plt.title('Histogram of the diagonal values')
    plt.grid()
    plt.show()

    for i in range(len(Ytst)):
        if Ytst[i] == 0:
            test0index = i
            break
    gaussian = multivariate_normal(mean=np.mean(Xtrn_A, axis=0), cov=cov, allow_singular=True)
    likelihood = gaussian.pdf(Xtst[test0index])
    print('The likelihood of the first element of class 0 in the test set is: ', likelihood)

iaml212cw2_q2_7()   # comment this out when you run the function

# Q2.8 
def iaml212cw2_q2_8():
    index = []
    for i in range(len(Xtrn_m)):
        if Ytrn[i] == 0:
            index.append(i)
    Xtrn_A = Xtrn_m[index]

    model = GaussianMixture(n_components=1, covariance_type='full')
    model.fit(Xtrn_A)
    log_likelihood = model.predict_proba(Xtst[0].reshape(1, -1))
    print('Log-likelihood of the first instance in the test set:', str(int(log_likelihood[0][0])))

    train_correct_num = []
    train_acc = []
    test_correct_num = []
    test_acc = []
    index = np.argsort(Ytst)
    Ytst_sorted = Ytst[index]
    Xtst_sorted = Xtst_m[index]
    for j in range(26):
        index = []
        for i in range(len(Xtrn_m)):
            if Ytrn[i] == j:
                index.append(i)
        Xtrn_A = Xtrn_m[index]

        index = []
        for i in range(len(Xtst_m)):
            if Ytst_sorted[i] == j:
                index.append(i)
        Xtst_A = Xtst_sorted[index]

        model = GaussianMixture(n_components=1, covariance_type='full')
        model.fit(Xtrn_A)

        train_pred = model.predict_proba(Xtrn_A)
        train_acc.append(np.mean(train_pred > 0.5))
        train_correct_num.append(np.mean(train_pred > 0.5) * len(Xtrn_A))

        test_pred = model.predict_proba(Xtst_A)
        test_acc.append(np.mean(test_pred > 0.5))
        test_correct_num.append(np.mean(test_pred > 0.5) * len(Xtst_A))

iaml212cw2_q2_8()   # comment this out when you run the function

# Q2.10 
def iaml212cw2_q2_10():
    sorted_index = np.argsort(Ytrn)
    Xtrn_sorted = Xtrn_m[sorted_index]
    Ytrn_sorted = Ytrn[sorted_index]

    sorted_index = np.argsort(Ytst)
    Xtst_sorted = Xtst_m[sorted_index]
    Ytst_sorted = Ytst[sorted_index]

    # # (a)
    train_acc = []
    test_acc = []
    for k in [1, 2, 4, 8]:
        train_x = Xtrn_sorted[:300*k]
        train_y = Ytrn_sorted[:300*k]
        test_x = Xtst_sorted[:100*k]
        test_y = Ytst_sorted[:100*k]
        model = GaussianMixture(n_components=k, random_state=0, covariance_type='full')
        model.fit(train_x)
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        train_acc.append(np.mean(train_pred == train_y))
        test_acc.append(np.mean(test_pred == test_y))

    print('train accuracy:', train_acc)
    print('test accuracy:', test_acc)

    # (b)
    paras = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
    train_acc = []
    test_acc = []
    for reg_covar in paras:
        k = 2
        train_x = Xtrn_sorted[:300*k]
        train_y = Ytrn_sorted[:300*k]
        test_x = Xtst_sorted[:100*k]
        test_y = Ytst_sorted[:100*k]
        model = GaussianMixture(n_components=k, reg_covar=reg_covar, random_state=0, covariance_type='full')
        model.fit(train_x)
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        train_acc.append(np.mean(train_pred == train_y))
        test_acc.append(np.mean(test_pred == test_y))

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(train_acc)), train_acc, 'o--', color="blue", label='train accuracy')
    ax1.set_xlabel("ref_covar")
    ax1.set_ylabel("Train accuracy")
    plt.xticks(range(len(train_acc)), paras, rotation=30)
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(range(len(test_acc)), test_acc, 'o--', color="red", label="test deviation")
    ax2.set_ylabel("Test accuracy")
    plt.legend(loc='upper right')
    plt.show()

    print('The highest test-set accuracy: ', max(test_acc))
    print('The corresponding parameter reg_covar: ', paras[np.argsort(test_acc)[-1]])

iaml212cw2_q2_10()   # comment this out when you run the function










