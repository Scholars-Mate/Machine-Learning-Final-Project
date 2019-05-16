import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def main():
    # Import training datasets
    gisette_train = np.loadtxt('datasets/gisette/gisette_train.data')
    gisette_train_labels = np.loadtxt('datasets/gisette/gisette_train.labels')

    # Import validation datasets
    gisette_valid = np.loadtxt('datasets/gisette/gisette_valid.data')
    gisette_valid_labels = np.loadtxt('datasets/gisette/gisette_valid.labels')

    (e_values,pca_w) = get_pca_t_matrix(gisette_train)
    (_,lda_w) = get_lda_t_matrix(gisette_train, gisette_train_labels, 5000, 3000, 3000)

    # Try logistic regression
    no_features = [5000, 2500, 1000, 500, 100, 50, 5, 2]
    pca_acc = []
    lda_acc = []
    for features in no_features:
        lg_pca = LogisticRegression(solver='liblinear')
        lg_pca.fit((pca_w[:, 0:features].transpose() @ gisette_train.transpose()).transpose(), gisette_train_labels)
        lg_lda = LogisticRegression(solver='liblinear')
        lg_lda.fit((lda_w[:, 0:features].transpose() @ gisette_train.transpose()).transpose(), gisette_train_labels)
        pca_res = lg_pca.predict((pca_w[:, 0:features].transpose() @ gisette_valid.transpose()).transpose())
        lda_res = lg_lda.predict((lda_w[:, 0:features].transpose() @ gisette_valid.transpose()).transpose())
        pca_cor = 0
        lda_cor = 0
        for (pca, lda, actual) in zip(pca_res, lda_res, gisette_valid_labels):
            if pca == actual: pca_cor += 1
            if lda == actual: lda_cor += 1
        pca_acc.append(pca_cor / len(gisette_valid_labels))
        lda_acc.append(lda_cor / len(gisette_valid_labels))
    plt.figure()
    ind = np.arange(len(no_features))
    width = 0.35
    plt.bar(ind, pca_acc, width, label='PCA')
    plt.bar(ind + width, lda_acc, width, label='LDA')
    plt.title('Logistic Regression of Gisette Dataset')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.xticks(ind + width / 2, map(str, no_features))
    plt.legend(loc='best')

    print('Logistic Regression of Gisette')
    for (pca, lda, feat) in zip(pca_acc, lda_acc, no_features):
        print(feat)
        print('PCA: {}'.format(pca))
        print('LDA: {}'.format(lda))
    print('')

    # Try knn
    pca_acc = []
    lda_acc = []
    for features in no_features:
        knn_pca = KNeighborsClassifier(n_neighbors=3, algorithm='brute', n_jobs=-1)
        knn_pca.fit(np.matmul(pca_w[:, 0:features].transpose(), gisette_train.transpose()).transpose(), gisette_train_labels)
        knn_lda = KNeighborsClassifier(n_neighbors=3, algorithm='brute', n_jobs=-1)
        knn_lda.fit(np.matmul(lda_w[:, 0:features].transpose(), gisette_train.transpose()).transpose(), gisette_train_labels)
        pca_res = knn_pca.predict(np.matmul(pca_w[:, 0:features].transpose(), gisette_valid.transpose()).transpose())
        lda_res = knn_lda.predict(np.matmul(lda_w[:, 0:features].transpose(), gisette_valid.transpose()).transpose())
        pca_cor = 0
        lda_cor = 0
        for (pca, lda, actual) in zip(pca_res, lda_res, gisette_valid_labels):
            if pca == actual: pca_cor += 1
            if lda == actual: lda_cor += 1
        pca_acc.append(pca_cor / len(gisette_valid_labels))
        lda_acc.append(lda_cor / len(gisette_valid_labels))
    plt.figure()
    ind = np.arange(len(no_features))
    width = 0.35
    plt.bar(ind, pca_acc, width, label='PCA')
    plt.bar(ind + width, lda_acc, width, label='LDA')
    plt.title('K-Nearest Neighbors (K=3) of Gisette Dataset')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.xticks(ind + width / 2, map(str, no_features))
    plt.legend(loc='best')

    print('KNN of Gisette')
    for (pca, lda, feat) in zip(pca_acc, lda_acc, no_features):
        print(feat)
        print('PCA: {}'.format(pca))
        print('LDA: {}'.format(lda))
    print('')

    # Plot 2D projection of data
    plt.figure()
    plt.title('PCA 2D Projection of Gisette Dataset')
    plt.scatter(pca_w[:, 0].transpose() @ gisette_train.transpose(), pca_w[:, 1].transpose() @ gisette_train.transpose(), c=[x + 1 for x in gisette_train_labels])
    plt.figure()
    plt.title('LDA 2D Projection of Gisette Dataset')
    plt.scatter(lda_w[:, 0].transpose() @ gisette_train.transpose(), lda_w[:, 1].transpose() @ gisette_train.transpose(), c=[x + 1 for x in gisette_train_labels])
    
    # Throw away old dataset
    gisette_train = None
    gisette_train_labels = None
    gisette_valid = None
    gisette_valid_labels = None
    pca_w = None
    lda_w = None

    # Try to load cached transformation matrices
    dexter_train = read_dexter_train()
    dexter_train_labels = np.loadtxt('datasets/dexter/dexter_train.labels')
    try:
        pca_w = np.load('dexter_pca.npy')
    except Exception as e:
        print('Calculating PCA transformation matrix...')
        (_,pca_w) = get_pca_t_matrix(dexter_train)
        np.save('dexter_pca.npy', pca_w)
    try:
        lda_w = np.load('dexter_lda.npy')
    except:
        print('Calculating LDA transformation matrix...')
        (_,lda_w) = get_lda_t_matrix(dexter_train, dexter_train_labels, 20000, 150, 150)
        np.save('dexter_lda.npy', lda_w)

    dexter_valid = read_dexter_valid()
    dexter_valid_labels = np.loadtxt('datasets/dexter/dexter_valid.labels')

    # Try logistic regression
    no_features = [20000, 10000, 5000, 1000, 500, 250, 100, 50, 2]
    pca_acc = []
    lda_acc = []
    for features in no_features:
        lg_pca = LogisticRegression(solver='liblinear')
        lg_pca.fit((pca_w[:, 0:features].transpose() @ dexter_train.transpose()).transpose(), dexter_train_labels)
        lg_lda = LogisticRegression(solver='liblinear')
        lg_lda.fit((lda_w[:, 0:features].transpose() @ dexter_train.transpose()).transpose(), dexter_train_labels)
        pca_res = lg_pca.predict((pca_w[:, 0:features].transpose() @ dexter_valid.transpose()).transpose())
        lda_res = lg_lda.predict((lda_w[:, 0:features].transpose() @ dexter_valid.transpose()).transpose())
        pca_cor = 0
        lda_cor = 0
        for (pca, lda, actual) in zip(pca_res, lda_res, dexter_valid_labels):
            if pca == actual: pca_cor += 1
            if lda == actual: lda_cor += 1
        pca_acc.append(pca_cor / len(dexter_valid_labels))
        lda_acc.append(lda_cor / len(dexter_valid_labels))
    plt.figure()
    ind = np.arange(len(no_features))
    width = 0.35
    plt.bar(ind, pca_acc, width, label='PCA')
    plt.bar(ind + width, lda_acc, width, label='LDA')
    plt.title('Logistic Regression of Dexter Dataset')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.xticks(ind + width / 2, map(str, no_features))
    plt.legend(loc='best')

    print('Logistic Regression of Dexter')
    for (pca, lda, feat) in zip(pca_acc, lda_acc, no_features):
        print(feat)
        print('PCA: {}'.format(pca))
        print('LDA: {}'.format(lda))
    print('')

    pca_acc = []
    lda_acc = []
    for features in no_features:
        knn_pca = KNeighborsClassifier(n_neighbors=3, algorithm='brute', n_jobs=-1)
        knn_pca.fit((pca_w[:, 0:features].transpose() @ dexter_train.transpose()).transpose(), dexter_train_labels)
        knn_lda = KNeighborsClassifier(n_neighbors=3, algorithm='brute', n_jobs=-1)
        knn_lda.fit((lda_w[:, 0:features].transpose() @ dexter_train.transpose()).transpose(), dexter_train_labels)
        pca_res = knn_pca.predict((pca_w[:, 0:features].transpose() @ dexter_valid.transpose()).transpose())
        lda_res = knn_lda.predict((lda_w[:, 0:features].transpose() @ dexter_valid.transpose()).transpose())
        pca_cor = 0
        lda_cor = 0
        for (pca, lda, actual) in zip(pca_res, lda_res, dexter_valid_labels):
            if pca == actual: pca_cor += 1
            if lda == actual: lda_cor += 1
        pca_acc.append(pca_cor / len(dexter_valid_labels))
        lda_acc.append(lda_cor / len(dexter_valid_labels))
    plt.figure()
    ind = np.arange(len(no_features))
    width = 0.35
    plt.bar(ind, pca_acc, width, label='PCA')
    plt.bar(ind + width, lda_acc, width, label='LDA')
    plt.title('K-Nearest Neighbors (K=3) of Dexter Dataset')
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.xticks(ind + width / 2, map(str, no_features))
    plt.legend(loc='best')

    print('KNN of Dexter')
    for (pca, lda, feat) in zip(pca_acc, lda_acc, no_features):
        print(feat)
        print('PCA: {}'.format(pca))
        print('LDA: {}'.format(lda))

    # Plot 2D projection of data
    plt.figure()
    plt.title('PCA 2D Projection of Dexter Dataset')
    plt.scatter(pca_w[:, 0].transpose() @ dexter_train.transpose(), pca_w[:, 1].transpose() @ dexter_train.transpose(), c=[x + 1 for x in dexter_train_labels])
    plt.figure()
    plt.title('LDA 2D Projection of Dexter Dataset')
    plt.scatter(lda_w[:, 0].transpose() @ dexter_train.transpose(), lda_w[:, 1].transpose() @ dexter_train.transpose(), c=[x + 1 for x in dexter_train_labels])
    
    plt.show()

def get_pca_t_matrix(training_data):
    # Get the transformation matrix using PCA
    e_values, w = np.linalg.eigh(np.cov(training_data.transpose()))
    # Eigenvalues and vectors are sorted in ascending order; Need descending order
    e_values = np.flip(e_values)
    # Can't use a flip like above. numpy.fliplr would return a view, which destroys
    # performance when using matrix multiplication down below
    for i in range(0, w.shape[1] // 2):
        tmp = w[:, i]
        w[:, i] = w[:, w.shape[1] - 1 - i]
        w[:, w.shape[1] - 1 - i] = tmp

    return (e_values, w)

def get_lda_t_matrix(training_data, training_labels, n_features, n_c0, n_c1):
    # Seperate classes
    train_class0 = np.empty((n_c0, n_features)) # Corresponds to label -1
    train_class1 = np.empty((n_c1, n_features)) # Correspends to label  1
    c0pos = 0
    c1pos = 0
    for (row, label) in zip(training_data, training_labels):
        if label == -1:
            train_class0[c0pos, :] = row
            c0pos += 1
        elif label == 1:
            train_class1[c1pos, :] = row
            c1pos += 1

    # Within class scatter
    sw = np.cov(train_class0.transpose()) + np.cov(train_class1.transpose())

    # Between class scatter
    c0mean = np.mean(train_class0, axis=0)
    c1mean = np.mean(train_class1, axis=0)
    sb = np.outer((c0mean - c1mean), (c0mean - c1mean))

    # Get eigenvectors
    e_values, w = np.linalg.eigh(np.cov(sw.transpose() @ sb))
    # Eigenvalues and vectors are sorted in ascending order; Need descending order
    e_values = np.flip(e_values)
    # Can't use a flip like above. numpy.fliplr would return a view, which destroys
    # performance when using matrix multiplication down below
    for i in range(0, w.shape[1] // 2):
        tmp = w[:, i]
        w[:, i] = w[:, w.shape[1] - 1- i]
        w[:, w.shape[1] - 1 - i] = tmp

    return (e_values, w)

def read_dexter_train():
    dexter_train = np.zeros((300, 20000))
    with open('datasets/dexter/dexter_train.data') as f:
        lineno = 0
        for line in f:
            for pair in line.split():
                (col, val) = pair.split(':')
                dexter_train[lineno, int(col) - 1] = int(val)
            lineno += 1
    return dexter_train

def read_dexter_valid():
    dexter_valid = np.zeros((300, 20000))
    with open('datasets/dexter/dexter_valid.data') as f:
        lineno = 0
        for line in f:
            for pair in line.split():
                (col, val) = pair.split(':')
                dexter_valid[lineno, int(col) - 1] = int(val)
            lineno += 1
    return dexter_valid

def read_dorothea_train():
    dorothea_train = np.zeros((800, 100000))
    with open('datasets/dorothea/dorothea_train.data') as f:
        lineno = 0
        for line in f:
            for number in line.split():
                dorothea_train[lineno, int(number) - 1] = 1
            lineno += 1
    return dorothea_train

def read_dorothea_valid():
    dorothea_valid = np.zeros((350, 100000))
    with open('datasets/dorothea/dorothea_valid.data') as f:
        lineno = 0
        for line in f:
            for number in line.split():
                dorothea_train[lineno, int(number) - 1] = 1
    return dorothea_valid

if __name__ == '__main__':
    main()
