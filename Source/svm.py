import numpy as np


class SVM:
    def __init__(self, max_iter=75000, alpha=0.0001):
        self.__w_train = None
        self.__max_iter = max_iter
        self.__alpha = alpha

    @staticmethod
    def __init_f_vector(x, n_dimens, n_sample):
        features = []
        for i in range(n_dimens):
            temp = x[:, i]
            temp = temp.reshape((n_sample, 1))
            features.append(temp)

        features = np.array(features)
        return features

    def __init_w_train(self, n_dimens, n_sample):
        self.__w_train = []
        for i in range(n_dimens):
            self.__w_train.append(np.zeros((n_sample, 1)))

        self.__w_train = np.array(self.__w_train)

    def __init_w_test(self, n_dimens, n_sample):
        w_test = []
        for i in range(n_dimens):
            temp = self.__w_train[i, :n_sample]
            w_test.append(temp)

        w_test = np.array(w_test)
        return w_test

    def fit(self, x_train, y_train):
        x = x_train
        n_sample = x.shape[0]
        n_dimens = x.shape[1]
        y = y_train.reshape((n_sample, 1))

        self.__init_w_train(n_dimens, n_sample)
        features = self.__init_f_vector(x, n_dimens, n_sample)

        for epoch in range(self.__max_iter):
            reg_param = 1 / (epoch + 1)
            f_x = self.__w_train * features
            f_x = f_x.sum(axis=0)
            pred = f_x * y

            for index, value in enumerate(pred):
                if value >= 1:
                    # Cost = 0
                    for i, w in enumerate(self.__w_train):
                        w_new = w - self.__alpha * (2 * reg_param * w)
                        self.__w_train[i] = w_new

                else:
                    # Cost = 1 - value
                    for i, w in enumerate(self.__w_train):
                        w_new = w + self.__alpha * (features[i][index] * y[index] - (2 * reg_param * w))
                        self.__w_train[i] = w_new

    def predict(self, x_test):
        x = x_test
        n_sample = x_test.shape[0]
        n_dimens = x_test.shape[1]

        features = self.__init_f_vector(x, n_dimens, n_sample)
        w_test = self.__init_w_test(n_dimens, n_sample)

        f_x = w_test * features
        f_x = f_x.sum(axis=0)

        predictions = []
        for value in f_x:
            if value > 1:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions

    def get_weights(self):
        return self.__w_train
