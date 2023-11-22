import numpy as np
import random
import time
from copy import deepcopy
from sklearn.neighbors import BallTree
from infinite_selection import InfFS


class MDSampler(object):
    def __init__(self, base_estimator, n_estimators=10, n_bins=10, threshold=-1, warmup_rounds=5, ratio_pseudo=1.0,
                 sampling_IR=1.0, ratio_under=1.0, rescale_variance=False, plus=False):
        self.base_estimator = deepcopy(base_estimator)
        self.n_estimators = n_estimators
        self.n_bins = n_bins
        self.threshold = threshold
        self.warmup_rounds = warmup_rounds
        self.ratio_pseudo = ratio_pseudo
        self.sampling_IR = sampling_IR
        self.ratio_under = ratio_under
        self.rescale_variance = rescale_variance
        self.plus = plus

        self.estimators = {}
        self.IR = 0
        self.confidence_N = None
        self.confidence_U = None
        self.is_fitted = False
        self.n_voters = -1
        self.SMOTE = False
        self.feature_selection = False

    def fit(self, X, y, X_test=None, y_test=None):
        if self.plus:
            self.SMOTE = True
            self.feature_selection = True

        X_L = X[np.where(y != -1)]
        y_L = y[np.where(y != -1)]

        # print('%.4f' % (sum(y_L) / len(y_L)))

        X_positive = X[np.where(y == 1)]
        X_negative = X[np.where(y == 0)]
        X_unlabeled = X[np.where(y == -1)]

        n_positive = len(X_positive)
        n_negative = len(X_negative)
        n_unlabeled = len(X_unlabeled)

        n_under = int(n_positive * self.ratio_under)

        self.IR = n_negative / n_positive

        weight = np.ones(len(X_L[0]))

        if self.SMOTE:
            X_positive_temp = deepcopy(X_positive)
            if self.feature_selection:
                inf = InfFS()
                [_, weight] = inf.infFS(X, None, alpha=0.5, supervision=False, verbose=False)
                weight = weight.astype('float64')
                # weight -= min(weight)
                if max(weight) == 0:
                    weight = np.zeros(len(weight))
                else:
                    weight /= max(weight)
                for j in range(0, len(weight)):
                    X_positive_temp[..., j] *= weight[j]
            tree = BallTree(X_positive_temp, leaf_size=int(np.log2(n_positive) + 0.5))

        time_all = []

        for i in range(0, self.warmup_rounds + self.n_estimators):
            # Complete case:
            if i < self.warmup_rounds:
                if i == 0:
                    X_N = X_negative[random.sample(range(n_negative), n_under)]
                else:
                    X_N = X_negative[self.histogram_sampler(n_under, task='N')]
                X_meta = np.concatenate((X_positive[random.sample(range(n_positive), n_under)], X_N))
                y_meta = np.concatenate((np.ones(n_under), np.zeros(len(X_N))))

            # SSL case
            else:
                n_samples = int(self.ratio_pseudo * n_under * (i - self.warmup_rounds + 1) / self.n_estimators + 0.5)
                X_N = X_negative[self.histogram_sampler(int(n_under * self.sampling_IR + 0.5), task='N')]
                X_meta = np.concatenate((X_positive[random.sample(range(n_positive), n_under)], X_N))
                y_meta = np.concatenate((np.ones(n_under), np.zeros(len(X_N))))

                if n_unlabeled > 0:
                    X_UP = X_unlabeled[self.histogram_sampler(n_samples, task='UP')]
                    n_samples_negative = int(len(X_UP) * self.sampling_IR + 0.5)

                    n_from_U = int(n_samples_negative * (1 / self.IR) + 0.5)
                    n_from_L = n_samples_negative - n_from_U
                    X_UN_U = X_unlabeled[self.histogram_sampler(n_from_U, task='UN')]
                    X_UN_L = X_negative[self.histogram_sampler(n_from_L, task='N')]
                    X_UN = np.concatenate((X_UN_U, X_UN_L))

                    if self.SMOTE and len(X_UP) >= 2:
                        X_temp = deepcopy(X_UP)
                        if self.feature_selection:
                            for j in range(0, len(X_temp[0])):
                                X_temp[..., j] *= weight[j]
                        n_neighbors = 3
                        distance, indices = tree.query(X_temp, k=n_neighbors)
                        for j in range(0, len(X_temp)):
                            alpha = 0.5 * random.random()
                            index = random.randint(0, n_neighbors - 1)
                            X_UP[j] = alpha * X_positive[indices[j]][index] + (1 - alpha) * X_UP[j]

                    if len(X_UP) > 0:
                        X_meta = np.concatenate((X_meta, X_UP))
                        y_meta = np.concatenate((y_meta, np.ones(len(X_UP))))
                    if len(X_UN) > 0:
                        X_meta = np.concatenate((X_meta, X_UN))
                        y_meta = np.concatenate((y_meta, np.zeros(len(X_UN))))

            model = deepcopy(self.base_estimator)
            if len(y_meta) - sum(y_meta) != sum(y_meta):
                print('%d %d <---- %d' % (len(y_meta) - sum(y_meta), sum(y_meta), i))

            time_start = time.time()
            model.fit(X_meta, y_meta)
            time_end = time.time()
            time_cost = time_end - time_start
            # print('%.2f' % time_cost)
            time_all.append(time_cost)
            self.estimators[i] = deepcopy(model)

            if i != self.warmup_rounds + self.n_estimators - 1:
                y_pred = model.predict_proba(X_negative)
                if self.confidence_N is None:
                    self.confidence_N = deepcopy([y_pred[..., 0]])
                else:
                    self.confidence_N = np.concatenate((self.confidence_N, [y_pred[..., 0]]))

                if n_unlabeled > 0:
                    y_pred = model.predict_proba(X_unlabeled)
                    if self.confidence_U is None:
                        self.confidence_U = deepcopy([y_pred[..., 0]])
                    else:
                        self.confidence_U = np.concatenate((self.confidence_U, [y_pred[..., 0]]))

            if X_test is not None and y_test is not None:
                pass

        # print('%.2f <---- Train' % (np.sum(np.array(time_all))))

        self.is_fitted = True

    def __sklearn_is_fitted__(self):
        return self.is_fitted

    def histogram_sampler(self, n_sample, task):
        histogram = {}
        indices = []
        indices_U = []

        for i in range(0, self.n_bins):
            histogram[i] = {}
            for j in range(0, self.n_bins):
                histogram[i][j] = []

        if task == 'N':
            confidence = np.mean(self.confidence_N, axis=0)
            agreement = np.std(self.confidence_N, axis=0)
        else:
            confidence = np.mean(self.confidence_U, axis=0)
            if task == 'UP':
                confidence = 1 - confidence
            agreement = np.std(self.confidence_U, axis=0)

            if self.threshold < 0:
                distribution = sorted(confidence, reverse=True)
                if self.threshold == -1:
                    self.threshold = max(0.9, distribution[int(0.5 * len(distribution) / (1 + self.IR) + 0.5)])
                else:
                    self.threshold = min(0.9, distribution[int(0.5 * len(distribution) / (1 + self.IR) + 0.5)])
                # print(('%.2f' % self.threshold).lstrip("0"))

            indices_U = np.where(confidence >= self.threshold)[0]

            if len(indices_U) == 0:
                return []

            confidence = confidence[indices_U]
            agreement = agreement[indices_U]

        if self.rescale_variance:
            n_ensemble = len(self.confidence_N)

            # print('(%.3f ~ %.3f) ----> ' % (np.min(agreement), np.max(agreement)), end='')

            for k in range(0, len(agreement)):
                total = confidence[k] * n_ensemble
                var = (int(total) + (total - int(total)) ** 2) / n_ensemble - confidence[k] ** 2
                agreement[k] /= var ** 0.5
                if agreement[k] != agreement[k]:
                    agreement[k] = 0

            # print('%.3f ~ %.3f' % (np.min(agreement), np.max(agreement)))

            if task != 'N':
                confidence -= self.threshold
                confidence /= (1 - self.threshold)

        confidence *= (self.n_bins - 1)
        agreement *= (self.n_bins - 1)

        for k in range(0, len(confidence)):
            if 0 <= confidence[k] <= self.n_bins - 1:
                bin_C = int(confidence[k] + 0.5)
            else:
                bin_C = random.randint(0, self.n_bins - 1)
            if 0 <= agreement[k] <= self.n_bins - 1:
                bin_A = int(agreement[k] + 0.5)
            else:
                bin_A = random.randint(0, self.n_bins - 1)
            histogram[bin_C][bin_A].append(k)

        n_non_empty = 0
        bin_non_empty = {}
        for i in range(0, self.n_bins):
            for j in range(0, self.n_bins):
                if len(histogram[i][j]) > 0:
                    bin_non_empty[n_non_empty] = histogram[i][j]
                    n_non_empty += 1

        for i in range(0, n_sample):
            if i > 0.5 * len(confidence):
                break
            need_break = False
            for _ in range(0, 100):
                if need_break:
                    break
                for j in random.sample(range(0, n_non_empty), n_non_empty):
                    index = bin_non_empty[j][random.randint(0, len(bin_non_empty[j]) - 1)]
                    if index not in indices:
                        indices.append(index)
                        need_break = True
                        break

        if task == 'N':
            return indices
        else:
            # print(len(indices_U[indices]))
            return indices_U[indices]

    def predict(self, X):
        label_pred = np.zeros(len(X), dtype=int)
        label_pred_proba = self.predict_proba(X)
        for i in range(0, len(label_pred)):
            if label_pred_proba[i, 1] >= 0.5:
                label_pred[i] = 1

        return label_pred

    def predict_proba(self, X, n_voters=None):
        if n_voters:
            self.n_voters = n_voters
        if self.n_voters < 0:
            self.n_voters = len(self.estimators)
        label_pred_proba = np.zeros((len(X), 2))
        for i in range(0, self.n_voters):
            label_pred_proba += self.estimators[i].predict_proba(X) / self.n_voters

        return label_pred_proba
