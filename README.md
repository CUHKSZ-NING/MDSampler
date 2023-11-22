# MDSampler

Code for manuscript `MDSampler: Meta-Distribution-Based Ensemble Sampler for Imbalanced Semi-Supervised Learning`. MDSampler is a unified framework that combines SSL, IL, and ensemble learning via iterative instance under-sampling and cascade classifier aggregation.

* Required Python 3 packages:
    1. `numpy==1.21.5`;
    2. `sklearn` (https://github.com/scikit-learn/scikit-learn);
    3. `scipy==1.7.3`.

* MDSampler is compatible with most sklearn APIs but is not strictly tested.

* Import: `from MDSampler import MDSampler`.

* Train: `fit(X, y)`, with target $\textbf{y}_i \in (-1, 0, 1)^l$ as the labels. 

* Predict: `predict(X)` (hard prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimator`: Classifier object with `predict_proba()` function;
    2. `feature_ranker`: Feature selection (ranking) object that gives weights to features in $\mathcal{X}$;
    3. `k_sup`: The number of warm-up training iterations;
    4. `k_semi`: The number of semi-supervised learning iterations;
    5. `w`: Controls the number of bins in MDSampler (i.e., $w^2$ bins in total);
    6. `adaptive_instance_interpolation`: When True, the adaptive instance interpolation module is employed.
