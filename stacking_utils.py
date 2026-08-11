"""
stacking_utils.py

Manual group-aware stacking ensemble. Implemented explicitly (rather than
relying on sklearn's StackingRegressor's internal cv=int shortcut) so the
cross-validation used to build the meta-learner's training data respects
duplicate-molecule groups -- sklearn's built-in cv=5 does NOT support
group-aware splitting out of the box.

Both train_tabpred_tuned.py and predict_from_smiles.py import this class,
since joblib/pickle needs the class definition importable to load a saved
model bundle.
"""

import numpy as np
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold


class GroupAwareStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators, meta_estimator, n_splits=5):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        n_models = len(self.base_estimators)
        oof_preds = np.zeros((len(y), n_models))

        if groups is None:
            groups = np.arange(len(y))

        gkf = GroupKFold(n_splits=self.n_splits)
        for tr_idx, ho_idx in gkf.split(X, y, groups=groups):
            for m_i, (name, est) in enumerate(self.base_estimators):
                model = clone(est)
                model.fit(X[tr_idx], y[tr_idx])
                oof_preds[ho_idx, m_i] = model.predict(X[ho_idx])

        self.meta_estimator_ = clone(self.meta_estimator)
        self.meta_estimator_.fit(oof_preds, y)

        self.fitted_base_estimators_ = []
        for name, est in self.base_estimators:
            model = clone(est)
            model.fit(X, y)
            self.fitted_base_estimators_.append((name, model))

        return self

    def _base_predictions(self, X):
        X = np.asarray(X)
        return np.column_stack([
            model.predict(X) for _, model in self.fitted_base_estimators_
        ])

    def predict(self, X):
        base_preds = self._base_predictions(X)
        return self.meta_estimator_.predict(base_preds)
