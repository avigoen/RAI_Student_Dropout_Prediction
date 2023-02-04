import pandas as pd
from fair_metrics import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class ACF:
    def __init__(self, X, Y, sensitive_cols):
        self.sensitive_cols = sensitive_cols
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
        self.sens_train = self.X_train[sensitive_cols]
        self.sens_test = self.X_test[sensitive_cols]
        self.independent_var = self.X_train.drop(sensitive_cols, axis=1).columns.values

    def get_residual(self, x, xhat):
        return (x - xhat) ** 2

    def fit(self):
        model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
        model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
        self.y_pred_prob = model.predict_proba(self.X_test)[:,0]

        residuals_train = {}
        self.residual_train_models = {}
        residuals_test = {}
        self.residual_test_models = {}
        for ind_col in self.X_train.columns.values:
            sens_train_model = None
            if self.X_train[ind_col].unique().size == 2:
                sens_train_model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10)
            else:
                sens_train_model = LinearRegression()
            sens_train_model.fit(self.sens_train, self.X_train[ind_col])
            self.residual_train_models[ind_col] = sens_train_model
            residual_train = self.get_residual(self.X_train[ind_col], sens_train_model.predict(self.sens_train))
            residuals_train[ind_col] = residual_train


            sens_test_model = None
            if self.X_train[ind_col].unique().size == 2:
                sens_test_model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10)
            else:
                sens_test_model = LinearRegression()
            sens_test_model.fit(self.sens_test, self.X_test[ind_col])

            self.residual_test_models[ind_col] = sens_test_model
            residual_test = self.get_residual(self.X_test[ind_col], sens_test_model.predict(self.sens_test))
            residuals_test[ind_col] = residual_test
        
        residual_train_df = pd.DataFrame(residuals_train)
        residual_test_df = pd.DataFrame(residuals_test)
        self.fair_model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10).fit(residual_train_df, self.y_train)

        self.y_pred_fair = self.fair_model.predict(residual_test_df)
        self.y_pred_prob_fair = self.fair_model.predict_proba(residual_test_df)

    def score(self, choice, adv_class, disadv_class):
        tn_disadv, fp_disadv, fn_disadv, tp_disadv = confusion_matrix(self.y_test[self.sens_test[choice]==disadv_class], self.y_pred_fair[self.sens_test[choice]==disadv_class]).ravel()
        tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(self.y_test[self.sens_test[choice]==adv_class], self.y_pred_fair[self.sens_test[choice]==adv_class]).ravel()

        acf_metrics = acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv)

        logistic_reg_fair_metrics = fair_metrics(self.y_test, self.y_pred_prob, self.y_pred, self.X_test, choice, adv_class, disadv_class)

        return acf_metrics, logistic_reg_fair_metrics

    def flip_sensitive(self):
        self.sens_train_flipped = self.sens_train.applymap(lambda x: 1 if x == 0 else 0)
        self.sens_test_flipped = self.sens_test.applymap(lambda x: 1 if x == 0 else 0)


    def cuf(self):
        residuals_train = {}
        residuals_test = {}
        for ind_col in self.X_train.columns.values:
            sens_train_model = self.residual_train_models[ind_col]
            residual_train = self.X_train[ind_col] - sens_train_model.predict(self.sens_train_flipped)
            residuals_train[ind_col] = residual_train


            sens_test_model = self.residual_test_models[ind_col]
            residual_test = self.X_test[ind_col] - sens_test_model.predict(self.sens_test_flipped)
            residuals_test[ind_col] = residual_test
        
        residual_train_df = pd.DataFrame(residuals_train)
        residual_test_df = pd.DataFrame(residuals_test)

        self.cuf_model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10).fit(residual_train_df, self.y_train)

        self.y_pred_cuf = self.cuf_model.predict(residual_test_df)
        self.y_pred_prob_cuf = self.cuf_model.predict_proba(residual_test_df)

    def score_cuf(self, choice, adv_class, disadv_class):
        tn_disadv, fp_disadv, fn_disadv, tp_disadv = confusion_matrix(self.y_test[self.sens_test_flipped[choice]==disadv_class], self.y_pred_cuf[self.sens_test_flipped[choice]==disadv_class]).ravel()
        tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(self.y_test[self.sens_test_flipped[choice]==adv_class], self.y_pred_cuf[self.sens_test_flipped[choice]==adv_class]).ravel()

        acf_metrics = acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv)

        logistic_reg_fair_metrics = fair_metrics(self.y_test, self.y_pred_prob_cuf, self.y_pred_cuf, self.X_test, choice, adv_class, disadv_class)

        return acf_metrics, logistic_reg_fair_metrics