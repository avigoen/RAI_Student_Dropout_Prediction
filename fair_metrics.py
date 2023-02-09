from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def statistical_parity_test(data, protected_group, adv_protect_group_value, disadv_protect_group_value, target, fav_outcome):
    """
    Statistical Parity Test to measure the difference between advantageous & disadvantageous groups of a protected variable getting a favourable outcome

    Parameters
    ----------

    :param data: data
    :param protected_group: string name of the protected group/sensitive feature
    :param adv_protect_group_value: binary value indicating membership to the advantageous protected group
    :param disadv_protect_group_value: binary value indicating membership to the disadvantageous protected group
    :param target: string target
    :param fav_outcome: binary value that indicates favourable outcome of target
    :return: statistical_parity, disparate_impact


    Examples
    --------
    statistical_parity, disparate_impact = statistical_parity_test(data, protected_group,
              adv_protected_group_value, disadv_protected_group_value, target, fav_outcome)

    """

    adv_group = data[data[protected_group] == adv_protect_group_value]
    favoured_adv = adv_group[adv_group[target] == fav_outcome]
    favoured_adv_count = len(favoured_adv)

    disadv_group = data[data[protected_group] == disadv_protect_group_value]
    favoured_disadv = disadv_group[disadv_group[target] == fav_outcome]
    favoured_disadv_count = len(favoured_disadv)

    total_adv = len(adv_group)
    total_disadv = len(disadv_group)

    statistical_parity = (favoured_disadv_count /
                          total_disadv) - (favoured_adv_count / total_adv)
    disparate_impact = (favoured_disadv_count / total_disadv) / \
        (favoured_adv_count / total_adv)

    return statistical_parity, disparate_impact


def fair_metrics(y_actual, y_pred_prob, y_pred_binary, X_test, protected_group_name,
                 adv_val, disadv_val):
    """
    Fairness performance metrics for a model to compare advantageous and disadvantageous groups of a protected variable

    Parameters
    ----------

    :param y_actual: Actual binary outcome
    :param y_pred_prob: predicted probabilities
    :param y_pred_binary: predicted binary outcome
    :param X_test: Xtest data
    :param protected_group_name: Sensitive feature
    :param adv_val: Priviliged value of protected label
    :param disadv_val: Unpriviliged value of protected label
    :return: roc, avg precision, Eq of Opportunity, Equalised Odds, Precision/Predictive Parity, Demographic Parity, Avg Odds Diff,
            Predictive Equality, Treatment Equality

    Examples
    --------
    fairness_metrics=[fair_metrics(y_test, y_pred_prob, y_pred,
                     X_test, choice, adv_val, disadv_val)]


    """
    tn_adv, fp_adv, fn_adv, tp_adv = confusion_matrix(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_binary[X_test[protected_group_name] == adv_val]).ravel()

    tn_disadv, fp_disadv, fn_disadv, tp_disadv = confusion_matrix(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_binary[X_test[protected_group_name] == disadv_val]).ravel()

    # Receiver operating characteristic
    roc_adv = roc_auc_score(y_actual[X_test[protected_group_name] == adv_val],
                            y_pred_prob[X_test[protected_group_name] == adv_val])
    roc_disadv = roc_auc_score(y_actual[X_test[protected_group_name] == disadv_val],
                               y_pred_prob[X_test[protected_group_name] == disadv_val])

    roc_diff = abs(roc_disadv - roc_adv)

    # Average precision score
    ps_adv = average_precision_score(
        y_actual[X_test[protected_group_name] == adv_val],
        y_pred_prob[X_test[protected_group_name] == adv_val])
    ps_disadv = average_precision_score(
        y_actual[X_test[protected_group_name] == disadv_val],
        y_pred_prob[X_test[protected_group_name] == disadv_val])

    ps_diff = abs(ps_disadv - ps_adv)

    # Equal Opportunity - advantageous and disadvantageous groups have equal FNR
    FNR_adv = fn_adv / (fn_adv + tp_adv)
    FNR_disadv = fn_disadv / (fn_disadv + tp_disadv)
    EOpp_diff = abs(FNR_disadv - FNR_adv)

    # Predictive equality  - advantageous and disadvantageous groups have equal FPR
    FPR_adv = fp_adv / (fp_adv + tn_adv)
    FPR_disadv = fp_disadv / (fp_disadv + tn_disadv)
    pred_eq_diff = abs(FPR_disadv - FPR_adv)

    # Equalised Odds - advantageous and disadvantageous groups have equal TPR + FPR
    TPR_adv = tp_adv / (tp_adv + fn_adv)
    TPR_disadv = tp_disadv / (tp_disadv + fn_disadv)
    EOdds_diff = abs((TPR_disadv + FPR_disadv) - (TPR_adv + FPR_adv))

    # Predictive Parity - advantageous and disadvantageous groups have equal PPV/Precision (TP/TP+FP)
    prec_adv = (tp_adv)/(tp_adv + fp_adv)
    prec_disadv = (tp_disadv)/(tp_disadv + fp_disadv)
    prec_diff = abs(prec_disadv - prec_adv)

    # Demographic Parity - ratio of (instances with favourable prediction) / (total instances)
    demo_parity_adv = (tp_adv + fp_adv) / (tn_adv + fp_adv + fn_adv + tp_adv)
    demo_parity_disadv = (tp_disadv + fp_disadv) / \
        (tn_disadv + fp_disadv + fn_disadv + tp_disadv)
    demo_parity_diff = abs(demo_parity_disadv - demo_parity_adv)

    # Average of Difference in FPR and TPR for advantageous and disadvantageous groups
    AOD = 0.5*((FPR_disadv - FPR_adv) + (TPR_disadv - TPR_adv))

    # Treatment Equality  - advantageous and disadvantageous groups have equal ratio of FN/FP
    TE_adv = fn_adv/fp_adv
    TE_disadv = fn_disadv/fp_disadv
    TE_diff = abs(TE_disadv - TE_adv)

    return [('AUC', roc_diff), ('Avg PrecScore', ps_diff), ('Equal Opps', EOpp_diff),
            ('PredEq', pred_eq_diff), ('Equal Odds',
                                       EOdds_diff), ('PredParity', prec_diff),
            ('DemoParity', demo_parity_diff), ('AOD', abs(AOD)), ('TEq', TE_diff)]

def acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv):
    """
    Fairness performance metrics for a additive counterfactually fair model to compare advantageous and
    disadvantageous groups of a protected variable

    :param tn_disadv: disadvantaged group's true negative
    :param fp_disadv: disadvantaged group's false positive
    :param fn_disadv: disadvantaged group's false negative
    :param tp_disadv: disadvantaged group's true positive
    :param tn_adv: advantaged group's true negative
    :param fp_adv: advantaged group's false positive
    :param fn_adv: advantaged group's false negative
    :param tp_adv: advantaged group's true positive
    :return: Equal Opportunity, Predictive Equality, Equalised Odds, Precision/Predictive Parity, Demographic Parity,
        Avg Odds Diff, Treatment Equality

    Examples
    --------
    acf_metrics=acf_fair_metrics(tn_disadv, fp_disadv, fn_disadv, tp_disadv, tn_adv, fp_adv, fn_adv, tp_adv)
    """

    # Equal Opportunity - advantageous and disadvantageous groups have equal FNR
    FNR_adv = fn_adv / (fn_adv + tp_adv)
    FNR_disadv = fn_disadv / (fn_disadv + tp_disadv)
    EOpp_diff = abs(FNR_disadv - FNR_adv)

    # Predictive equality  - advantageous and disadvantageous groups have equal FPR
    FPR_adv = fp_adv / (fp_adv + tn_adv)
    FPR_disadv = fp_disadv / (fp_disadv + tn_disadv)
    pred_eq_diff = abs(FPR_disadv - FPR_adv)

    # Equalised Odds - advantageous and disadvantageous groups have equal TPR + FPR
    TPR_adv = tp_adv / (tp_adv + fn_adv)
    TPR_disadv = tp_disadv / (tp_disadv + fn_disadv)
    EOdds_diff = abs((TPR_disadv + FPR_disadv) - (TPR_adv + FPR_adv))

    # Predictive Parity - advantageous and disadvantageous groups have equal PPV/Precision (TP/TP+FP)
    prec_adv = (tp_adv)/(tp_adv + fp_adv)
    prec_disadv = (tp_disadv)/(tp_disadv + fp_disadv)
    prec_diff = abs(prec_disadv - prec_adv)

    # Demographic Parity - ratio of (instances with favourable prediction) / (total instances)
    demo_parity_adv = (tp_adv + fp_adv) / (tn_adv + fp_adv + fn_adv + tp_adv)
    demo_parity_disadv = (tp_disadv + fp_disadv) / \
        (tn_disadv + fp_disadv + fn_disadv + tp_disadv)
    demo_parity_diff = abs(demo_parity_disadv - demo_parity_adv)

    # Average of Difference in FPR and TPR for advantageous and disadvantageous groups
    AOD = 0.5*((FPR_disadv - FPR_adv) + (TPR_disadv - TPR_adv))

    # Treatment Equality  - advantageous and disadvantageous groups have equal ratio of FN/FP
    TE_adv = fn_adv/fp_adv
    TE_disadv = fn_disadv/fp_disadv
    TE_diff = abs(TE_disadv - TE_adv)

    return [('Equal Opps', EOpp_diff),
            ('PredEq', pred_eq_diff), ('Equal Odds',
                                       EOdds_diff), ('PredParity', prec_diff),
            ('DemoParity', demo_parity_diff), ('AOD', abs(AOD)), ('TEq', TE_diff)]


def accuracy_metrics(y_actual, y_pred_prob_ww, y_pred_prob_wow, y_pred_binary_ww,
                     y_pred_binary_wow, X_test):
    """
        Model performance metrics to compare advantageous and disadvantageous groups of a protected variable

        Parameters
        ----------

        :param y_actual: Actual binary outcome
        :param y_pred_prob_ww: predicted probabilities with weights
        :param y_pred_prob_wow: predicted probabilities without weights
        :param y_pred_binary_ww: predicted binary outcome with weights
        :param y_pred_binary_wow: predicted binary outcome without weights
        :param X_test: Xtest data
        :return: AUC, Gini, Avg PrecScore, Precision, Recall, FNR, F1_score

        Examples
        --------
        accuracy_metrics=[accuracy_metrics(y_test, y_pred_prob_ww, y_pred_prob_wow, y_pred_binary_ww,
                    y_pred_binary_wow, X_test)]

    """

    tn_ww, fp_ww, fn_ww, tp_ww = confusion_matrix(
        y_actual, y_pred_binary_ww).ravel()  # y_true, y_pred
    tn_wow, fp_wow, fn_wow, tp_wow = confusion_matrix(
        y_actual, y_pred_binary_wow).ravel()

    # Receiver operating characteristic
    auroc_ww = roc_auc_score(y_actual, y_pred_prob_ww)
    auroc_wow = roc_auc_score(y_actual, y_pred_prob_wow)

    # Gini = 2*AUROC - 1
    gini_ww = 2*auroc_ww - 1
    gini_wow = 2*auroc_wow - 1

    # Average precision score
    ps_ww = average_precision_score(y_actual, y_pred_binary_ww)
    ps_wow = average_precision_score(y_actual, y_pred_binary_wow)

    # Precision (TP/TP+FP)
    prec_ww = (tp_ww)/(tp_ww + fp_ww)
    prec_wow = (tp_wow)/(tp_wow + fp_wow)

    # Recall - True Positive Rate = TP/TP+FN
    recall_ww = tp_ww/(tp_ww + fn_ww)
    recall_wow = tp_wow/(tp_wow + fn_wow)

    # False Negative Rate = FN/(FN+TP)
    FNR_ww = fn_ww/(fn_ww + tp_ww)
    FNR_wow = fn_wow/(fn_wow + tp_wow)

    F1_ww = (2*prec_ww*recall_ww)/(prec_ww + recall_ww)
    F1_wow = (2*prec_wow*recall_wow)/(prec_wow + recall_wow)

    return auroc_ww, gini_ww, ps_ww, prec_ww, recall_ww, FNR_ww, F1_ww, auroc_wow, gini_wow, ps_wow,  prec_wow, recall_wow, FNR_wow, F1_wow,
