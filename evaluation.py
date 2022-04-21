from sklearn.metrics import f1_score, mean_squared_error, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eval_clf(y_test, y_pred):
    if isinstance(y_test, (pd.core.frame.DataFrame, pd.core.series.Series)):
        y_test = y_test.astype(str)
    if isinstance(y_pred, (pd.core.frame.DataFrame, pd.core.series.Series)):
        y_pred = y_pred.astype(str)
    clf_report = classification_report(y_test,
                                       y_pred, )

    print(clf_report)

    test_labels = set(np.unique(y_test))
    pred_labels = set(np.unique(y_pred))
    labels = test_labels.union(pred_labels)

    conf_matrix = confusion_matrix(y_test,
                                   y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.grid(False)
    disp.plot(ax=ax)

    return round(f1_score(y_test, y_pred, average='micro'), 2)


def half_round(inp):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""
    if isinstance(inp, (pd.core.frame.DataFrame, pd.core.series.Series, int)):
        return round(inp * 2) / 2
    else:
        print('type is not half rounded', type(inp), inp)


def eval_regr(y_test, y_pred):
    raw_rmse = round(mean_squared_error(y_test, y_pred, squared=True), 2)

    y_test = half_round(y_test)
    y_pred = half_round(y_pred)
    rounded_rmse = round(mean_squared_error(y_test, y_pred, squared=True), 2)

    if rounded_rmse == raw_rmse:
        print('probably, already rounded values were passed')

    return rounded_rmse, raw_rmse


def eval_all(y_test, y_pred, model_name: str):
    f1 = eval_clf(half_round(y_test), half_round(y_pred))
    rounded_rmse, raw_rmse = eval_regr(y_test, y_pred)
    return {model_name: {'f1': f1, 'rounded_rmse': rounded_rmse, 'rmse': raw_rmse}}
