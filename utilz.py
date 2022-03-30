import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.preprocessing import Binarizer


from google.colab import output  # sound


def done_sound():
    output.eval_js(
        'new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()'
    )


def show_df_stat(df):
    df_stat = df
    df_stat = df.describe(include="all", datetime_is_numeric=True)
    df_stat.loc["dtype"] = df.dtypes
    df_stat.loc["size"] = len(df)
    df_stat.loc["null count"] = df.isnull().sum()
    df_stat.loc["null count(%)"] = df.isnull().mean()
    return df_stat


def freq_encode_full(df1, df2, col, normalize=True):
    """
    Encode
    https://www.kaggle.com/cdeotte/high-scoring-lgbm-malware-0-702-0-775
    """
    df = pd.concat([df1[col], df2[col]])
    freq_dict = df.value_counts(dropna=False, normalize=normalize).to_dict()
    col_name = col + "_freq_enc_full"
    return col_name, freq_dict


import matplotlib.pyplot as plt
import seaborn as sns


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print("confusion matrix")
    print(confusion)
    print(
        f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1: .4f}, AUC: {roc_auc: .4f}"
    )


def get_clf_eval_heatmap(
    y_test, pred=None, pred_proba=None, ax=None, custom_threshold=None
):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    sns.heatmap(confusion, annot=True, cbar=None, cmap="Blues", ax=ax, fmt="g")
    ax.set_title(f"confusion matrix. threshold:{custom_threshold}")
    ax.set_ylabel("True Class"), ax.set_xlabel("Predicted Class")
    print(
        f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1: .4f}, AUC: {roc_auc: .4f}"
    )


def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임곗값:", custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)


def get_eval_by_threshold_heatmap(y_test, pred_proba_c1, thresholds, axs):
    for custom_threshold, ax in zip(thresholds, axs.flat):
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임곗값:", custom_threshold)
        get_clf_eval_heatmap(
            y_test, custom_predict, pred_proba_c1, ax, custom_threshold
        )


def roc_curve_plot(y_test, pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    plt.plot(fprs, tprs, label="ROC")
    # 가운데 대각선 직선을 그림.
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FPR( 1 - Sensitivity )")
    plt.ylabel("TPR( Recall )")
    plt.legend()
    plt.show()


def precision_recall_curve_plot(y_test, pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출.
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(
        thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision"
    )
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel("Threshold value")
    plt.ylabel("Precision and Recall value")
    plt.legend()
    plt.grid()
    plt.show()


def reduce_mem_usage(df):
    # iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df
