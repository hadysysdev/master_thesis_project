from logging import warning
import os
from pathlib import Path
from typing import Iterable, Callable, Any, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from timeeval import DatasetManager, Datasets, Status
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from prts import ts_precision, ts_recall, ts_fscore


def aggregate(df, aggregated: bool = True, short: bool = True):
        # metric_names: List[str] = ['ROC_AUC', 'PR_AUC', 'RANGE_PR_AUC'],
        keysNames = ["algo_training_type" ,"status", "ROC_AUC", "PR_AUC",
                 "RANGE_PR_AUC", "train_main_time", "execute_main_time"]
        if np.any(np.isin(df.status.unique(), [Status.ERROR, Status.TIMEOUT, Status.OOM])):
            warnings.warn("The results contain errors which are filtered out for the final aggregation. "
                             "To see all results, call .get_results(aggregated=False)")
            df = df[df.status == Status.OK]

        if short:
            # time_names = ["train_main_time", "execute_main_time"]
            group_names = ["algorithm", "collection", "hyper_params_id",
                           "dataset", "status", "algo_training_type", "dataset_input_dimensionality"]
        else:
            # time_names = Times.result_keys()
            group_names = ["algorithm", "collection", "dataset", "hyper_params_id"]
        keys = [key for key in keysNames if key in df.columns]
        grouped_results = df.groupby(group_names)
        results: pd.DataFrame = grouped_results[keys].mean()

        if short:
            results = results.rename(columns=dict([(k, f"{k}_mean") for k in keys]))
        else:
            std_results = grouped_results.std()[keys]
            results = results.join(std_results, lsuffix="_mean", rsuffix="_std")
        results["repetitions"] = grouped_results["repetition"].count()
        return results


def load_results(results_path: Path) -> pd.DataFrame:
    res = pd.read_csv(results_path / "results.csv")
    train_time_column_name= "execute_main_time"
    execution_time_column_name= "execute_main_time"
    if not res["algorithm"].is_unique:
         res = aggregate(res)
         res = res.reset_index()
         train_time_column_name = "train_main_time_mean"
         execution_time_column_name = "execute_main_time_mean"


    res["dataset_name"] = res["dataset"].str.split(".").str[0]
    res["overall_time"] = res[train_time_column_name].fillna(
        0) + res[execution_time_column_name].fillna(0)
    res["algorithm-index"] = res.algorithm + "-" + res.index.astype(str)
    res = res.drop_duplicates()
    return res


def load_scores_df(df: pd.DataFrame, result_path, algorithm_name, dataset_id, repetition=1):
    params_id = df.loc[(df["algorithm"] == algorithm_name) & (df["collection"] == dataset_id[0]) & (
        df["dataset"] == dataset_id[1]), "hyper_params_id"].item()
    path = (
        result_path /
        algorithm_name /
        params_id /
        dataset_id[0] /
        dataset_id[1] /
        str(repetition) /
        "anomaly_scores.ts"
    )
    return pd.read_csv(path, header=None)


def plot_scores(algorithm_name, collection_name: str, dataset_name: str, df: pd.DataFrame, dmgr: DatasetManager, result_path, **kwargs) -> go.Figure:
    if not isinstance(algorithm_name, list):
        algorithms = [algorithm_name]
    else:
        algorithms = algorithm_name
    # construct dataset ID
    if collection_name == "GutenTAG" and not dataset_name.endswith("supervised"):
        dataset_id = (collection_name, f"{dataset_name}.unsupervised")
    else:
        dataset_id = (collection_name, dataset_name)
    # load dataset details
    df_dataset = dmgr.get_dataset_df(dataset_id)

    # check if dataset is multivariate
    dataset_dim = df.loc[(df["collection"] == collection_name) & (
        df["dataset_name"] == dataset_name), "dataset_input_dimensionality"].unique().item()
    dataset_dim = dataset_dim.lower()

    auroc = {}
    aupr = {}
    aurange_pr = {}
    df_scores = pd.DataFrame(index=df_dataset.index)
    skip_algos = []
    algos = []
    for algo in algorithms:
        algos.append(algo)
        # get algorithm metrics results
        try:
            auroc[algo] = df.loc[(df["algorithm"] == algo) & (df["collection"] == collection_name) & (
                df["dataset_name"] == dataset_name), "ROC_AUC"].item()
        except ValueError:
            warning(
                f"No ROC_AUC/PR_AUC/RANGE_PR_AUC score found! Probably {algo} was not executed on {dataset_name}.")
            auroc[algo] = -1
            skip_algos.append(algo)
            continue
        try:
            aupr[algo] = df.loc[(df["algorithm"] == algo) & (df["collection"] == collection_name) & (
                df["dataset_name"] == dataset_name), "PR_AUC"].item()
        except ValueError:
            warning(
                f"No PR_AUC score found! Probably {algo} was not executed on {dataset_name}.")
            aupr[algo] = -1
            skip_algos.append(algo)
            continue
        try:
            aurange_pr[algo] = df.loc[(df["algorithm"] == algo) & (df["collection"] == collection_name) & (
                df["dataset_name"] == dataset_name), "RANGE_PR_AUC"].item()
        except ValueError:
            warning(
                f"No RANGE_PR_AUC score found! Probably {algo} was not executed on {dataset_name}.")
            aurange_pr[algo] = -1
            skip_algos.append(algo)
            continue
        # load scores
        training_type = df.loc[df["algorithm"] == algo,
                               "algo_training_type"].values[0].lower().replace("_", "-")
        try:
            df_scores[algo] = load_scores_df(
                df, result_path, algo, dataset_id).iloc[:, 0]
        except (ValueError, FileNotFoundError):
            warning(
                f"No anomaly scores found! Probably {algo} was not executed on {dataset_name}.")
            df_scores[algo] = np.nan
            skip_algos.append(algo)
    algorithms = [a for a in algos if a not in skip_algos]

    fig = plot_scores_plotly(algorithms, auroc, aupr, aurange_pr, df_scores,
                             df_dataset, dataset_dim, dataset_name)
    return fig


def plot_scores_plotly(algorithms, auroc, aupr, aurange_pr, df_scores, df_dataset, dataset_dim, dataset_name, **kwargs) -> go.Figure:
    # Create plot
    fig = make_subplots(3, 1)
    if dataset_dim == "multivariate":
        colors = px.colors.qualitative.Vivid
        for i in range(1, df_dataset.shape[1] - 1):
            colorIndex = i % len(colors)
            fig.add_trace(go.Scatter(x=df_dataset.index,
                          y=df_dataset.iloc[:, i], name=df_dataset.columns[i], marker_color=colors[colorIndex]), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=df_dataset.index,
                      y=df_dataset.iloc[:, 1], name="timeseries"), 1, 1)
    fig.add_trace(go.Scatter(x=df_dataset.index,
                  y=df_dataset["is_anomaly"], name="label", marker_color="red"), 2, 1)

    for algo in algorithms:
        fig.add_trace(go.Scatter(
            x=df_scores.index, y=df_scores[algo], name=f"{algo} anomaly scores "), 3, 1)
    fig.update_xaxes(matches="x")
    fig.update_layout(
        title=f"Results of {','.join(np.unique(algorithms))} on {dataset_name} ROC_AUC{auroc[algo]:.4f}: PR_AUC{aupr[algo]:.4f}: RANGE_PR_AUC{aurange_pr[algo]:.4f}",
        height=400
    )
    return fig


def plot_barplot(df, n_show: Optional[int] = None, title="Bar plots", ax_label="data", ay_label="Values", metric="ROC_AUC", _fmt_label=lambda x: x, log: bool = False) -> go.Figure:
    labels = dict()
    labels["algorithm"] = "Algorithm"
    labels[metric] = "Metric Score"
    color_discrete_sequence = ['#ec7c34']*len(df)
    fig = px.bar(df, x='algorithm', y=metric,
                 color='algorithm',
                 text_auto=True,
                 labels=labels
                 #  color_discrete_sequence=color_discrete_sequence,
                 )
    fig.update_layout(
        title={"text": title, "xanchor": "center", "x": 0.5},
        xaxis_title=ax_label,
        yaxis_title=ay_label,
        legend_title="Algorithms"

    )
    if log:
        fig.update_xaxes(type="log")
    return fig


def range_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    _max_samples: int = 50
    _r_alpha: float = 0.5
    _p_alpha: float = 0
    _cardinality = "one"
    _bias = "flat"

    thresholds = np.unique(y_score)
    thresholds.sort()
    # The first precision and recall values are precision=class balance and recall=1.0, which corresponds to a
    # classifier that always predicts the positive class, independently of the threshold. This means that we can
    # skip the first threshold!
    p0 = y_true.sum() / len(y_true)
    r0 = 1.0
    thresholds = thresholds[1:]

    # sample thresholds
    n_thresholds = thresholds.shape[0]
    if n_thresholds > _max_samples:
        every_nth = n_thresholds // (_max_samples - 1)
        sampled_thresholds = thresholds[::every_nth]
        if thresholds[-1] == sampled_thresholds[-1]:
            thresholds = sampled_thresholds
        else:
            thresholds = np.r_[sampled_thresholds, thresholds[-1]]

    recalls = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(np.int64)
        recalls[i] = ts_recall(y_true, y_pred,
                               alpha=_r_alpha,
                               cardinality=_cardinality,
                               bias=_bias)
        precisions[i] = ts_precision(y_true, y_pred,
                                     alpha=_p_alpha,
                                     cardinality=_cardinality,
                                     bias=_bias)
    # first sort by recall, then by precision to break ties (important for noisy scorings)
    sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]
    return np.r_[p0, precisions[sorted_idx], 1], np.r_[r0, recalls[sorted_idx], 0], thresholds


def auc_plot(
    algorithm_name: str,
    y_true: np.ndarray,
    y_score: Iterable[float],
    curve_function: Callable[[np.ndarray, np.ndarray], Any],
        store_plot=False, store_plot_dir: str = '', file_extention: str = 'pdf') -> float:
    x, y, thresholds = curve_function(y_true, np.array(y_score))
    if "precision_recall" in curve_function.__name__:
        # swap x and y
        x, y = y, x
    area: float = auc(x, y)
    name = curve_function.__name__
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines',
                             name=name))
    fig.update_layout(
        title={"text": f"{name} | area = {area:.4f}",
               "xanchor": "center", "x": 0.5})
    if store_plot:
        fig.write_image()
    # if self._plot:
    #     import matplotlib.pyplot as plt

    #     name = curve_function.__name__
    #     plt.plot(x, y, label=name, drawstyle="steps-post")
    #     # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    #     plt.title(f"{name} | area = {area:.4f}")
    #     if self._plot_store:
    #         plt.savefig(f"fig-{name}.pdf")
    #     plt.show()
    return area, fig


def auc_plot_matplotlib(
        algorithm_name: str,
        y_true: np.ndarray,
        y_score: Iterable[float],
        curve_function: Callable[[np.ndarray, np.ndarray], Any],
        plot=False, store_plot=False, store_plot_dir: str = '', file_extention: str = 'pdf') -> float:
    x, y, thresholds = curve_function(y_true, np.array(y_score))
    if "precision_recall" in curve_function.__name__:
        # swap x and y
        x, y = y, x
    area: float = auc(x, y)
    if plot:
        import matplotlib.pyplot as plt
        name = f'{algorithm_name}  {curve_function.__name__}'
        plt.plot(x, y, label=name, drawstyle="steps-post")
        # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.title(f"{name} | area = {area:.4f}")
        if store_plot:
            plt.savefig(f"{store_plot_dir}/fig-{name}.{file_extention}")
        plt.show()
    return area


def curve_store(y_true: np.ndarray,
                y_score: Iterable[float],
                curve_function: Callable[[np.ndarray, np.ndarray], Any]):

    x, y, thresholds = curve_function(y_true, np.array(y_score))
    if "precision_recall" in curve_function.__name__:
        # swap x and y
        x, y = y, x
    area: float = auc(x, y)
    return x, y, area
