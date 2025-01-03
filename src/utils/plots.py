import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scatter(x, y, title="", x_label="X", y_label="Y", plot_show=True, file_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if file_path is not None:
        plt.savefig(file_path)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return None

def plot_box(data, x, y, hue=None, title="", x_label="X", y_label="Y", plot_show=True, file_path=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if file_path is not None:
        plt.savefig(file_path)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return None


def plot_bar(data, x, y, hue=None, title="", x_label="X", y_label="Y", plot_show=True, file_path=None):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if file_path is not None:
        plt.savefig(file_path)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return None

def plot_AUROC(x=[], y=[], labels=[], title="", x_label="False positive rate", y_label="True positive rate", plot_show=True, file_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--")
    for i, label_i in enumerate(labels):
        plt.plot(x[i], y[i], label=label_i)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best", fontsize=6)

    if file_path is not None:
        plt.savefig(file_path)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return None

def plot_AUPRC(x=[], y=[], labels=[], title="", x_label="Recall", y_label="Precision", plot_show=True, file_path=None):
    plt.figure(figsize=(8, 6))
    for i, label_i in enumerate(labels):
        plt.plot(x[i], y[i], label=label_i)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best", fontsize=6)

    if file_path is not None:
        plt.savefig(file_path)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return None


