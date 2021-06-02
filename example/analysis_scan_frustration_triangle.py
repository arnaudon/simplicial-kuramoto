import itertools
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

import pickle


def plot_stationarity(path, filename):
    """Plot mean of variance of second half of simulation to see stationary state."""
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    var = np.empty([len(alpha1), len(alpha2)])
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        result = results[i][0].y
        var[idx_a1, idx_a2] = np.mean(np.var(result[:, : -int(np.shape(result)[1] / 2)], axis=1))
    plt.figure()
    plt.imshow(var, origin="lower", extent=(alpha2[0], alpha2[-1], alpha1[0], alpha1[-1]), vmin=0)
    plt.colorbar()
    plt.savefig(filename, bbox_inches="tight")


def rec_plot(s, eps=0.1, steps=10):
    """Compute recurence plot.

    Adapted from: https://github.com/laszukdawid/recurrence-plot/blob/master/plot_recurrence.py
    """
    return distance.squareform(np.clip(np.floor_divide(distance.pdist(s.T), eps), 0, steps))


def rec(X, i):
    # to install this on linux: pip install pocl-binary-distribution
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    time_series = TimeSeries(X.T, embedding_dimension=2, time_delay=2)
    settings = Settings(
        time_series,
        analysis_type=Classic,
        neighbourhood=FixedRadius(0.65),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    # print(result.recurrence_rate)
    # result.min_diagonal_line_length = 2
    # result.min_vertical_line_length = 2
    # result.min_white_vertical_line_length = 2
    # from pyrqa.computation import RPComputation
    # from pyrqa.image_generator import ImageGenerator

    # computation = RPComputation.create(settings)
    # result = computation.run()
    # ImageGenerator.save_recurrence_plot(
    #    result.recurrence_matrix_reverse, f"rec/recurrence_plot_{i}.png"
    # )
    #from pyrpde import rpde

    #res = rpde(X.T.astype(np.float32) / np.pi, dim=2, tau=2, epsilon=0.01, parallel=True)
    print(result)
    return result.recurrence_rate


def plot_recurences(path, filename):
    """Plot grid of recurence plots."""
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
    print(len(alpha1) * len(alpha2))
    axs = np.flip(axs, axis=0)
    rr = np.empty([len(alpha1), len(alpha2)])
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        result = results[i][0]
        rr[idx_a1, idx_a2] = rec(result.y[:, int(np.shape(result.y)[1] / 4):], i)
        plt.imshow(
            rec_plot(np.round(result.y + np.pi, 2) % (2 * np.pi) - np.pi),
            origin="lower",
            aspect="auto",
            cmap="Blues_r",
            interpolation="nearest",
            extent=(result.t[0], result.t[-1], 0, len(result.y)),
            vmin=0,
            # vmax=10,
        )
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for idx_a1 in range(len(alpha1)):
        axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
    for idx_a2 in range(len(alpha2)):
        axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

    fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

    plt.figure()
    plt.imshow(np.log10(rr), origin="lower", extent=(alpha2[0], alpha2[-1], alpha1[0], alpha1[-1]))#, vmin=0)
    plt.colorbar()
    plt.savefig("RR.pdf", bbox_inches="tight")


def plot_phases(path, filename):
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
    axs = np.flip(axs, axis=0)
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        result = results[i][0]

        plt.imshow(
            np.round(result.y + np.pi, 2) % (2 * np.pi) - np.pi,
            origin="lower",
            aspect="auto",
            cmap="twilight_shifted",
            interpolation="nearest",
            extent=(result.t[0], result.t[-1], 0, len(result.y)),
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for idx_a1 in range(len(alpha1)):
        axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
    for idx_a2 in range(len(alpha2)):
        axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

    fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    folder = "./results/"

    path = folder + "2_faces_common_edge.pkl"
    filename = "phase_triangle_face_flip_recurence.pdf"
    plot_recurences(path, filename)

def kjlkj():

    path = folder + "triangle_face_flip.pkl"
    filename = "phase_triangle_face_flip.pdf"
    # plot_phases(path, filename)
    filename = "phase_triangle_face_flip_var.pdf"
    # plot_stationarity(path, filename)
    filename = "phase_triangle_face_flip_recurence.pdf"
    #plot_recurences(path, filename)

    path = folder + "triangle_noface_flip.pkl"
    filename = "phase_triangle_noface_flip.pdf"
    #plot_phases(path, filename)
    filename = "phase_triangle_noface_flip_var.pdf"
    #plot_stationarity(path, filename)
    filename = "phase_triangle_noface_flip_recurence.pdf"
    #plot_recurences(path, filename)

    path = folder + "triangle_face.pkl"
    filename = "phase_triangle_face.pdf"
    #plot_phases(path, filename)
    filename = "phase_triangle_face_var.pdf"
    #plot_stationarity(path, filename)
    filename = "phase_triangle_face_recurence.pdf"
    plot_recurences(path, filename)

    path = folder + "triangle_noface.pkl"
    filename = "phase_triangle_noface.pdf"
    #plot_phases(path, filename)
    filename = "phase_triangle_noface_var.pdf"
    #plot_stationarity(path, filename)
    filename = "phase_triangle_noface_recurence.pdf"
    #plot_recurences(path, filename)
