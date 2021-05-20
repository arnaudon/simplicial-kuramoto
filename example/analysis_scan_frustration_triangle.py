import numpy as np
import itertools
import matplotlib.pyplot as plt

import pickle


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
            vmax=np.pi
        )
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for idx_a1 in range(len(alpha1)):
        axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
    for idx_a2 in range(len(alpha2)):
        axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

    fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    folder = "./results/"

    path = folder + "triangle_face_flip.pkl"
    filename = "phase_triangle_face_flip.pdf"
    plot_phases(path, filename)

    path = folder + "triangle_noface_flip.pkl"
    filename = "phase_triangle_noface_flip.pdf"
    plot_phases(path, filename)

    path = folder + "triangle_face.pkl"
    filename = "phase_triangle_face.pdf"
    plot_phases(path, filename)

    path = folder + "triangle_noface.pkl"
    filename = "phase_triangle_noface.pdf"
    plot_phases(path, filename)
