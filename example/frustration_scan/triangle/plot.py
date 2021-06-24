from simplicial_kuramoto.frustration_scan import (
    plot_phases,
    plot_recurences,
    plot_rqa,
    plot_projections,
    plot_harmonic_order
)


if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    path = folder + "triangle_face_flip.pkl"
    filename = figure_folder + "phase_triangle_face_flip.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "phase_triangle_face_flip_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "phase_triangle_face_flip_order.pdf"
    plot_harmonic_order(path, filename)
    filename = figure_folder + "phase_triangle_face_flip_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "phase_triangle_face_flip_rqa.pdf"
    print('rqa')
    #plot_rqa(path, filename)

def llkj():
    path = folder + "triangle_noface_flip.pkl"
    filename = figure_folder + "phase_triangle_noface_flip.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "phase_triangle_noface_flip_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "phase_triangle_noface_flip_rqa.pdf"
    plot_rqa(path, filename)

    path = folder + "triangle_face.pkl"
    filename = figure_folder + "phase_triangle_face.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "phase_triangle_face_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "phase_triangle_face_rqa.pdf"
    plot_rqa(path, filename)

    path = folder + "triangle_noface.pkl"
    filename = figure_folder + "phase_triangle_noface.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "phase_triangle_noface_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "phase_triangle_noface_rqa.pdf"
    plot_rqa(path, filename)
