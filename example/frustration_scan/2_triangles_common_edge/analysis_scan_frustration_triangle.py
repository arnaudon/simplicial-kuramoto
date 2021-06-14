from simplicial_kuramoto.frustration_scan import (
    plot_phases,
    plot_projections,
    plot_recurences,
    plot_rqa,
)


if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    path = folder + "2_triangles_face_flip.pkl"
    filename = figure_folder + "2_triangles_face_flip.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "2_triangles_face_flip_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "2_triangles_face_flip_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "2_triangles_face_flip_rqa.pdf"
    # plot_rqa(path, filename, min_rr=0.2)

    path = folder + "2_triangles_face.pkl"
    filename = figure_folder + "2_triangles_face.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "2_triangles_face_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "2_triangles_face_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "2_triangles_face_rqa.pdf"
    # plot_rqa(path, filename, min_rr=0.2)

    path = folder + "2_triangles_face_flip_middle.pkl"
    filename = figure_folder + "2_triangles_face_flip_middle.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "2_triangles_face_flip_middle_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "2_triangles_face_flip_middle_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "2_triangles_face_flip_middle_rqa.pdf"
    # plot_rqa(path, filename, min_rr=0.2)

    path = folder + "2_triangles_face_one_face.pkl"
    filename = figure_folder + "2_triangles_face_one_face.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_rqa.pdf"
    # plot_rqa(path, filename, min_rr=0.2)

    path = folder + "2_triangles_face_one_face_flip.pkl"
    filename = figure_folder + "2_triangles_face_one_face_flip.pdf"
    # plot_phases(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_flip_proj.pdf"
    plot_projections(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_flip_recurence.pdf"
    # plot_recurences(path, filename)
    filename = figure_folder + "2_triangles_face_one_face_flip_rqa.pdf"
    # plot_rqa(path, filename, min_rr=0.2)
