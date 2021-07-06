import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.frustration_scan import (
    scan_sigma_parameters,
    get_subspaces,
    compute_simplicial_order_parameter,
)

from scan import my_house


if __name__ == "__main__":
    sizes = [3, 4, 5, 6, 7]  # 4, 5, 6, 7, 8, 9, 10]
    alpha_2 = 0
    t_max = 1000
    n_workers = 80
    sigmas = np.logspace(-2.5, 0, 50)
    plt.figure(figsize=(5, 3))
    for size in sizes:
        Gsc = my_house(size)

        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
        alpha_1 = np.random.normal(harm_subspace.sum(1), 0.2, Gsc.n_edges)
        results = scan_sigma_parameters(
            Gsc,
            sigmas=sigmas,
            alpha1=alpha_1,
            alpha2=0,
            t_max=t_max,
            n_workers=n_workers
        )

        #plt.figure()
        order = []
        for i in range(len(results)):
            result = results[i][0]
            sigma = sigmas[i]
            global_order = compute_simplicial_order_parameter(result.y, harm_subspace)
            #plt.plot(result.t, global_order)
            order.append(np.mean(global_order[-10:]))
        #plt.xscale("log")
        #plt.savefig(f"sigma_orders_{size}.pdf")

        plt.plot(sigmas, order, "-+", label=f"size={size}")
        #plt.axis([sigmas[0], sigmas[-1], 0, 1])
        plt.xscale("log")
        plt.legend()
        plt.xlabel("sigma")
        plt.ylabel("global order")
        plt.savefig(f"sigma_vs_hop.pdf", bbox_inches='tight')
