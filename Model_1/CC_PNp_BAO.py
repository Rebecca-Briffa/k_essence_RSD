import emcee
import sys
from multiprocessing import Pool, cpu_count
from Likelihoods import *
from scipy.integrate import cumulative_trapezoid

ncpu = cpu_count()

# Reuse likelihood objects (avoid re-reading data every call)
_lkl_bao = Likelihood_DESI2()
_lkl_cc = Likelihood_clocks()
#_lkl_fs8 = Likelihood_fs8()
_lkl_pn = Likelihood_Pantheon_SH0ES()


def log_prior(theta):
    H_0, Om, B, M, r_drag = theta
    B_min = 1.0 / (16.0 - 16.0 * Om)
    B_max = (4.0 - 3.0 * Om) / (6.0 * (Om - 2.0) ** 2)

    if not (50.0 < H_0 < 100.0):
        return -np.inf
    if not (0.1 < Om < 0.9):
        return -np.inf
    if not (B_min < B < B_max):
        return -np.inf
    # if not (0.2 < s80 < 1.0):
    #     return -np.inf
    # if not (-1.0 < f0 < 1.0):
    #     return -np.inf
    if not (-80.0 < M < 60.0):
        return -np.inf
    if not (100.0 < r_drag < 300.0):
        return -np.inf
    return 0.0


def log_probability(theta):
    H_0, Om, B, M, r_drag = theta
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    z_int = np.linspace(0.0, 2.4, 800)

    Hz, dH_dz = H_dH_arrays(H_0, B, Om, z_int)
    if not np.all(np.isfinite(Hz)) or not np.all(np.isfinite(dH_dz)):
        return -np.inf

    hz_interp = interpolate.interp1d(
        z_int, Hz, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    dhz_interp = interpolate.interp1d(
        z_int, dH_dz, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )

    d_H = c / Hz
    d_C = cumulative_trapezoid(d_H, z_int, initial=0)

    # D_on_grid = diff_d_d0(z_int, H_0, B, Om, f0, hz_interp, dhz_interp)
    # if not np.all(np.isfinite(D_on_grid)):
    #     return -np.inf

    lp = (
        lp
        + _lkl_bao.loglkl(theta, z_int, Hz, d_C)
        + _lkl_cc.loglkl(theta, z_int, Hz)
        #+ _lkl_fs8.loglkl(theta, z_int, D_on_grid, Hz)
        + _lkl_pn.loglkl(theta, z_int, Hz, d_C)
    )

    if np.isnan(lp):
        return -np.inf
    return lp


def build_initial_walkers(nwalkers, eps=1e-4):
    """Draw walkers strictly inside the prior, then keep only finite likelihood."""
    p0 = []
    n_tries = 0
    while len(p0) < nwalkers:
        n_tries += 1
        H0 = np.random.uniform(50.0, 100.0)
        Om = np.random.uniform(0.1, 0.9)

        # Compute B bounds for this omegam
        B_min = 1.0 / (16.0 - 16 * Om)
        B_max = (4.0 - 3 * Om) / (6 * (Om - 2) ** 2)
        # Sample B safely
        if B_min >= B_max:
            continue  # skip pathological case
        B = np.random.uniform(B_min, B_max)

        s80 = np.random.uniform(0.2, 1.0)
        f0 = np.random.uniform(-1.0, 1.0)
        M = np.random.uniform(-80.0, 60.0)
        r_drag = np.random.uniform(100.0, 300.0)

        theta = [H0, Om, B,  M, r_drag]
        if log_prior(theta) != 0.0:
            continue
        if not np.isfinite(log_probability(theta)):
            continue

        p0.append(theta)


    return np.array(p0)


ndim, nwalkers = 5, 100
niter = 50000
p0 = build_initial_walkers(nwalkers)


if __name__ == '__main__':

    with Pool(6) as pool:
        moves = [emcee.moves.StretchMove(), emcee.moves.DEMove()]
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, moves=moves, pool=pool
        )

        print("\nRunning burn-in...")
        p0, lp0, _ = sampler.run_mcmc(p0, 100, progress=True)
        n_finite = np.sum(np.isfinite(lp0))
        print(f"Burn-in: {n_finite}/{nwalkers} walkers have finite log prob")
        sampler.reset()

        print("Running production...")
        index = 0
        autocorr = np.empty(niter)
        old_tau = np.inf

        for sample in sampler.sample(p0, iterations=niter, progress=True):
            if sampler.iteration % 100:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.001)
            print(
                "\n --> old_tau = {0}, new_tau = {1} not convergent!!".format(
                    old_tau, tau
                )
            )
            if converged:
                print(
                    "\n --> old_tau = {0}, new_tau = {1} convergent!!".format(
                        old_tau, tau
                    )
                )
                break
            old_tau = tau

        flat_samples = sampler.get_chain(flat=True)
        np.savetxt("CC_PN_BAO_chain_rd_2710.txt", flat_samples)

        log_prob = sampler.get_log_prob(flat=True)
        np.savetxt("CC_PN_BAO_log_prob_rd_2710.txt", log_prob)
