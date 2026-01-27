from .QSR_solvers import *
from tqdm import tqdm
from .QSR_dct import *
from .computation_functions import *
from .use_cupy_dtype import *
from joblib import Parallel, delayed

xp_name = getattr(globals().get('xp', None), '__name__', '')
on_gpu = (xp_name == 'cupy')
backend   = 'threading'
#backend = 'loky'

def scft_iteration(Scft_params, wsr, qsr_initial, qsr_d_initial):

    if Scft_params.D3 == False: #2D

        if Scft_params.close_boundary == True: # DCT
            '''
            qsr, qsr_d_reverse = Parallel(n_jobs=2, backend=backend)(
                [
                    delayed(solve_qsr_2d_neumann)(Scft_params, wsr, qsr_initial, reverse = False),
                    delayed(solve_qsr_2d_neumann)(Scft_params, wsr, qsr_d_initial, reverse = True),
                ]
            )
            '''
            qsr = solve_qsr_2d_fft_even_expend(Scft_params, wsr, qsr_initial, reverse = False)
            qsr_d_reverse = solve_qsr_2d_fft_even_expend(Scft_params, wsr, qsr_d_initial, reverse = True)

        else: # FFT
            qsr = solve_qsr_2d(Scft_params, wsr, qsr_initial, reverse = False)
            qsr_d_reverse = solve_qsr_2d(Scft_params, wsr, qsr_d_initial, reverse = True)

        qsr_d = xp.flip(qsr_d_reverse, axis = 0)
        phi_blocks, phi_s, phi, Qc = get_phi(Scft_params, qsr, qsr_d)

    else: # 3D

        if Scft_params.close_boundary == True: # Neumann
            if use_cupy == True:
                qsr = solve_qsr_3d_neumann(Scft_params, wsr, qsr_initial, reverse = False)
                qsr_d_reverse = solve_qsr_3d_neumann(Scft_params, wsr, qsr_d_initial, reverse = True)
            else:
                qsr, qsr_d_reverse = Parallel(n_jobs=2, backend=backend)(
                    [
                        delayed(solve_qsr_3d_neumann)(Scft_params, wsr, qsr_initial, reverse = False),
                        delayed(solve_qsr_3d_neumann)(Scft_params, wsr, qsr_d_initial, reverse = True),
                    ]
                )

        else: # for 3D fft, doing in series is faster on GPU
            if use_cupy == True:
                qsr = solve_qsr_3d(Scft_params, wsr, qsr_initial, reverse = False)
                qsr_d_reverse = solve_qsr_3d(Scft_params, wsr, qsr_d_initial, reverse = True)
            else:
                qsr, qsr_d_reverse = Parallel(n_jobs=2, backend=backend)(
                    [
                        delayed(solve_qsr_3d)(Scft_params, wsr, qsr_initial, reverse = False),
                        delayed(solve_qsr_3d)(Scft_params, wsr, qsr_d_initial, reverse = True),
                    ]
                )
        qsr_d = xp.flip(qsr_d_reverse, axis=0)
        phi_blocks, phi_s, phi, Qc = get_phi_3d(Scft_params, qsr, qsr_d)

    new_wsr, ws = get_wsr(Scft_params, phi_blocks, phi_s)
    wsr, err = wsr_update(Scft_params, wsr, new_wsr)
    free_energy = calculate_free_energy(Scft_params, wsr, phi_blocks, Qc, ws)

    return wsr, phi, phi_blocks, qsr, qsr_d, err, free_energy

def gaussian_com_2d(phi):

    x_coord = xp.arange(phi.shape[0])
    y_coord = xp.arange(phi.shape[1])

    x_com = xp.sum(x_coord * xp.sum(phi, axis=1))/xp.sum(phi)
    y_com = xp.sum(y_coord * xp.sum(phi, axis=0))/xp.sum(phi)
    com = xp.array([x_com, y_com])

    width_gaussian = phi.shape[0]/10 # narrower = larger denominator

    x_coord = xp.arange(phi.shape[0])
    y_coord = xp.arange(phi.shape[1])
    X,Y = xp.meshgrid(x_coord, y_coord, indexing='ij')
    gaussian = xp.exp(-((X-com[0])**2 + (Y-com[1])**2)/width_gaussian)
    return gaussian/xp.mean(gaussian)

def scft_loop(Scft_params, wsr, qsr_initial, qsr_d_initial, diff, free_energy_hist, polymer_loop = False, return_qsr = True, tethered = False, tether_start = 0):

    for i in tqdm(range(Scft_params.iterations)):
        Scft_params.iteration = i
        wsr, phi, phi_blocks, qsr, qsr_d, err, free_energy = scft_iteration(Scft_params, wsr, qsr_initial, qsr_d_initial)

        diff.append(to_numpy(err))
        free_energy_hist.append(to_numpy(free_energy))
        #print(qsr.dtype)

        if (polymer_loop == True):
            phi_sr = qsr * qsr_d
            if (tethered == False) or i<tether_start:
                qsr_initial = (1-Scft_params.mixing_rate)*qsr_initial + Scft_params.mixing_rate*(phi_sr[0] / xp.mean(phi_sr[0]))
                qsr_d_initial = (1-Scft_params.mixing_rate)*qsr_d_initial + Scft_params.mixing_rate*(phi_sr[-1] / xp.mean(phi_sr[-1]))
            else: # tether
                qsr_initial = (1 - Scft_params.mixing_rate) * qsr_initial + Scft_params.mixing_rate * gaussian_com_2d(qsr_d[0])
                qsr_d_initial = (1 - Scft_params.mixing_rate) * qsr_initial + Scft_params.mixing_rate * gaussian_com_2d(qsr_d[-1])

        if np.isnan(to_numpy(err)):
            diff.append(100)
            break

    if return_qsr == True:
        return wsr, phi, phi_blocks, qsr, qsr_d, diff, free_energy_hist
    else:
        return wsr, phi, phi_blocks, qsr_initial, qsr_d_initial, diff, free_energy_hist

def initialization(Scft_params, shift, seed = 7443):
    np.random.seed(seed)
    wsr = 1e-2 * np.random.random((Scft_params.chain_interaction.shape[0], *Scft_params.initial_qsr.shape)) / Scft_params.N
    if use_cupy == True:
        wsr = cp.asarray(wsr, dtype=DTYPE)
    qsr_initial = Scft_params.initial_qsr
    qsr_d_initial = Scft_params.initial_qsr

    qsr_initial = xp.roll(qsr_initial, (shift, shift), axis = (0,1))
    qsr_d_initial = xp.roll(qsr_d_initial, (-shift, -shift), axis = (0,1))

    diff = []
    free_energy_hist = []

    return wsr, qsr_initial, qsr_d_initial, diff, free_energy_hist
