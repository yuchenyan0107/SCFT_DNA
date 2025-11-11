from .QSR_solvers import *
from tqdm import tqdm
from .QSR_dct import *
from .computation_functions import *
from joblib import Parallel, delayed

xp_name = getattr(globals().get('xp', None), '__name__', '')
on_gpu = (xp_name == 'cupy')
#backend   = 'threading' if on_gpu else 'loky'
backend = 'loky'

def scft_iteration(Scft_params, wsr, qsr_initial, qsr_d_initial):

    if Scft_params.D3 == False: #2D

        if Scft_params.close_boundary == True: # DCT

            qsr, qsr_d_reverse = Parallel(n_jobs=2, backend=backend)(
                [
                    delayed(solve_qsr_2d_neumann)(Scft_params, wsr, qsr_initial, reverse = False),
                    delayed(solve_qsr_2d_neumann)(Scft_params, wsr, qsr_d_initial, reverse = True),
                ]
            )
            '''
            qsr = solve_qsr_2d_dct_expand(Scft_params, wsr, qsr_initial)
            qsr_d = solve_qsr_d_2d_dct_expand(Scft_params, wsr, qsr_d_initial)
            '''
        else: # FFT
            '''
            qsr, qsr_d_reverse = Parallel(n_jobs=2, backend=backend)(
                [
                    delayed(solve_qsr_2d)(Scft_params, wsr, qsr_initial, reverse = False),
                    delayed(solve_qsr_2d)(Scft_params, wsr, qsr_d_initial, reverse = True),
                ]
            )
            '''
            qsr = solve_qsr_2d(Scft_params, wsr, qsr_initial, reverse = False)
            qsr_d_reverse = solve_qsr_2d(Scft_params, wsr, qsr_d_initial, reverse = True)

        qsr_d = xp.flip(qsr_d_reverse, axis = 0)
        phi_blocks, phi_s, phi = get_phi(Scft_params, qsr, qsr_d)

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
        phi_blocks, phi_s, phi = get_phi_3d(Scft_params, qsr, qsr_d)

    new_wsr = get_wsr(Scft_params, phi_blocks, phi_s)
    wsr, err = wsr_update(Scft_params, wsr, new_wsr)

    return wsr, phi, phi_blocks, qsr, qsr_d, err

def scft_loop(Scft_params, wsr, qsr_initial, qsr_d_initial, diff):

    for _ in tqdm(range(Scft_params.iterations)):
        wsr, phi, phi_blocks, qsr, qsr_d, err = scft_iteration(Scft_params, wsr, qsr_initial, qsr_d_initial)
        diff.append(to_numpy(err))

        qsr_initial = qsr_d[-1].copy()
        qsr_d_initial = qsr[0].copy()

    return wsr, phi, phi_blocks, qsr_initial, qsr_d_initial, diff

def initialization(Scft_params, seed = 5652):
    np.random.seed(seed)
    wsr = 1e-3 * np.random.random((Scft_params.chain_interaction.shape[0], *Scft_params.initial_qsr.shape))
    if use_cupy == True:
        wsr = cp.asarray(wsr)
    qsr_initial = Scft_params.initial_qsr
    qsr_d_initial = Scft_params.initial_qsr
    diff = []
    return wsr, qsr_initial, qsr_d_initial, diff