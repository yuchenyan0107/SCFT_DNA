from datashader.datashape import integral

from .QSR_solvers import *
from .QSR_dct import *
from .use_cupy_dtype import *

class Scft:
    def __init__(self, N, Lx, Ly, phibar, l_p, n_seg, nx, ny, ns,
                 iterations, error_tol, chain_interaction,
                 self_avoiding, initial_qsr, mixing_rate, chi_polymer_block,
                 chi_polymer_s, PB = 1, close_boundary = True, D3 = False, Lz = None, nz = None,
                 thickness = 1, wall_strength = None,
                 anderson_m = 5, anderson_start_iter = 100, anderson_beta = None,
                 anderson_regularization = 1e-8,
                 ):
        # parameters
        self.Lx, self.Ly = Lx, Ly
        self.phibar = phibar
        self.l_p = l_p
        self.n_seg = n_seg
        self.nx, self.ny = nx, ny
        self.ns = ns
        self.iterations = iterations
        self.error_tol = error_tol
        self.PB = PB

        self.N = N # multiply to ws in get_wsr

        # move arrays to GPU
        self.chain_interaction = xp.array(chain_interaction)
        self.self_avoiding = self_avoiding
        self.initial_qsr = xp.array(initial_qsr)
        self.chi_polymer_block = xp.array(chi_polymer_block)
        self.chi_polymer_s = xp.array(chi_polymer_s)
        self.mixing_rate = mixing_rate
        self.close_boundary = close_boundary

        # discretization
        self.ds = n_seg / ns
        self.dx = Lx / nx
        self.dy = Ly / ny

        self.D3 = D3
        if D3 == True:
            self.Lz = Lz
            self.nz = nz
            self.dz = Lz/nz

        self.thickness = thickness
        if wall_strength == None:
            self.wall_strength = xp.mean(self.chi_polymer_s)
        else:
            self.wall_strength = wall_strength

        # Anderson mixing controls (2D default)
        self.anderson_m = anderson_m
        self.anderson_start_iter = anderson_start_iter
        self.anderson_beta = mixing_rate if anderson_beta is None else anderson_beta
        self.anderson_regularization = anderson_regularization
        self._anderson_residuals = []
        self._anderson_candidates = []


def get_phi(Scft_params, qsr, qsr_d):

    phi_blocks = xp.empty((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny), dtype=DTYPE)
    Qc = xp.sum(qsr[Scft_params.ns]) * Scft_params.dx * Scft_params.dy / (Scft_params.Lx * Scft_params.Ly)
    for m in range(Scft_params.chain_interaction.shape[0]):
        integrand = qsr * qsr_d * Scft_params.chain_interaction[m][:, None, None]
        phi_blocks[m] = Scft_params.ds * xp.sum(integrand, axis=0) / Qc
    phi = Scft_params.ds * xp.sum(qsr * qsr_d, axis=0) / Qc
    scale = Scft_params.phibar / xp.mean(phi)
    phi_blocks *= scale
    phi = phi * scale
    phi_s = 1 - phi
    phi_s = xp.clip(phi_s, 1e-12, 1 - 1e-12)
    return phi_blocks, phi_s, phi, Qc

def get_phi_3d(Scft_params, qsr, qsr_d):

    phi_blocks = xp.empty((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny, Scft_params.nz), dtype=DTYPE)
    Qc = xp.sum(qsr[Scft_params.ns]) * Scft_params.dx * Scft_params.dy * Scft_params.dz / (Scft_params.Lx * Scft_params.Ly * Scft_params.Lz)
    for m in range(Scft_params.chain_interaction.shape[0]):
        integrand = qsr * qsr_d * Scft_params.chain_interaction[m][:, None, None, None]
        phi_blocks[m] = Scft_params.ds * xp.sum(integrand, axis=0) / Qc
    phi = Scft_params.ds * xp.sum(qsr * qsr_d, axis=0) / Qc
    scale = Scft_params.phibar / xp.mean(phi)
    phi_blocks *= scale
    phi = phi * scale
    phi_s = 1 - phi
    phi_s = xp.clip(phi_s, 1e-12, 1 - 1e-12)
    return phi_blocks, phi_s, phi, Qc

def get_wsr(Scft_params, phi_blocks, phi_s):

    ws = -xp.log(phi_s) * Scft_params.N # field of solvent
    eta = ws - xp.tensordot(Scft_params.chi_polymer_s, phi_blocks, axes=1) # incompressible
    #new_wsr = xp.zeros((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny), dtype=DTYPE)
    new_wsr = xp.zeros(phi_blocks.shape, dtype=DTYPE)
    for m in range(Scft_params.chain_interaction.shape[0]):
        blend = xp.zeros((phi_s.shape), dtype=DTYPE) # phi_s shape : nx,ny,nz
        for n in range(Scft_params.chain_interaction.shape[0]):
            if n != m:
                blend += Scft_params.chi_polymer_block[m, n] * phi_blocks[n]
        w_block = blend + Scft_params.chi_polymer_s[m] * phi_s + eta
        new_wsr[m] = w_block

    if Scft_params.close_boundary: # if qsr can't pass boundary, set the boundary potential high
        new_wsr = wall_potential(Scft_params, new_wsr)

    # A uniform shift of the wsr field doesn't change the conformation and free energy. It prevents the qsr propagator to run out floating point limit, because it decays exponentially.
    new_wsr = new_wsr - xp.mean(new_wsr)

    return new_wsr, ws

def wall_potential(Scft_params, new_wsr):
    if Scft_params.D3 == True:
        new_wsr[:, :Scft_params.thickness, :, :] += Scft_params.wall_strength
        new_wsr[:, -Scft_params.thickness:, :, :] += Scft_params.wall_strength
        new_wsr[:,: , :Scft_params.thickness, :] += Scft_params.wall_strength
        new_wsr[:, :, -Scft_params.thickness:, :] += Scft_params.wall_strength
        new_wsr[:, :, :, :Scft_params.thickness] += Scft_params.wall_strength
        new_wsr[:, :, :, -Scft_params.thickness:] += Scft_params.wall_strength
    else:
        new_wsr[:, :Scft_params.thickness, :] += Scft_params.wall_strength
        new_wsr[:, -Scft_params.thickness:, :] += Scft_params.wall_strength
        new_wsr[:, :, :Scft_params.thickness] += Scft_params.wall_strength
        new_wsr[:, :, -Scft_params.thickness:] += Scft_params.wall_strength

    return new_wsr

def wsr_update(Scft_params, old_wsr, new_wsr):
    wsr = Scft_params.mixing_rate * new_wsr + (1 - Scft_params.mixing_rate) * old_wsr
    err = xp.linalg.norm(wsr - old_wsr) / xp.sqrt(wsr.size)
    return wsr, err

def wsr_update_Anderson(Scft_params, old_wsr, new_wsr): # this function is given by chatGPT but we don't actually use it

    # Handle degenerate case or early iterations: fall back to simple mixing
    if Scft_params.anderson_m <= 0 or getattr(Scft_params, "iteration", 0) < Scft_params.anderson_start_iter:
        return wsr_update(Scft_params, old_wsr, new_wsr)

    # Book-keeping containers: residuals f_i = F(w_i) - w_i and states w_i
    if not hasattr(Scft_params, "_anderson_residuals"):
        Scft_params._anderson_residuals = []
        Scft_params._anderson_candidates = []

    # Reset history at the beginning of a run
    if getattr(Scft_params, "iteration", 0) == 0:
        Scft_params._anderson_residuals.clear()
        Scft_params._anderson_candidates.clear()

    # Fixed-point residual at current iterate w_k
    residual = new_wsr - old_wsr

    # Store (w_k, f_k) in flattened form
    Scft_params._anderson_residuals.append(residual.reshape(-1))
    Scft_params._anderson_candidates.append(old_wsr.reshape(-1))

    # Keep at most 'anderson_m' most recent pairs
    if len(Scft_params._anderson_residuals) > Scft_params.anderson_m:
        Scft_params._anderson_residuals = Scft_params._anderson_residuals[-Scft_params.anderson_m:]
        Scft_params._anderson_candidates = Scft_params._anderson_candidates[-Scft_params.anderson_m:]

    beta = Scft_params.anderson_beta

    # Not enough history yet: simple linear mixing
    if len(Scft_params._anderson_residuals) < 2:
        wsr = old_wsr + beta * residual
        err = xp.linalg.norm(wsr - old_wsr) / xp.sqrt(wsr.size)
        return wsr, err

    k = len(Scft_params._anderson_residuals)
    res_hist = Scft_params._anderson_residuals[-k:]
    w_hist = Scft_params._anderson_candidates[-k:]

    # Build Gram matrix B_ij = <f_i, f_j>
    B = xp.empty((k, k), dtype=DTYPE)
    for i in range(k):
        for j in range(i, k):
            val = xp.vdot(res_hist[i], res_hist[j]).real
            B[i, j] = val
            B[j, i] = val
    if Scft_params.anderson_regularization > 0:
        B += Scft_params.anderson_regularization * xp.eye(k, dtype=DTYPE)

    # Solve DIIS/Anderson system:
    #   minimize || sum_i gamma_i f_i ||  s.t. sum_i gamma_i = 1
    ones = xp.ones(k, dtype=DTYPE)
    mat = xp.empty((k + 1, k + 1), dtype=DTYPE)
    mat[:k, :k] = B
    mat[:k, -1] = ones
    mat[-1, :k] = ones
    mat[-1, -1] = 0.0

    rhs = xp.zeros(k + 1, dtype=DTYPE)
    rhs[-1] = 1.0

    try:
        gamma = xp.linalg.solve(mat, rhs)[:k]
        # Affine combination of past fields w_i
        acc_flat = gamma[0] * w_hist[0]
        for idx in range(1, k):
            acc_flat = acc_flat + gamma[idx] * w_hist[idx]
        w_acc = acc_flat.reshape(new_wsr.shape)
        # Optional damping towards latest iterate
        wsr = (1.0 - beta) * old_wsr + beta * w_acc
    except Exception:
        # If the small linear system is ill-conditioned, revert to simple mixing
        wsr = old_wsr + beta * residual

    err = xp.linalg.norm(wsr - old_wsr) / xp.sqrt(wsr.size)
    return wsr, err

def calculate_free_energy(Scft_params, wsr, phi_blocks, Qc, ws):

    if Scft_params.D3 == True:
        integral_divide_by_volume = Scft_params.dx * Scft_params.dy * Scft_params.dz / (Scft_params.Lx * Scft_params.Ly * Scft_params.Lz)
    else:
        integral_divide_by_volume = Scft_params.dx * Scft_params.dy / (Scft_params.Lx * Scft_params.Ly)

    if Scft_params.N == 1:
        entropy_polymer = Scft_params.phibar * xp.log(Qc) / Scft_params.n_seg
        Qs = xp.sum(xp.exp(-ws)) * integral_divide_by_volume
        entropy_solvent = (1 - Scft_params.phibar) * xp.log(Qs)
    else:
        entropy_polymer = Scft_params.phibar * xp.log(Qc/Scft_params.phibar)
        ws_scaled = ws / Scft_params.N
        Qs = xp.sum(xp.exp(-ws_scaled)) * integral_divide_by_volume
        entropy_solvent = Scft_params.N * (1-Scft_params.phibar) * xp.log(Qs)

    phi_p = xp.sum(phi_blocks, axis = 0)
    phi_s = 1-phi_p
    field_constrain = (xp.sum(wsr * phi_blocks) + xp.sum(phi_s * ws)) * integral_divide_by_volume

    constrains = - field_constrain - entropy_polymer - entropy_solvent

    enthalpy_PP = 0
    for m in range(Scft_params.chi_polymer_s.shape[0]-1):
        for n in range(m+1, Scft_params.chi_polymer_s.shape[0]):
            enthalpy_PP += Scft_params.chi_polymer_block[m, n] * xp.sum(phi_blocks[m]* phi_blocks[n])

    if Scft_params.D3 == True:
        enthalpy_PS = xp.sum(Scft_params.chi_polymer_s[:, None, None, None] * phi_blocks * phi_s[None, :, :, :])
    else:
        enthalpy_PS = xp.sum(Scft_params.chi_polymer_s[:, None, None] * phi_blocks * phi_s[None, :, :])

    interaction_enthalpy = (enthalpy_PP + enthalpy_PS) * integral_divide_by_volume

    free_energy = interaction_enthalpy + constrains

    return to_numpy(xp.array([free_energy, interaction_enthalpy, constrains]))