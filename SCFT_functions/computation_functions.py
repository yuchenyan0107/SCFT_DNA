from .QSR_solvers import *
from .QSR_dct import *

class Scft:
    def __init__(self, N, Lx, Ly, phibar, l_p, n_seg, nx, ny, ns,
                 iterations, error_tol, chain_interaction,
                 self_avoiding, initial_qsr, mixing_rate, chi_polymer_block,
                 chi_polymer_s, PB = 1, close_boundary = True, D3 = False, Lz = None, nz = None,):
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


def get_phi(Scft_params, qsr, qsr_d):

    phi_blocks = xp.empty((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny), dtype=xp.float64)
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
    return phi_blocks, phi_s, phi

def get_phi_3d(Scft_params, qsr, qsr_d):

    phi_blocks = xp.empty((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny, Scft_params.nz), dtype=xp.float64)
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
    return phi_blocks, phi_s, phi

def get_wsr(Scft_params, phi_blocks, phi_s):

    ws = -xp.log(phi_s) * Scft_params.N # field of solvent
    eta = ws - xp.tensordot(Scft_params.chi_polymer_s, phi_blocks, axes=1) # incompressible
    #new_wsr = xp.zeros((Scft_params.chain_interaction.shape[0], Scft_params.nx, Scft_params.ny), dtype=xp.float64)
    new_wsr = xp.zeros(phi_blocks.shape, dtype=xp.float64)
    for m in range(Scft_params.chain_interaction.shape[0]):
        blend = xp.zeros((phi_s.shape), dtype=xp.float64) # phi_s shape : nx,ny,nz
        for n in range(Scft_params.chain_interaction.shape[0]):
            if n != m:
                blend += Scft_params.chi_polymer_block[m, n] * phi_blocks[n]
        w_block = blend + Scft_params.chi_polymer_s[m] * phi_s + eta
        new_wsr[m] = w_block
    '''
    if Scft_params.close_boundary: # if qsr can't pass boundary, set the boundary potential inf
        new_wsr[:, 0, :] = xp.inf
        new_wsr[:, -1, :] = xp.inf
        new_wsr[:, :, 0] = xp.inf
        new_wsr[:, :, -1] = xp.inf        
    '''
    return new_wsr

def wsr_update(Scft_params, old_wsr, new_wsr):
    wsr = Scft_params.mixing_rate * new_wsr + (1 - Scft_params.mixing_rate) * old_wsr
    err = xp.linalg.norm(wsr - old_wsr) / xp.sqrt(wsr.size)
    return wsr, err
