from .QSR_solvers import *

from scipy.fftpack import dct, idct
def dct2(a):
    return dct(dct(a, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
def idct2(a):
    return idct(idct(a, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')


def _even_extension_2d(A, nx, ny):
    A2 = xp.empty((nx * 2, ny * 2), dtype=xp.float64)
    A2[:nx, :ny] = A
    A2[nx:, :ny] = A[::-1, :]  # x-axis inverted
    A2[:nx, ny:] = A[:, ::-1]
    A2[nx:, ny:] = A[::-1, ::-1]
    return A2

def solve_qsr_2d_neumann(Scft_params, wsr, qsr_initial, reverse):

    qsr = xp.zeros((Scft_params.ns+1, Scft_params.nx, Scft_params.ny), dtype=xp.float64)
    qsr[0] = qsr_initial/xp.mean(qsr_initial)

    nx2, ny2 = Scft_params.nx * 2, Scft_params.ny * 2  # extend the grid
    kx = xp.fft.fftfreq(nx2, d=Scft_params.dx)
    ky = xp.fft.fftfreq(ny2, d=Scft_params.dy)
    K2 = (2.0 * xp.pi * kx)[:, None] ** 2 + (2.0 * xp.pi * ky)[None, :] ** 2  # this line is the same

    D = Scft_params.l_p ** 2 / 6.0
    exp_L = xp.exp(-Scft_params.ds / Scft_params.PB * D * K2) # diffusion operator in fourier space

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0) # correlates s index to the segment type
    if reverse:
        which_block = xp.flip(which_block)
    exponent_W = xp.exp(-wsr * Scft_params.ds /Scft_params.PB /2) # W operator for each types of segment

    q_real = xp.empty_like(qsr_initial)  # temporary q in the split op computation
    q_real_extend = xp.empty((nx2, ny2), dtype=xp.float64)  # extended
    qsr_temp_s = qsr[0].copy() # for storage of intermediate qsr that's not actually in the output array

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_temp_s, exponent_W[which_block[s//Scft_params.PB]], out = q_real)
        q_real_extend = _even_extension_2d(q_real, Scft_params.nx, Scft_params.ny)

        q_k  = xp.fft.fft2(q_real_extend)
        xp.multiply(q_k, exp_L, out = q_k)
        q_real_extend = xp.fft.ifft2(q_k).real

        q_real = q_real_extend[:Scft_params.nx, :Scft_params.ny]

        xp.multiply(q_real, exponent_W[which_block[s//Scft_params.PB]], out = qsr_temp_s)

        if (s+1) % Scft_params.PB == 0:
            qsr[(s+1) // Scft_params.PB] = qsr_temp_s.copy()

    return qsr

'''
def solve_qsr_d_2d_neumann(Scft_params, wsr, qsr_d_initial):
    qsr_d = xp.zeros((Scft_params.ns + 1, Scft_params.nx, Scft_params.ny), dtype=xp.float64)
    qsr_d[0] = qsr_d_initial / xp.mean(qsr_d_initial)

    nx2, ny2 = Scft_params.nx * 2, Scft_params.ny * 2  # extend the grid
    kx = xp.fft.fftfreq(nx2, d=Scft_params.dx)
    ky = xp.fft.fftfreq(ny2, d=Scft_params.dy)
    K2 = (2.0 * xp.pi * kx)[:, None] ** 2 + (2.0 * xp.pi * ky)[None, :] ** 2  # this line is the same

    D = Scft_params.l_p ** 2 / 6.0
    exp_L = xp.exp(-Scft_params.ds / Scft_params.PB * D * K2)

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0)
    exponent_W = xp.exp(-wsr * Scft_params.ds / Scft_params.PB / 2)

    q_real = xp.empty_like(qsr_d_initial)
    q_real_extend = xp.empty((nx2, ny2), dtype=xp.float64)  # extended
    qsr_d_temp_s = qsr_d[0].copy()

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_d_temp_s, exponent_W[which_block[Scft_params.ns - 1 - s // Scft_params.PB]], out=q_real)
        q_real_extend = _even_extension_2d(q_real, Scft_params.nx, Scft_params.ny)

        q_k = xp.fft.fft2(q_real_extend)
        xp.multiply(q_k, exp_L, out=q_k)
        q_real_extend = xp.fft.ifft2(q_k).real

        q_real = q_real_extend[:Scft_params.nx, :Scft_params.ny]

        xp.multiply(q_real, exponent_W[which_block[Scft_params.ns - 1 - s // Scft_params.PB]], out=qsr_d_temp_s)

        if (s + 1) % Scft_params.PB == 0:
            qsr_d[(s + 1) // Scft_params.PB] = qsr_d_temp_s.copy()

    return xp.flip(qsr_d, axis=0)
'''

###################### the DCT method (CPU only) ###################

def fix_wall(A):
    A[:, 0, :] = A[:, 1, :]
    A[:, -1, :] = A[:, -2, :]
    A[:, :, 0] = A[:, :, 1]
    A[:, :, -1] = A[:, :, -2]
    return A

def solve_qsr_2d_dct_expand(Scft_params, wsr, qsr_initial, reverse):
    nx = Scft_params.nx + 2
    ny = Scft_params.ny + 2
    dx = Scft_params.dx
    dy = Scft_params.dy
    Lx = (nx - 1) * dx
    Ly = (ny - 1) * dy

    qsr = xp.zeros((Scft_params.ns + 1, nx, ny), dtype=xp.float64)
    qsr[0, 1:-1, 1:-1] = qsr_initial / xp.mean(qsr_initial)
    qsr = qsr/2 + fix_wall(qsr)/2

    px = xp.arange(nx)
    py = xp.arange(ny)

    kx = xp.pi * px / Lx
    ky = xp.pi * py / Ly
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2

    D = Scft_params.l_p ** 2 / 6.0
    exp_L = xp.exp(-Scft_params.ds / Scft_params.PB * D * K2)  # diffusion operator in fourier space

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0)  # correlates s index to the segment type

    expand_wsr = xp.zeros((Scft_params.chain_interaction.shape[0], nx, ny), dtype=xp.float64)
    expand_wsr[:, 1:-1, 1:-1] = wsr
    expand_wsr = fix_wall(expand_wsr)
    if reverse:
        which_block = xp.flip(which_block)
    exponent_W = xp.exp(-expand_wsr * Scft_params.ds / Scft_params.PB / 2)  # W operator for each types of segment

    q_real = xp.empty_like(qsr[0])  # temporary q in the split op computation
    qsr_temp_s = qsr[0].copy()  # for storage of intermediate qsr that's not actually in the output array

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_temp_s, exponent_W[which_block[s // Scft_params.PB]], out=q_real)

        q_k = dct2(q_real)

        xp.multiply(q_k, exp_L, out=q_k)

        q_real = idct2(q_k).real

        xp.multiply(q_real, exponent_W[which_block[s // Scft_params.PB]], out=qsr_temp_s)

        if (s + 1) % Scft_params.PB == 0:
            qsr[(s + 1) // Scft_params.PB] = qsr_temp_s.copy()

    qsr = fix_wall(qsr)

    return qsr[:, 1:-1, 1:-1]
