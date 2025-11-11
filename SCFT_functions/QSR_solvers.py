import numpy as np
import matplotlib.pyplot as plt
use_cupy = True

if use_cupy:

    try:
        import cupy as cp
        xp = cp
        print("Using cupy")
    except ImportError:
        cp = None
        xp = np
        print("Using numpy")
else:
    cp = None
    xp = np
    print("Using numpy")

def to_numpy(arr):
    if cp is not None and hasattr(cp, "asnumpy") and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def solve_qsr_2d(Scft_params, wsr, qsr_initial, reverse):


    qsr = xp.zeros((Scft_params.ns+1, Scft_params.nx, Scft_params.ny), dtype=xp.float64)
    qsr[0] = qsr_initial/xp.mean(qsr_initial)

    # --- pre-compute k-grid & constant factor ----------------------
    kx, ky = _k_mesh_2d(Scft_params.nx, Scft_params.ny, Scft_params.dx, Scft_params.dy)
    D      = Scft_params.l_p**2 / 6.0
    exp_L = xp.exp(-Scft_params.ds/Scft_params.PB * D * ((2*xp.pi*kx)**2 + (2*xp.pi*ky)**2)) # diffusion operator in fourier space

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0) # correlates s index to the segment type
    if reverse:
        which_block = xp.flip(which_block)
    exponent_W = xp.exp(-wsr * Scft_params.ds /Scft_params.PB /2) # W operator for each types of segment

    q_real = xp.empty_like(qsr_initial)  # temporary q in the split op computation
    qsr_temp_s = qsr[0].copy() # for storage of intermediate qsr that's not actually in the output array

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_temp_s, exponent_W[which_block[s//Scft_params.PB]], out = q_real)

        q_k  = xp.fft.fft2(q_real)

        xp.multiply(q_k, exp_L, out = q_k)

        q_real = xp.fft.ifft2(q_k).real

        xp.multiply(q_real, exponent_W[which_block[s//Scft_params.PB]], out = qsr_temp_s)

        if (s+1) % Scft_params.PB == 0:
            qsr[(s+1) // Scft_params.PB] = qsr_temp_s.copy()

    return qsr

def _k_mesh_2d(nx, ny, dx, dy):
    kx = xp.fft.fftfreq(nx, d=dx)               # (nx,)
    ky = xp.fft.fftfreq(ny, d=dy)               # (ny,)
    return xp.meshgrid(ky, kx, indexing='ij')   # both (ny,nx)

def _k_mesh_3d(nx: int, ny: int, nz: int,
              dx: float, dy: float, dz: float):
    kx = xp.fft.fftfreq(nx, d=dx)
    ky = xp.fft.fftfreq(ny, d=dy)
    kz = xp.fft.fftfreq(nz, d=dz)
    kx3d, ky3d, kz3d = xp.meshgrid(kx, ky, kz, indexing="ij")
    return kx3d, ky3d, kz3d

def solve_qsr_3d(Scft_params, wsr, qsr_initial, reverse):


    qsr = xp.zeros((Scft_params.ns+1, Scft_params.nx, Scft_params.ny, Scft_params.nz), dtype=xp.float64)
    qsr[0] = qsr_initial/xp.mean(qsr_initial)

    # --- pre-compute k-grid & constant factor ----------------------
    kx, ky, kz = _k_mesh_3d(Scft_params.nx, Scft_params.ny, Scft_params.ny, Scft_params.dx, Scft_params.dy, Scft_params.dz)
    D      = Scft_params.l_p**2 / 6.0
    exp_L = xp.exp(-Scft_params.ds/Scft_params.PB * D * (
            (2*xp.pi*kx)**2 + (2*xp.pi*ky)**2 + (2*xp.pi*kz)**2
    )) # diffusion operator in fourier space

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0) # correlates s index to the segment type
    if reverse:
        which_block = xp.flip(which_block)
    exponent_W = xp.exp(-wsr * Scft_params.ds /Scft_params.PB /2) # W operator for each types of segment

    q_real = xp.empty_like(qsr_initial)  # temporary q in the split op computation
    qsr_temp_s = qsr[0].copy() # for storage of intermediate qsr that's not actually in the output array

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_temp_s, exponent_W[which_block[s//Scft_params.PB]], out = q_real)

        q_k  = xp.fft.fftn(q_real)

        xp.multiply(q_k, exp_L, out = q_k)

        q_real = xp.fft.ifftn(q_k).real

        xp.multiply(q_real, exponent_W[which_block[s//Scft_params.PB]], out = qsr_temp_s)

        if (s+1) % Scft_params.PB == 0:
            qsr[(s+1) // Scft_params.PB] = qsr_temp_s.copy()

    return qsr

def _even_extension_3d(A, nx, ny, nz):
    A2 = xp.empty((nx * 2, ny * 2, nz * 2), dtype=xp.float64)
    A2[:nx, :ny, :nz] = A

    A2[nx:, :ny, :nz] = A[::-1, :, :] # x
    A2[:nx, ny:, :nz] = A[:, ::-1, :] # y
    A2[:nx, :ny, nz:] = A[:, :, ::-1] # z

    A2[nx:, ny:, :nz] = A[::-1, ::-1, :] # x y
    A2[nx:, :ny, nz:] = A[::-1, :, ::-1] # x z
    A2[:nx, ny:, nz:] = A[:, ::-1, ::-1] # y z

    A2[nx:, ny:, nz:] = A[::-1, ::-1, ::-1] # xyz
    return A2

def solve_qsr_3d_neumann(Scft_params, wsr, qsr_initial, reverse):


    qsr = xp.zeros((Scft_params.ns+1, Scft_params.nx, Scft_params.ny, Scft_params.nz), dtype=xp.float64)
    qsr[0] = qsr_initial/xp.mean(qsr_initial)

    nx2, ny2, nz2 = Scft_params.nx * 2, Scft_params.ny * 2, Scft_params.nz * 2  # extend the grid
    kx = xp.fft.fftfreq(nx2, d=Scft_params.dx)
    ky = xp.fft.fftfreq(ny2, d=Scft_params.dy)
    kz = xp.fft.fftfreq(nz2, d=Scft_params.dz)
    K2 = (
            (2.0 * xp.pi * kx)[:, None, None] ** 2
            + (2.0 * xp.pi * ky)[None, :, None] ** 2
            + (2.0 * xp.pi * kz)[None, None, :] ** 2
    )

    D = Scft_params.l_p**2 / 6.0
    exp_L = xp.exp(-Scft_params.ds/Scft_params.PB * D * K2) # diffusion operator in fourier space

    which_block = xp.argmax(Scft_params.chain_interaction, axis=0) # correlates s index to the segment type
    if reverse:
        which_block = xp.flip(which_block)
    exponent_W = xp.exp(-wsr * Scft_params.ds /Scft_params.PB /2) # W operator for each types of segment

    q_real = xp.empty_like(qsr_initial)  # temporary q in the split op computation
    q_real_extend = xp.empty((nx2, ny2, nz2), dtype=xp.float64)
    qsr_temp_s = qsr[0].copy() # for storage of intermediate qsr that's not actually in the output array

    # --- main propagation loop (GPU) -------------------------------
    for s in range(Scft_params.ns * Scft_params.PB):

        xp.multiply(qsr_temp_s, exponent_W[which_block[s//Scft_params.PB]], out = q_real)
        q_real_extend = _even_extension_3d(q_real, Scft_params.nx, Scft_params.ny, Scft_params.nz)

        q_k  = xp.fft.fftn(q_real_extend)
        xp.multiply(q_k, exp_L, out = q_k)
        q_real_extend = xp.fft.ifftn(q_k).real

        q_real = q_real_extend[:Scft_params.nx, :Scft_params.ny, :Scft_params.nz]
        xp.multiply(q_real, exponent_W[which_block[s//Scft_params.PB]], out = qsr_temp_s)

        if (s+1) % Scft_params.PB == 0:
            qsr[(s+1) // Scft_params.PB] = qsr_temp_s.copy()

    return qsr