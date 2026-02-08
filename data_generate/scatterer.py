"""Generate arbitrary-shaped scatterers via filtered random noise.

This module exposes helpers to create a boolean mask from filtered random
noise and to convert that mask into a list of small `mp.Block` objects
that approximate an arbitrary-shaped scatterer. The returned list can be
passed directly as the `geometry` argument to a Meep `mp.Simulation`.

Usage example:
    from utils.scatterer import generate_scatterer_geometry
    geometry = generate_scatterer_geometry(cell=mp.Vector3(2,2,0), nx=200, ny=200,
                                          sigma=3.0, threshold=0.5, epsilon=3.0)
    sim = mp.Simulation(cell_size=cell, geometry=geometry, ...)
"""
from typing import Tuple, List
import numpy as np
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    import imageio
except Exception:
    imageio = None

import meep as mp



def _gaussian_blur_numpy(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Fallback Gaussian blur using FFT convolution when scipy is not available."""
    if sigma <= 0:
        return arr.copy()
    # construct 2D gaussian kernel in frequency domain
    ny, nx = arr.shape
    y = np.fft.fftfreq(ny).reshape(ny, 1)
    x = np.fft.fftfreq(nx).reshape(1, nx)
    # gaussian in freq domain: exp(-2*pi^2*sigma^2*(f_x^2 + f_y^2))
    f2 = x * x + y * y
    kernel_ft = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * f2)
    arr_ft = np.fft.fft2(arr)
    res = np.fft.ifft2(arr_ft * kernel_ft).real
    return res


def generate_filtered_noise(shape: Tuple[int, int] = (200, 200),
                            sigma: float = 3.0,
                            seed: int = None) -> np.ndarray:
    """Generate a filtered random noise field in [0,1].

    Parameters
    - shape: (ny, nx) grid size
    - sigma: gaussian blur sigma (in pixels)
    - seed: optional RNG seed for reproducibility
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.rand(*shape)
    else:
        noise = np.random.rand(*shape)

    if gaussian_filter is not None:
        filt = gaussian_filter(noise, sigma=sigma, mode='wrap')
    else:
        filt = _gaussian_blur_numpy(noise, sigma)

    # normalize to 0..1
    mi = filt.min()
    ma = filt.max()
    if ma > mi:
        filt = (filt - mi) / (ma - mi)
    else:
        filt = np.zeros_like(filt)
    return filt


def mask_from_noise(noise: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold noise to produce a boolean mask.

    Values >= threshold are considered inside the scatterer.
    """
    return (noise >= threshold)


def mask_to_meep_blocks(mask: np.ndarray, cell: mp.Vector3,
                        epsilon: float = 3.0) -> List[mp.Block]:
    """Convert boolean mask to a list of small `mp.Block` objects.

    The mask is interpreted with shape (ny, nx). The physical extent is
    assumed to be the full `cell.x` by `cell.y` centered at the origin.
    Each True pixel becomes a block sized `dx = cell.x / nx`, `dy = cell.y / ny`.
    """
    ny, nx = mask.shape
    dx = float(cell.x) / float(nx)
    dy = float(cell.y) / float(ny)

    # pixel centers: evenly spaced across the cell
    xs = np.linspace(-float(cell.x) / 2.0 + dx / 2.0,
                     float(cell.x) / 2.0 - dx / 2.0, nx)
    ys = np.linspace(-float(cell.y) / 2.0 + dy / 2.0,
                     float(cell.y) / 2.0 - dy / 2.0, ny)

    blocks: List[mp.Block] = []
    # iterate where mask is True and create blocks
    true_idx = np.transpose(np.nonzero(mask))  # rows of (iy, ix)
    for iy, ix in true_idx:
        cx = float(xs[ix])
        cy = float(ys[iy])
        bl = mp.Block(mp.Vector3(dx, dy, mp.inf), center=mp.Vector3(cx, cy, 0), material=mp.Medium(epsilon=epsilon))
        blocks.append(bl)
    return blocks


def generate_scatterer_geometry(cell: mp.Vector3 = mp.Vector3(2, 2, 0),
                                nx: int = 200,
                                ny: int = 200,
                                sigma: float = 3.0,
                                threshold: float = 0.5,
                                seed: int = None,
                                epsilon: float = 3.0,
                                return_mask: bool = False):
    """Generate geometry list for Meep based on filtered random noise.

    If `return_mask` is True, returns a tuple `(blocks, mask)` where `mask`
    is a boolean ndarray of shape `(ny, nx)` used to create the blocks.
    Otherwise returns only the list of `mp.Block` objects ready to be
    passed to `Simulation(..., geometry=...)`.
    """
    noise = generate_filtered_noise(shape=(ny, nx), sigma=sigma, seed=seed)
    mask = mask_from_noise(noise, threshold=threshold)
    blocks = mask_to_meep_blocks(mask, cell=cell, epsilon=epsilon)
    if return_mask:
        return blocks, mask
    return blocks


def generate_filtered_noise_3d(shape: Tuple[int, int, int] = (60, 60, 60),
                               sigma: float = 3.0,
                               seed: int = None) -> np.ndarray:
    """Generate a filtered random noise field in 3D [0,1].
    shape: (nz, ny, nx)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.rand(*shape)
    else:
        noise = np.random.rand(*shape)

    if gaussian_filter is not None:
        filt = gaussian_filter(noise, sigma=sigma, mode='wrap')
    else:
        raise ImportError("Scipy is required for 3D gaussian filter.")

    mi = filt.min()
    ma = filt.max()
    if ma > mi:
        filt = (filt - mi) / (ma - mi)
    else:
        filt = np.zeros_like(filt)
    return filt

def mask_from_noise(noise: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (noise >= threshold)

def mask_to_meep_blocks_3d(mask: np.ndarray, cell: mp.Vector3,
                           epsilon: float = 3.0) -> List[mp.Block]:
    """Convert 3D boolean mask to a list of small mp.Block objects."""
    # mask shape is (nz, ny, nx)
    nz, ny, nx = mask.shape
    dx = float(cell.x) / float(nx)
    dy = float(cell.y) / float(ny)
    dz = float(cell.z) / float(nz)

    # Calculate coordinates
    xs = np.linspace(-float(cell.x)/2.0 + dx/2.0, float(cell.x)/2.0 - dx/2.0, nx)
    ys = np.linspace(-float(cell.y)/2.0 + dy/2.0, float(cell.y)/2.0 - dy/2.0, ny)
    zs = np.linspace(-float(cell.z)/2.0 + dz/2.0, float(cell.z)/2.0 - dz/2.0, nz)

    blocks: List[mp.Block] = []
    
    true_idx = np.transpose(np.nonzero(mask)) 
    
    for iz, iy, ix in true_idx:
        cx = float(xs[ix])
        cy = float(ys[iy])
        cz = float(zs[iz])
        bl = mp.Block(mp.Vector3(dx, dy, dz), 
                      center=mp.Vector3(cx, cy, cz), 
                      material=mp.Medium(epsilon=epsilon))
        blocks.append(bl)
    return blocks

def generate_scatterer_geometry_3d(cell: mp.Vector3,
                                   nx: int = 60, ny: int = 60, nz: int = 60,
                                   sigma: float = 3.0,
                                   threshold: float = 0.5,
                                   seed: int = None,
                                   epsilon: float = 3.0,
                                   return_mask: bool = False):
    """3D Geometry generation wrapper."""
    noise = generate_filtered_noise_3d(shape=(nz, ny, nx), sigma=sigma, seed=seed)
    mask = mask_from_noise(noise, threshold=threshold)
    blocks = mask_to_meep_blocks_3d(mask, cell=cell, epsilon=epsilon)
    if return_mask:
        return blocks, mask
    return blocks


__all__ = [
    'generate_filtered_noise',
    'mask_from_noise',
    'mask_to_meep_blocks',
    'generate_scatterer_geometry',
]


if __name__ == '__main__':
    # Quick visual check: generate noise, mask and save a PNG showing the mask
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Visualize generated scatterer mask')
    parser.add_argument('--nx', type=int, default=160, help='mask grid nx')
    parser.add_argument('--ny', type=int, default=160, help='mask grid ny')
    parser.add_argument('--sigma', type=float, default=4.0, help='gaussian blur sigma')
    parser.add_argument('--threshold', type=float, default=0.55, help='mask threshold')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--epsilon', type=float, default=3.0, help='permittivity (unused for plot)')
    parser.add_argument('--out', default='./utils/filter_mask_figures/', help='Output PNG path (defaults to figures/2d_Ez)')
    parser.add_argument('--show-blocks', action='store_true', help='Also draw small block rectangles (may be slow)')
    args = parser.parse_args()

    noise = generate_filtered_noise(shape=(args.ny, args.nx), sigma=args.sigma, seed=args.seed)
    mask = mask_from_noise(noise, threshold=args.threshold)

    # prepare output path: default -> save in current working directory
    default_name = f'filter_mask_n{args.nx}_res{args.ny}_s{int(args.sigma)}_t{int(args.threshold*100)}.png'
    if args.out is not None:
        outpath = os.path.join(args.out, default_name) if os.path.isdir(args.out) else args.out
    else:
        outpath = default_name

    # lazy-matplotlib import
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        raise RuntimeError('matplotlib is required to run this demo')

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mask, origin='lower', cmap='gray', extent=[-1, 1, -1, 1])
    ax.set_title('Scatterer mask (True = scatterer)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if args.show_blocks:
        # draw a subset of blocks to avoid huge number of rectangles
        blocks = mask_to_meep_blocks(mask, cell=mp.Vector3(2, 2, 0), epsilon=args.epsilon)
        max_draw = 2000
        step = max(1, len(blocks) // max_draw)
        for bl in blocks[::step]:
            try:
                sx = float(getattr(bl.size, 'x', getattr(bl.size, 0)))
                sy = float(getattr(bl.size, 'y', getattr(bl.size, 1)))
                cx = float(getattr(bl.center, 'x', getattr(bl.center, 0)))
                cy = float(getattr(bl.center, 'y', getattr(bl.center, 1)))
                rect = Rectangle((cx - sx/2.0, cy - sy/2.0), sx, sy, edgecolor='r', facecolor='none', linewidth=0.3)
                ax.add_patch(rect)
            except Exception:
                continue

    plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print('Saved mask image to', outpath)

# python -m utils.scatterer --nx 160 --ny 160 --sigma 4.0 --threshold 0.55 --seed 42