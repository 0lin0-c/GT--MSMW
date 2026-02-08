from meshpy.triangle import build, MeshInfo
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pickle, os
import random
from data_generate.scatterer import (
    generate_scatterer_geometry,
)

# configuration variables (moved into main guard to avoid import side-effects)


class Triangle:
    def __init__(self, a, b, c):
        self.vertices = (a, b, c)
        self.edges = self.get_edges()

    def get_edges(self):
        # Return directed (bidirectional) edges for this triangle
        a, b, c = self.vertices
        raw = [(a, b), (b, c), (a, c)]
        edges = []
        seen = set()
        for u, v in raw:
            if u == v:
                continue
            # add both directions
            for e in ((u, v), (v, u)):
                if e not in seen:
                    edges.append(e)
                    seen.add(e)
        return edges

    @staticmethod
    def collect_edges(triangles, bidirectional=True):
        """Collect edges from an iterable of Triangle objects.
        If bidirectional=True, returns directed edges (u,v) and (v,u) for each undirected edge.
        Returns a sorted list for deterministic ordering.
        """
        edges = set()
        for tri in triangles:
            for e in tri.edges:
                if bidirectional:
                    edges.add(e)
                else:
                    u, v = e
                    if u != v:
                        edges.add((min(u, v), max(u, v)))
        return sorted(edges)


# Generating a mesh on n random vertices in a rectangle
class GenerateRandomMesh:
    def __init__(self, bottom_left, top_right, n, resolution=1000):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.n = n
        self.resolution = resolution

    def generate(self):
        min_x, min_y = self.bottom_left
        max_x, max_y = self.top_right
        area = (max_x - min_x) * (max_y - min_y)

        x_values = np.random.uniform(min_x, max_x, self.n)
        y_values = np.random.uniform(min_y, max_y, self.n)

        points = list(zip(x_values, y_values))
        points.append(self.bottom_left)
        points.append((min_x, max_y))
        points.append(self.top_right)
        points.append((max_x, min_y))

        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        # point attributes (compact): Ez, eps, Hx, Hy
        # Time is no longer stored per point, but saved as a separate scalar for each snapshot
        mesh_info.number_of_point_attributes = 4
        mesh_info.set_facets([[self.n, self.n+1], [self.n+1, self.n+2], [self.n+2, self.n+3], [self.n, self.n+3]])
        maxvolume = area / self.resolution + np.random.normal(0, area / (5 * self.resolution))
        mesh = build(mesh_info, max_volume=maxvolume)

        return mesh


# Custom simulation step, which at each point of time calculates and
# stores grid structure and vertex and face attribute

def triang_step(sim):
    global dx, cell_vec
    """Collect field samples, curls, and time-derivatives at every mesh point and edge midpoint."""

    def clamp_to_cell(xy):
        """Clamp a point to the simulation cell to avoid sampling outside bounds."""
        x, y = xy
        lx = -0.5 * float(cell_vec.x)
        ux = 0.5 * float(cell_vec.x)
        ly = -0.5 * float(cell_vec.y)
        uy = 0.5 * float(cell_vec.y)
        return max(lx, min(ux, x)), max(ly, min(uy, y))

    def get_field(comp, x, y):
        px, py = clamp_to_cell((x, y))
        return sim.get_field_point(comp, mp.Vector3(px, py, 0)).real

    current_t = sim.meep_time()  # precise simulation time for this callback (per snapshot scalar)

    # assigning field values, curls, time-derivatives, and epsilon as point attributes
    Points = []
    for i in range(len(mesh.points)):
        px, py = mesh.points[i][0], mesh.points[i][1]
        pt = mp.Vector3(px, py, 0)

        ez = sim.get_field_point(mp.Ez, pt).real
        eps_val = sim.get_epsilon_point(pt).real
        hx = sim.get_field_point(mp.Hx, pt).real
        hy = sim.get_field_point(mp.Hy, pt).real

        mesh.point_attributes[i, 0] = ez
        mesh.point_attributes[i, 1] = eps_val
        mesh.point_attributes[i, 2] = hx
        mesh.point_attributes[i, 3] = hy
        Points.append((px, py))

    mesh_attributes = np.array(mesh.point_attributes)

    # assigning field on faces (edge midpoints)
    # edge features: edge vector (2), Hx, Hy, curlH_z
    mesh_face_attr = np.zeros((mesh_face_num, 5))
    for i in range(mesh_face_num):
        if mesh_faces[i][0] == mesh_faces[i][1]:
            mesh_face_attr[i, :] = np.zeros((1, 5))
        else:
            vertex1 = np.array(mesh.points[mesh_faces[i][0]])
            vertex2 = np.array(mesh.points[mesh_faces[i][1]])
            middle = (vertex1 + vertex2) / 2
            mesh_face_attr[i, :2] = vertex1 - vertex2
            mid_pt = mp.Vector3(float(middle[0]), float(middle[1]), 0)
            hx_mid = sim.get_field_point(mp.Hx, mid_pt).real
            hy_mid = sim.get_field_point(mp.Hy, mid_pt).real
            mesh_face_attr[i, 2] = hx_mid
            mesh_face_attr[i, 3] = hy_mid

    # Record a scalar Time for each snapshot to avoid repeating t in each point attribute
    local_data = {
        'PointAttributes': mesh_attributes,
        'Points': Points,
        'FaceAttributes': mesh_face_attr,
        'Time': float(current_t),
    }
    AttributeData.append(local_data)


def generate_dataset(args):
    """Run the dataset generation loop using parsed CLI `args`.

    This sets module-level variables expected by `triang_step` and
    performs mesh generation, scatterer creation, optional saving, and
    Meep simulation runs.
    """
    global mesh, mesh_faces, mesh_face_num, AttributeData, cell_vec, dx

    if args.file_generation:
        os.makedirs(args.out_folder, exist_ok=True)

    for i in range(args.start, args.end):
        AttributeData = []
        # 1. Randomly select FDTD resolution
        resolution = random.randint(args.res_min, args.res_max)
        # 2. Dynamic mesh resolution, satisfying mesh_resolution ~ (FDTD_resolution)^2
        dynamic_mesh_res = int((resolution ** 2) / 4.0)
        mesh = GenerateRandomMesh((-1, -1), (1, 1), 10, resolution=dynamic_mesh_res).generate()
        points = np.array(mesh.points)  # (N, 2)
        mesh_elements = np.array(mesh.elements)
        mesh_triangles = [Triangle(*mesh_elements[j, :]) for j in range(len(mesh.elements))]
        mesh_faces = Triangle.collect_edges(mesh_triangles, bidirectional=True)
        mesh_face_num = len(mesh_faces)

        cell = mp.Vector3(args.cell_x, args.cell_y, 0)
        cell_vec = cell

        permittivity = random.uniform(args.perm_min, args.perm_max)
        frequency = random.uniform(args.frequency_min, args.frequency_max)
        time = args.time

        center_x = random.uniform(-0.5, 0.5)
        center_y = random.uniform(-0.5, 0.5)
        source_original = (float(center_x), float(center_y))

        nx = args.nx
        ny = args.ny
        sigma = args.sigma
        threshold = args.threshold
        seed_for_shape = args.seed if args.seed is not None else i
        geometry, mask = generate_scatterer_geometry(cell=cell, nx=nx, ny=ny,
                                                     sigma=sigma, threshold=threshold,
                                                     seed=seed_for_shape, epsilon=permittivity,
                                                     return_mask=True)

        # --- Main simulation object (for epsilon and fields) ---
        pulse_fwidth = getattr(args, 'pulse_fwidth', 1.0)
        source_type = getattr(args, 'source_type', 'point')
        if source_type == 'point':
            sources = [mp.Source(mp.GaussianSource(frequency=float(frequency), fwidth=float(pulse_fwidth)), component=mp.Ez, center=mp.Vector3(float(center_x), float(center_y)))]
        else:
            plane_x = getattr(args, 'plane_x', None)
            if plane_x is None:
                plane_x = -0.9
            sources = [mp.Source(mp.GaussianSource(frequency=float(frequency), fwidth=float(pulse_fwidth)), component=mp.Ez, center=mp.Vector3(float(plane_x), 0), size=mp.Vector3(0, float(cell.y), 0))]
        sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.5)], geometry=geometry, sources=sources, resolution=resolution)
        sim.init_sim()

        # --- Optimization 1: Use main simulation object to extract epsilon ---
        eps_data = np.zeros(len(points), dtype=np.float32)
        for j, p in enumerate(points):
            eps_data[j] = sim.get_epsilon_point(mp.Vector3(p[0], p[1])).real

        # --- Save static data ---
        MeshData = {
            'coords': points,           # (N, 2)
            'edges': mesh_faces,        # (E, 2)
            'GraphStructure': mesh_faces, # (E, 2) compatible with loading script
            'epsilon': eps_data,        # (N,)
            'source_pos': source_original,
            'resolution': resolution,
            'mesh_resolution': dynamic_mesh_res
        }
        if args.file_generation:
            file_path = os.path.join(args.out_folder, f'MeshData_{i}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(MeshData, f)

        # --- Save mask ---
        if args.file_generation:
            masks_dir = os.path.join(args.out_folder, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            mask_name = f'mask_{i}_nx{nx}_ny{ny}_s{int(sigma)}_t{int(threshold*100)}_seed{seed_for_shape}.npz'
            mask_path = os.path.join(masks_dir, mask_name)
            np.savez_compressed(mask_path,
                                mask=mask.astype(np.uint8),
                                nx=nx, ny=ny, sigma=sigma,
                                threshold=threshold, seed=seed_for_shape,
                                epsilon=permittivity,
                                cell_x=2.0, cell_y=2.0,
                                extent=[-1,1,-1,1])

        # --- Run simulation and collect dynamic fields ---
        # Sampling step size set to 0.01, cutoff=3.0, t0=1.5, total_time=5, theoretical sampling steps approx 350
        fixed_snapshot_dt = 0.01
        sim.run(mp.at_every(fixed_snapshot_dt, triang_step), until=time)

        # --- Time synchronization correction (Cut off the initial segment of snapshots) ---
        try:
            start_time = float(getattr(args, 'start_time', 0.0))
            cutoff = float(getattr(args, 'cutoff', 5.0))
            fwidth = float(getattr(args, 'pulse_fwidth', pulse_fwidth))
            t0 = start_time + 2.0 * (cutoff / max(1e-12, fwidth))
            n_skip = 0
            for idx, snap in enumerate(AttributeData):
                if snap['Time'] < t0:
                    n_skip += 1
                else:
                    break
            if n_skip > 0 and len(AttributeData) > 0:
                AttributeData = AttributeData[n_skip:]
        except Exception:
            pass

        # --- Save dynamic data (full physical quantity snapshots) ---
        if args.file_generation:
            file_path = os.path.join(args.out_folder, f'MeshAttributes_{i}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(AttributeData, f)

# create a folder to store the data

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description='Generate dataset with filtered-noise scatterers')
    p.add_argument('--start', type=int, default=100, help='Start index (inclusive)')
    p.add_argument('--end', type=int, default=150, help='End index (exclusive)')
        # --num-steps was unused and has been removed to avoid confusion
    p.add_argument('--out-folder', type=str, default='./dataset/data_re40-80_fr1/', help='Output folder for dataset')
    p.add_argument('--file-generation', type=bool, default=True, help='Enable writing files to disk')
    p.add_argument('--mesh-resolution', type=int, default=1000, help='Mesh generation resolution')

    # scatterer / mask generation params
    p.add_argument('--nx', type=int, default=160, help='Mask grid NX')
    p.add_argument('--ny', type=int, default=160, help='Mask grid NY')
    p.add_argument('--sigma', type=float, default=6.0, help='Gaussian blur sigma for mask')
    p.add_argument('--threshold', type=float, default=0.75, help='Mask threshold (0..1)')
    p.add_argument('--cell-x', type=float, default=3.0, help='Physical cell x size (includes PML)')
    p.add_argument('--cell-y', type=float, default=3.0, help='Physical cell y size (includes PML)')

    # permittivity / source / sim params
    p.add_argument('--perm-min', type=float, default=2.0, help='Permittivity random min')
    p.add_argument('--perm-max', type=float, default=10.0, help='Permittivity random max')
    p.add_argument('--frequency-min', type=float, default=1, help='Frequency random min')
    p.add_argument('--frequency-max', type=float, default=1, help='Frequency random max')
    p.add_argument('--time', type=float, default=5.0, help='Simulation time (until)')
    p.add_argument('--res-min', type=int, default=60, help='Min Meep resolution')
    p.add_argument('--res-max', type=int, default=60, help='Max Meep resolution')

    # pulsed source options (for generated simulations)
    p.add_argument('--source-type', choices=['point', 'plane'], default='point', help='Source type for dataset sims (point or plane)')
    p.add_argument('--pulse-fwidth', type=float, default=4.0, help='Pulse fwidth (spectral width) for GaussianSource')
    p.add_argument('--plane-x', type=float, default=None, help='Absolute x coordinate for plane source if using plane source')
    p.add_argument('--start-time', type=float, default=0.0, help='Start time offset (start_time) used when computing GIF start t0')
    p.add_argument('--cutoff', type=float, default=3.0, help='Cutoff multiplier used to compute t0 = start_time + 2*(cutoff / fwidth); default=3.0, t0=1.5 when fwidth=4.0')

    # seed handling
    p.add_argument('--seed', type=int, default=None, help='Fixed seed for all sims (if provided)')

    args = p.parse_args()

    # delegate main work to function
    generate_dataset(args)
# python generate_data_v2.py --out-folder ./dataset/point_re60_fr1-2 --frequency-max 2