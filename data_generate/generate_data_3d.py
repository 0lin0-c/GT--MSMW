import meep as mp
import numpy as np
import pickle, os
import random
from scipy.spatial import Delaunay
import argparse

# Import the previous generation module
# Assume the utils.scatterer file is in the corresponding directory
from data_generate.scatterer import generate_scatterer_geometry_3d

# Global variables
mesh_points = None
mesh_edges = None
mesh_edge_num = 0
AttributeData = []
cell_vec = None

class GenerateRandomMesh3D:
    def __init__(self, bounds_min, bounds_max, n_points):
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.n = n_points

    def generate(self):
        points = np.random.uniform(self.bounds_min, self.bounds_max, (self.n, 3))
        # Add corner points to ensure the convex hull is complete
        corners = np.array([
            [self.bounds_min[0], self.bounds_min[1], self.bounds_min[2]],
            [self.bounds_max[0], self.bounds_min[1], self.bounds_min[2]],
            [self.bounds_min[0], self.bounds_max[1], self.bounds_min[2]],
            [self.bounds_max[0], self.bounds_max[1], self.bounds_min[2]],
            [self.bounds_min[0], self.bounds_min[1], self.bounds_max[2]],
            [self.bounds_max[0], self.bounds_min[1], self.bounds_max[2]],
            [self.bounds_min[0], self.bounds_max[1], self.bounds_max[2]],
            [self.bounds_max[0], self.bounds_max[1], self.bounds_max[2]],
        ])
        points = np.vstack([points, corners])
        tri = Delaunay(points)
        return points, tri

def get_edges_from_delaunay_3d(tri_obj):
    edges = set()
    for simplex in tri_obj.simplices:
        simplex.sort()
        edges.add((simplex[0], simplex[1]))
        edges.add((simplex[0], simplex[2]))
        edges.add((simplex[0], simplex[3]))
        edges.add((simplex[1], simplex[2]))
        edges.add((simplex[1], simplex[3]))
        edges.add((simplex[2], simplex[3]))
    return list(edges)

def triang_step_3d(sim):
    global mesh_points, mesh_edges, mesh_edge_num, AttributeData
    current_t = sim.meep_time()

    # 1. Collect node attributes
    N = len(mesh_points)
    point_attr = np.zeros((N, 7), dtype=np.float32)
    
    # mesh_points coordinates here are based on the physical region [-1, 1]
    # Since the Simulation is center-aligned (0,0,0), coordinates correspond directly without offset
    for i in range(N):
        pt = mp.Vector3(mesh_points[i][0], mesh_points[i][1], mesh_points[i][2])
        ex = sim.get_field_point(mp.Ex, pt).real
        ey = sim.get_field_point(mp.Ey, pt).real
        ez = sim.get_field_point(mp.Ez, pt).real
        hx = sim.get_field_point(mp.Hx, pt).real
        hy = sim.get_field_point(mp.Hy, pt).real
        hz = sim.get_field_point(mp.Hz, pt).real
        eps = sim.get_epsilon_point(pt).real
        point_attr[i, :] = [ex, ey, ez, hx, hy, hz, eps]

    # 2. Collect edge attributes
    edge_attr = np.zeros((mesh_edge_num, 6), dtype=np.float32)
    for i in range(mesh_edge_num):
        idx_src, idx_dst = mesh_edges[i]
        p_src = mesh_points[idx_src]
        p_dst = mesh_points[idx_dst]
        diff = p_src - p_dst
        mid = (p_src + p_dst) / 2.0
        mid_pt = mp.Vector3(mid[0], mid[1], mid[2])
        
        hx_mid = sim.get_field_point(mp.Hx, mid_pt).real
        hy_mid = sim.get_field_point(mp.Hy, mid_pt).real
        hz_mid = sim.get_field_point(mp.Hz, mid_pt).real
        edge_attr[i, :] = [diff[0], diff[1], diff[2], hx_mid, hy_mid, hz_mid]

    local_data = {
        'PointAttributes': point_attr, 
        'Points': mesh_points,        
        'FaceAttributes': edge_attr,   
        'Time': float(current_t),
    }
    AttributeData.append(local_data)

def generate_dataset(args):
    global mesh_points, mesh_edges, mesh_edge_num, AttributeData, cell_vec

    if args.file_generation:
        os.makedirs(args.out_folder, exist_ok=True)

    # --- Key modification: Define physical dimensions and simulation dimensions ---
    pml_thickness = 0.5
    
    # Physical Region of Interest (ROI), e.g., [-1, 1] -> size=2.0
    phys_size_x = args.cell_x
    phys_size_y = args.cell_y
    phys_size_z = args.cell_z
    phys_cell = mp.Vector3(phys_size_x, phys_size_y, phys_size_z)
    
    # Simulation total size = physical size + 2 * PML -> size=3.0
    sim_size_x = phys_size_x + 2 * pml_thickness
    sim_size_y = phys_size_y + 2 * pml_thickness
    sim_size_z = phys_size_z + 2 * pml_thickness
    sim_cell = mp.Vector3(sim_size_x, sim_size_y, sim_size_z)

    # Save cell vector for Dataset (usually we only care about the physical region)
    cell_vec = phys_cell

    for i in range(args.start, args.end):
        AttributeData = []
        print(f"\n--- Generating Simulation {i} (Source: {args.source_type}) ---")
        
        # ==============================================================
        # [Core remains unchanged] Automatically compute GNN points - still based on physical region bounds
        # ==============================================================
        bounds_half_x = phys_size_x / 2.0
        bounds_half_y = phys_size_y / 2.0
        bounds_half_z = phys_size_z / 2.0
        
        bounds_min = np.array([-bounds_half_x, -bounds_half_y, -bounds_half_z])
        bounds_max = np.array([bounds_half_x, bounds_half_y, bounds_half_z])
        
        # Calculate volume (based on args.cell, i.e., physical volume)
        dims = bounds_max - bounds_min
        volume = np.prod(dims) # 2*2*2 = 8.0
        
        calculated_n_points = int(volume * (args.resolution ** 3) * args.gnn_density)
        calculated_n_points = max(calculated_n_points, 50)
        
        print(f"   Physical Volume={volume:.1f} (in [-1,1]), Resolution={args.resolution}")
        print(f"   -> GNN Nodes: {calculated_n_points} (Sampled strictly inside ROI)")

        # Generate mesh - only inside the physical region
        mesh_gen = GenerateRandomMesh3D(bounds_min, bounds_max, calculated_n_points)
        mesh_points, delaunay_tri = mesh_gen.generate()
        
        mesh_edges = get_edges_from_delaunay_3d(delaunay_tri)
        mesh_edge_num = len(mesh_edges)
        # ==============================================================

        # Random parameter generation
        permittivity = random.uniform(args.perm_min, args.perm_max)
        frequency = random.uniform(args.frequency_min, args.frequency_max)
        
        # Generate scatterer shape
        # Note: Pass phys_cell here so the mask corresponds to the [-1, 1] region
        mask_res = int(args.resolution * 3)
        seed_for_shape = args.seed if args.seed else i
        
        geometry, mask = generate_scatterer_geometry_3d(
            cell=phys_cell, 
            nx=mask_res, ny=mask_res, nz=mask_res,
            sigma=args.sigma, threshold=args.threshold,
            seed=seed_for_shape,
            epsilon=permittivity, return_mask=True
        )

        # Set up source
        fwidth = args.pulse_fwidth
        sources = []
        src_pos_record = (0, 0, 0) # Used for recording

        if args.source_type == 'point':
            # Point source: Random within physical region [-0.5*cell, 0.5*cell]
            # Coefficient 0.4 shrinks slightly to ensure source isn't right on the physical boundary
            center_x = random.uniform(-0.4*phys_size_x, 0.4*phys_size_x)
            center_y = random.uniform(-0.4*phys_size_y, 0.4*phys_size_y)
            center_z = random.uniform(-0.4*phys_size_z, 0.4*phys_size_z)
            
            sources = [mp.Source(
                mp.GaussianSource(frequency=float(frequency), fwidth=fwidth),
                component=mp.Ex, 
                center=mp.Vector3(center_x, center_y, center_z)
            )]
            src_pos_record = (center_x, center_y, center_z)
            
        elif args.source_type == 'plane':
            # Plane wave:
            # 1. Position: Bottom of physical region (-cell_z/2) + buffer (0.2)
            #    This places the source inside the physical region, and away from the PML behind it
            z_pos = -phys_size_z / 2.0 + 0.2
            
            sources = [mp.Source(
                mp.GaussianSource(frequency=float(frequency), fwidth=fwidth),
                component=mp.Ex, 
                center=mp.Vector3(0, 0, z_pos),
                size=mp.Vector3(phys_size_x, phys_size_y, 0) # Cover physical cross-section
            )]
            src_pos_record = (0, 0, z_pos)

        # Initialize simulation
        # Key: cell_size uses sim_cell (large box including external PML)
        # geometry is automatically placed at center (i.e., inside physical region)
        sim = mp.Simulation(
            cell_size=sim_cell,
            boundary_layers=[mp.PML(pml_thickness)],
            geometry=geometry,
            sources=sources,
            resolution=args.resolution
        )
        sim.init_sim()
        
        # Save static data (epsilon sampling)
        eps_data = np.zeros(len(mesh_points), dtype=np.float32)
        for j, p in enumerate(mesh_points):
            # p here is physical coordinate, sim.get_epsilon will map correctly
            eps_data[j] = sim.get_epsilon_point(mp.Vector3(p[0], p[1], p[2])).real
            
        MeshData = {
            'coords': mesh_points,
            'edges': mesh_edges,
            'GraphStructure': mesh_edges,
            'epsilon': eps_data,
            'source_pos': src_pos_record,
            'source_type': args.source_type, # Record source type
            'resolution': args.resolution,
            'frequency': frequency,
            'phys_cell_size': (phys_size_x, phys_size_y, phys_size_z)
        }
        
        if args.file_generation:
            with open(os.path.join(args.out_folder, f'MeshData_{i}.pkl'), 'wb') as f:
                pickle.dump(MeshData, f)
            
            # Save Mask (corresponding to physical region)
            masks_dir = os.path.join(args.out_folder, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(masks_dir, f'mask_{i}.npz'),
                mask=mask.astype(np.uint8),
                epsilon=permittivity,
                nx=mask_res, ny=mask_res, nz=mask_res
            )

        # Run simulation
        snapshot_dt = getattr(args, 'snapshot_dt', 0.1)
        sim.run(mp.at_every(snapshot_dt, triang_step_3d), until=args.time)
        
        # Time Cutoff processing
        try:
            start_time = float(getattr(args, 'start_time', 0.0))
            cutoff = float(getattr(args, 'cutoff', 3.0))
            t0 = start_time + 2.0 * (cutoff / max(1e-12, fwidth))
            n_skip = 0
            for idx, snap in enumerate(AttributeData):
                if snap['Time'] < t0:
                    n_skip += 1
                else:
                    break
            if n_skip > 0 and len(AttributeData) > 0:
                print(f"   Time Cutoff applied: Skipping first {n_skip} frames (t < {t0:.4f})")
                AttributeData = AttributeData[n_skip:]
        except Exception as e:
            print(f"Warning: Time cutoff logic failed: {e}")

        if args.file_generation:
            with open(os.path.join(args.out_folder, f'MeshAttributes_{i}.pkl'), 'wb') as f:
                pickle.dump(AttributeData, f)
        
        print(f"Finished simulation {i}. Saved {len(AttributeData)} frames.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Generate 3D GNN Dataset (PML Outside)')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=100)
    p.add_argument('--out-folder', type=str, default='./dataset/point_res15_freq1_3d/')
    p.add_argument('--file-generation', type=bool, default=True)
    
    # Core configuration
    p.add_argument('--resolution', type=int, default=15, 
                   help='Unified resolution for FDTD and GNN density')
    p.add_argument('--gnn-density', type=float, default=0.05, 
                   help='Ratio of GNN points relative to FDTD voxels')
    
    # Physical region dimensions (excluding PML)
    p.add_argument('--cell-x', type=float, default=2.0)
    p.add_argument('--cell-y', type=float, default=2.0)
    p.add_argument('--cell-z', type=float, default=2.0)
    
    # Source configuration
    p.add_argument('--source-type', type=str, default='point', choices=['point', 'plane'],
                   help='Source type: point or plane')
    
    # Simulation parameters
    p.add_argument('--time', type=float, default=10.0)
    p.add_argument('--snapshot-dt', type=float, default=0.05)
    
    # Scatterer parameters
    p.add_argument('--sigma', type=float, default=3.0)
    p.add_argument('--threshold', type=float, default=0.75)
    p.add_argument('--perm-min', type=float, default=2.0)
    p.add_argument('--perm-max', type=float, default=5.0)
    p.add_argument('--frequency-min', type=float, default=1.0)
    p.add_argument('--frequency-max', type=float, default=1.0)
    p.add_argument('--pulse-fwidth', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--start-time', type=float, default=0.0)
    p.add_argument('--cutoff', type=float, default=3.0)

    args = p.parse_args()
    generate_dataset(args)

'''
python -m generate_data_3d \
    --source-type plane \
    --out-folder dataset/plane_res15_freq1_3d/
'''