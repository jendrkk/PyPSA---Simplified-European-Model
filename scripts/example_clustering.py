"""
Example: Network Simplification and Clustering

This script demonstrates how to use the network_clust module
to simplify and cluster a PyPSA network.

Run with:
    python scripts/example_clustering.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pypsa

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from scripts.network_clust import (
    simplify_and_cluster_network,
    distribute_n_clusters_to_countries,
    get_final_busmap,
)
from src.pypsa_simplified.network import build_network
from src.pypsa_simplified.data_prep import RawData


def example_1_basic_simplification():
    """
    Example 1: Basic network simplification without clustering.
    
    Steps:
    1. Load network
    2. Simplify to 380kV
    3. Remove converters
    4. Remove stubs
    5. Save simplified network
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Simplification (No Clustering)")
    print("="*70)
    
    # Load your network (replace with actual path)
    print("\n1. Loading network...")
    # For this example, we'll assume you have a network
    # n = pypsa.Network("data/networks/base.nc")
    print("   [Replace with your network loading code]")
    
    # Simplify only
    print("\n2. Simplifying network...")
    # n_simple, busmaps = simplify_and_cluster_network(
    #     n,
    #     n_clusters=None,  # No clustering
    #     remove_stubs_before=True,
    #     remove_stubs_matching=['country'],
    # )
    
    print("\n3. Network simplified!")
    print("   [Save with: n_simple.export_to_netcdf('network_simplified.nc')]")


def example_2_clustering_with_kmeans():
    """
    Example 2: Clustering with k-means algorithm.
    
    Steps:
    1. Load network and load data
    2. Simplify and cluster to 50 buses
    3. Save clustered network and busmaps
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Clustering with K-Means")
    print("="*70)
    
    # Load network
    print("\n1. Loading network and load data...")
    # n = pypsa.Network("data/networks/base.nc")
    # load = pd.read_csv("data/processed/load_per_bus.csv", index_col=0, squeeze=True)
    print("   [Replace with your network and load data]")
    
    # For demonstration, create synthetic load
    # load = pd.Series(
    #     np.random.rand(len(n.buses)) * 1000,
    #     index=n.buses.index
    # )
    
    print("\n2. Clustering to 50 buses with k-means...")
    # n_clustered, busmaps = simplify_and_cluster_network(
    #     n,
    #     n_clusters=50,
    #     cluster_weights=load,
    #     algorithm='kmeans',
    #     solver_name='gurobi',  # or 'scip'
    #     remove_stubs_before=True,
    #     random_state=42,  # Reproducibility
    # )
    
    print("\n3. Saving results...")
    # n_clustered.export_to_netcdf("data/networks/network_50.nc")
    
    # Save busmap
    # final_busmap = get_final_busmap(busmaps)
    # final_busmap.to_csv("data/resources/busmap_to_50.csv")
    
    print("\n4. Clustering complete!")
    print(f"   Original buses: [N_ORIGINAL]")
    print(f"   Clustered buses: 50")
    print(f"   Reduction: [PERCENTAGE]%")


def example_3_clustering_with_hac():
    """
    Example 3: Topology-aware clustering with HAC.
    
    Steps:
    1. Load network and create feature matrix
    2. Cluster with HAC algorithm
    3. Compare with k-means results
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Clustering with HAC (Topology-Aware)")
    print("="*70)
    
    print("\n1. Loading network...")
    # n = pypsa.Network("data/networks/base.nc")
    # load = pd.read_csv("data/processed/load_per_bus.csv", index_col=0, squeeze=True)
    
    print("\n2. Creating feature matrix...")
    print("   Options:")
    print("   a) From renewable profiles")
    print("   b) From load patterns")
    print("   c) From geographic/network features")
    
    # Example: Create simple feature matrix from bus properties
    # features = pd.DataFrame({
    #     'x_coord': n.buses.x,
    #     'y_coord': n.buses.y,
    #     'v_nom': n.buses.v_nom,
    # })
    
    # Or from profiles (better approach)
    # import xarray as xr
    # profiles = xr.open_dataset("data/processed/renewable_profiles.nc")
    # features = pd.concat([
    #     profiles['solar'].to_pandas().T,
    #     profiles['wind'].to_pandas().T,
    # ], axis=1).fillna(0.0)
    
    print("\n3. Clustering with HAC...")
    # n_clustered, busmaps = simplify_and_cluster_network(
    #     n,
    #     n_clusters=50,
    #     cluster_weights=load,
    #     algorithm='hac',
    #     features=features,
    #     solver_name='gurobi',
    #     linkage='ward',
    #     affinity='euclidean',
    # )
    
    print("\n4. HAC clustering complete!")
    print("   HAC considers network topology and feature similarity")
    print("   Results in electrically coherent clusters")


def example_4_focused_clustering():
    """
    Example 4: Focused clustering with regional emphasis.
    
    Steps:
    1. Define focus weights for countries
    2. Cluster with emphasis on specific regions
    3. Analyze cluster distribution
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Focused Clustering (Regional Emphasis)")
    print("="*70)
    
    print("\n1. Setting up focus weights...")
    focus_weights = {
        'DE': 2.0,   # Germany: double clusters
        'FR': 1.5,   # France: 50% more clusters
        'IT': 1.3,   # Italy: 30% more clusters
    }
    print(f"   Focus weights: {focus_weights}")
    
    print("\n2. Clustering with focus weights...")
    # n_clustered, busmaps = simplify_and_cluster_network(
    #     n,
    #     n_clusters=100,
    #     cluster_weights=load,
    #     algorithm='kmeans',
    #     solver_name='gurobi',
    #     focus_weights=focus_weights,
    # )
    
    print("\n3. Analyzing cluster distribution...")
    # for country in ['DE', 'FR', 'IT']:
    #     mask = n_clustered.buses.country == country
    #     n_buses = mask.sum()
    #     print(f"   {country}: {n_buses} buses")


def example_5_gurobi_optimization():
    """
    Example 5: Demonstrating Gurobi cluster distribution.
    
    Shows how the optimization problem is formulated and solved.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Gurobi Optimization for Cluster Distribution")
    print("="*70)
    
    print("\n1. Problem Formulation:")
    print("   Minimize: Σ (n_c - L_c * N)²")
    print("   Subject to: Σ n_c = N")
    print("               1 <= n_c <= N_c")
    print("")
    print("   Where:")
    print("   - n_c: clusters in country c")
    print("   - L_c: normalized weight (load share)")
    print("   - N: total clusters")
    print("   - N_c: buses in country c")
    
    print("\n2. Example with synthetic data:")
    # Create synthetic network data
    countries = ['DE', 'FR', 'ES', 'IT', 'PL']
    n_buses = {'DE': 150, 'FR': 120, 'ES': 80, 'IT': 90, 'PL': 60}
    loads = {'DE': 500, 'FR': 400, 'ES': 250, 'IT': 300, 'PL': 150}
    
    print(f"\n   Countries: {countries}")
    print(f"   Buses: {n_buses}")
    print(f"   Loads: {loads}")
    
    # Calculate proportional allocation
    total_load = sum(loads.values())
    n_clusters = 50
    
    print(f"\n3. Proportional allocation (baseline):")
    for country in countries:
        prop = (loads[country] / total_load) * n_clusters
        print(f"   {country}: {prop:.1f} clusters")
    
    print(f"\n4. With Gurobi optimization:")
    print("   [Ensures integer values and respects constraints]")
    # In practice:
    # n_clusters_c = distribute_n_clusters_to_countries(
    #     n, n_clusters=50, cluster_weights=load, solver_name='gurobi'
    # )
    # print(n_clusters_c)


def example_6_full_workflow():
    """
    Example 6: Complete workflow from raw data to clustered network.
    
    This is the recommended workflow for production use.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete Production Workflow")
    print("="*70)
    
    print("\nStep-by-step workflow:")
    print("\n1. Load and prepare data")
    print("   - Load network from build_network()")
    print("   - Load demand data from demand.py")
    print("   - Prepare any features for HAC")
    
    print("\n2. Configure clustering parameters")
    config = {
        'n_clusters': 50,
        'algorithm': 'kmeans',
        'solver_name': 'gurobi',
        'remove_stubs_before': True,
        'remove_stubs_matching': ['country'],
        'focus_weights': {'DE': 1.5},
    }
    print(f"   Config: {config}")
    
    print("\n3. Run simplification and clustering")
    print("   n_clustered, busmaps = simplify_and_cluster_network(n, **config)")
    
    print("\n4. Aggregate demand to clustered buses")
    print("   - Use get_final_busmap(busmaps)")
    print("   - Group demand by new buses")
    print("   - Add to network with add_loads_from_series()")
    
    print("\n5. Add generators (if needed)")
    print("   - Aggregate generators to clustered buses")
    print("   - Or reload/reassign generation")
    
    print("\n6. Save outputs")
    print("   - Save network: n.export_to_netcdf()")
    print("   - Save busmaps: busmap.to_csv()")
    print("   - Save metadata: config to JSON")
    
    print("\n7. Validation")
    print("   - Check total capacity preserved")
    print("   - Verify bus count matches target")
    print("   - Test optimization (n.optimize())")


def print_menu():
    """Print example menu."""
    print("\n" + "="*70)
    print("NETWORK CLUSTERING EXAMPLES")
    print("="*70)
    print("\nAvailable examples:")
    print("  1. Basic simplification (no clustering)")
    print("  2. Clustering with k-means")
    print("  3. Clustering with HAC (topology-aware)")
    print("  4. Focused clustering (regional emphasis)")
    print("  5. Gurobi optimization demonstration")
    print("  6. Complete production workflow")
    print("  0. Exit")
    print("\nNote: Examples show structure and comments.")
    print("      Replace placeholder code with your actual data.")


def main():
    """Main function to run examples."""
    print("\nWelcome to Network Clustering Examples!")
    print("These examples demonstrate the network_clust module.")
    
    examples = {
        '1': example_1_basic_simplification,
        '2': example_2_clustering_with_kmeans,
        '3': example_3_clustering_with_hac,
        '4': example_4_focused_clustering,
        '5': example_5_gurobi_optimization,
        '6': example_6_full_workflow,
    }
    
    while True:
        print_menu()
        choice = input("\nSelect example (0-6): ").strip()
        
        if choice == '0':
            print("\nExiting. See scripts/docs/NETWORK_CLUSTERING.md for full documentation.")
            break
        elif choice in examples:
            try:
                examples[choice]()
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
            except Exception as e:
                print(f"\nError running example: {e}")
                import traceback
                traceback.print_exc()
            
            input("\nPress Enter to continue...")
        else:
            print("Invalid choice. Please select 0-6.")


if __name__ == "__main__":
    main()
