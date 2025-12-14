"""
Example usage of the geometry module.

This script demonstrates the main features of the geometry module:
- Downloading country and NUTS-3 shapes
- Joining shapes
- Point-in-polygon checks
- Intersection operations
"""

from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from geometry import (
    download_country_shapes,
    download_nuts3_shapes,
    join_shapes,
    point_in_shape,
    mask_shape,
    get_shape_area,
    buffer_shape,
    simplify_shape,
    get_european_union_shape,
)


def example_1_download_and_join():
    """Example 1: Download countries and join them."""
    print("=" * 60)
    print("Example 1: Download and Join Country Shapes")
    print("=" * 60)
    
    # Download Germany and Poland
    print("\nDownloading Germany and Poland...")
    countries = download_country_shapes(['DE', 'PL'])
    print(f"Downloaded {len(countries)} countries")
    
    # Join them into a single shape
    print("\nJoining shapes...")
    combined = join_shapes(countries)
    area = get_shape_area(combined)
    print(f"Combined area: {area:,.0f} km²")
    
    return combined


def example_2_point_checks():
    """Example 2: Check if points are within boundaries."""
    print("\n" + "=" * 60)
    print("Example 2: Point-in-Polygon Checks")
    print("=" * 60)
    
    # Download Germany
    print("\nDownloading Germany...")
    germany = download_country_shapes(['DE'])
    
    # Test various cities
    cities = {
        'Berlin': (52.5200, 13.4050),
        'Munich': (48.1351, 11.5820),
        'Paris': (48.8566, 2.3522),
        'Warsaw': (52.2297, 21.0122),
    }
    
    print("\nChecking if cities are in Germany:")
    for city, (lat, lon) in cities.items():
        is_in = point_in_shape(lat, lon, germany)
        print(f"  {city:10s}: {is_in}")


def example_3_nuts_regions():
    """Example 3: Work with NUTS-3 regions."""
    print("\n" + "=" * 60)
    print("Example 3: NUTS-3 Regions")
    print("=" * 60)
    
    # Download NUTS-3 for Germany
    print("\nDownloading NUTS-3 regions for Germany...")
    nuts3_de = download_nuts3_shapes(['DE'])
    print(f"Downloaded {len(nuts3_de)} regions")
    
    # Show some region names
    print("\nSample NUTS-3 regions:")
    for idx, row in nuts3_de.head(5).iterrows():
        region_id = row.get('NUTS_ID', idx)
        region_name = row.get('NAME_LATN', 'Unknown')
        print(f"  {region_id}: {region_name}")
    
    # Find which NUTS-3 region contains Berlin
    berlin = (52.5200, 13.4050)
    print(f"\nFinding NUTS-3 region for Berlin ({berlin[0]}, {berlin[1]})...")
    
    for idx, row in nuts3_de.iterrows():
        if point_in_shape(berlin[0], berlin[1], row['geometry']):
            region_id = row.get('NUTS_ID', idx)
            region_name = row.get('NAME_LATN', 'Unknown')
            print(f"  Berlin is in: {region_id} ({region_name})")
            break


def example_4_intersection():
    """Example 4: Compute intersection of boundaries."""
    print("\n" + "=" * 60)
    print("Example 4: Boundary Intersection")
    print("=" * 60)
    
    # Download two countries
    print("\nDownloading France and Germany...")
    france = download_country_shapes(['FR'])
    germany = download_country_shapes(['DE'])
    
    france_area = get_shape_area(france)
    germany_area = get_shape_area(germany)
    
    print(f"France area:  {france_area:,.0f} km²")
    print(f"Germany area: {germany_area:,.0f} km²")
    
    # Note: Countries don't actually overlap, but this demonstrates the function
    print("\nComputing intersection (should be empty for non-overlapping countries)...")
    intersection = mask_shape(france, germany)
    
    if intersection.is_empty:
        print("  No intersection (as expected - countries don't overlap)")
    else:
        intersection_area = get_shape_area(intersection)
        print(f"  Intersection area: {intersection_area:,.0f} km²")


def example_5_buffer():
    """Example 5: Create buffers around shapes."""
    print("\n" + "=" * 60)
    print("Example 5: Buffer Operations")
    print("=" * 60)
    
    # Download Luxembourg (small country)
    print("\nDownloading Luxembourg...")
    luxembourg = download_country_shapes(['LU'])
    
    original_area = get_shape_area(luxembourg)
    print(f"Original area: {original_area:,.0f} km²")
    
    # Create 50km buffer
    print("\nCreating 50km buffer around Luxembourg...")
    buffered = buffer_shape(luxembourg, distance_km=50)
    buffered_area = get_shape_area(buffered)
    
    print(f"Buffered area: {buffered_area:,.0f} km²")
    print(f"Area increase: {buffered_area - original_area:,.0f} km²")


def example_6_eu_shape():
    """Example 6: Get European Union shape."""
    print("\n" + "=" * 60)
    print("Example 6: European Union Boundary")
    print("=" * 60)
    
    print("\nDownloading and joining all EU member states...")
    eu = get_european_union_shape()
    
    eu_area = get_shape_area(eu)
    print(f"Total EU area: {eu_area:,.0f} km²")
    
    # Check capital cities
    capitals = {
        'Brussels (Belgium)': (50.8503, 4.3517),
        'Berlin (Germany)': (52.5200, 13.4050),
        'London (UK - not EU)': (51.5074, -0.1278),
        'Oslo (Norway - not EU)': (59.9139, 10.7522),
    }
    
    print("\nChecking if capital cities are in EU:")
    for city, (lat, lon) in capitals.items():
        is_in = point_in_shape(lat, lon, eu)
        print(f"  {city:25s}: {is_in}")


def example_7_simplification():
    """Example 7: Simplify complex geometries."""
    print("\n" + "=" * 60)
    print("Example 7: Geometry Simplification")
    print("=" * 60)
    
    # Download a country with complex coastline
    print("\nDownloading Norway (complex coastline)...")
    norway = download_country_shapes(['NO'])
    
    # Count vertices (approximate)
    original_geom = norway.geometry.iloc[0]
    if hasattr(original_geom, 'exterior'):
        original_vertices = len(original_geom.exterior.coords)
    else:
        # MultiPolygon
        original_vertices = sum(len(geom.exterior.coords) for geom in original_geom.geoms if hasattr(geom, 'exterior'))
    
    print(f"Original vertices: {original_vertices:,}")
    
    # Simplify
    print("\nSimplifying with tolerance=0.01 (~1km)...")
    simplified = simplify_shape(norway, tolerance=0.01)
    
    if hasattr(simplified, 'exterior'):
        simplified_vertices = len(simplified.exterior.coords)
    else:
        simplified_vertices = sum(len(geom.exterior.coords) for geom in simplified.geoms if hasattr(geom, 'exterior'))
    
    print(f"Simplified vertices: {simplified_vertices:,}")
    print(f"Reduction: {100 * (1 - simplified_vertices/original_vertices):.1f}%")
    
    # Compare areas
    original_area = get_shape_area(norway)
    simplified_area = get_shape_area(simplified)
    area_diff_pct = 100 * abs(simplified_area - original_area) / original_area
    
    print(f"\nArea difference: {area_diff_pct:.2f}% (should be minimal)")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GEOMETRY MODULE EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates the main features of the geometry module.")
    print("Data will be cached locally after first download.\n")
    
    try:
        # Run examples
        example_1_download_and_join()
        example_2_point_checks()
        example_3_nuts_regions()
        example_4_intersection()
        example_5_buffer()
        example_6_eu_shape()
        example_7_simplification()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
