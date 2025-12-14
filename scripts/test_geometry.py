"""
Quick test script for the geometry module.

This runs a minimal set of tests to verify the module is working correctly.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all functions can be imported."""
    print("Testing imports...", end=" ")
    try:
        from geometry import (
            download_country_shapes,
            download_nuts3_shapes,
            join_shapes,
            point_in_shape,
            mask_shape,
            buffer_shape,
            get_shape_area,
            simplify_shape,
            get_european_union_shape,
        )
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_download_countries():
    """Test downloading country shapes."""
    print("Testing country download...", end=" ")
    try:
        from geometry import download_country_shapes
        
        # Download a small country
        luxembourg = download_country_shapes(['LU'])
        
        assert len(luxembourg) == 1, "Should have 1 country"
        assert luxembourg.geometry.iloc[0] is not None, "Should have geometry"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_point_in_shape():
    """Test point-in-polygon functionality."""
    print("Testing point-in-shape...", end=" ")
    try:
        from geometry import download_country_shapes, point_in_shape
        
        # Get Germany
        germany = download_country_shapes(['DE'])
        
        # Berlin should be in Germany
        berlin_in = point_in_shape(52.5200, 13.4050, germany)
        assert berlin_in == True, "Berlin should be in Germany"
        
        # Paris should NOT be in Germany
        paris_in = point_in_shape(48.8566, 2.3522, germany)
        assert paris_in == False, "Paris should not be in Germany"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_join_shapes():
    """Test joining shapes."""
    print("Testing join shapes...", end=" ")
    try:
        from geometry import download_country_shapes, join_shapes, get_shape_area
        
        # Download Benelux countries
        benelux = download_country_shapes(['BE', 'NL', 'LU'])
        
        # Join them
        combined = join_shapes(benelux)
        
        # Should have a geometry
        assert combined is not None, "Should have geometry"
        
        # Area should be reasonable (Benelux is about 75,000 km²)
        area = get_shape_area(combined)
        assert 60000 < area < 90000, f"Area should be ~75,000 km², got {area:.0f}"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_download_nuts3():
    """Test downloading NUTS-3 regions."""
    print("Testing NUTS-3 download...", end=" ")
    try:
        from geometry import download_nuts3_shapes
        
        # Download NUTS-3 for Luxembourg (small country, fast download)
        nuts3 = download_nuts3_shapes(['LU'])
        
        # Luxembourg should have at least 1 NUTS-3 region
        assert len(nuts3) >= 1, "Should have at least 1 NUTS-3 region"
        assert nuts3.geometry.iloc[0] is not None, "Should have geometry"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mask_shape():
    """Test intersection/masking."""
    print("Testing mask/intersection...", end=" ")
    try:
        from geometry import download_country_shapes, mask_shape
        
        # Download two countries
        germany = download_country_shapes(['DE'])
        france = download_country_shapes(['FR'])
        
        # Intersection should be empty (countries don't overlap)
        intersection = mask_shape(germany, france)
        
        # Should be empty or very small
        assert intersection.is_empty or intersection.area < 0.001, "Countries shouldn't overlap"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GEOMETRY MODULE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_download_countries,
        test_point_in_shape,
        test_join_shapes,
        test_download_nuts3,
        test_mask_shape,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"ALL TESTS PASSED ({passed}/{total})")
        print("=" * 60)
        return 0
    else:
        print(f"SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
