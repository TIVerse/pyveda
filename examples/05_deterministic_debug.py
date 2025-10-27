"""Deterministic debugging example."""

import random

import vedart as veda


def flaky_computation(seed_offset):
    """Simulates a computation that might produce different results."""
    # Without determinism, this could vary due to scheduling
    data = veda.par_iter(range(100)).map(lambda x: x * seed_offset).collect()
    return sum(data)


def main():
    """Demonstrate deterministic debugging."""
    print("VedaRT - Deterministic Debugging\n")
    
    # Run without determinism
    print("1. Without deterministic mode:")
    results_normal = []
    for i in range(3):
        result = flaky_computation(i + 1)
        results_normal.append(result)
        print(f"   Run {i+1}: {result}")
    
    print()
    
    # Run with deterministic mode
    print("2. With deterministic mode (seed=42):")
    results_det1 = []
    for i in range(3):
        with veda.deterministic(seed=42):
            result = flaky_computation(i + 1)
            results_det1.append(result)
            print(f"   Run {i+1}: {result}")
    
    print()
    
    # Run again with same seed
    print("3. Repeat with same seed (seed=42):")
    results_det2 = []
    for i in range(3):
        with veda.deterministic(seed=42):
            result = flaky_computation(i + 1)
            results_det2.append(result)
            print(f"   Run {i+1}: {result}")
    
    print()
    
    # Verify determinism
    print("4. Verification:")
    print(f"   Results match: {results_det1 == results_det2}")
    print("   ✓ Deterministic mode ensures reproducibility!")


if __name__ == "__main__":
    main()
    veda.shutdown()
