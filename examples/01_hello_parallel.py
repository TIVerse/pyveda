"""Basic parallel iteration example."""

import vedart as veda


def main():
    """Demonstrate basic parallel operations."""
    print("VedaRT - Hello Parallel World\n")
    
    # Simple map operation
    print("1. Parallel map:")
    result = veda.par_iter(range(10)).map(lambda x: x * 2).collect()
    print(f"   Input: range(10)")
    print(f"   Output: {result}\n")
    
    # Map and filter chain
    print("2. Map + Filter chain:")
    result = (
        veda.par_iter(range(20))
        .map(lambda x: x * 2)
        .filter(lambda x: x > 20)
        .collect()
    )
    print(f"   Result: {result}\n")
    
    # Sum operation
    print("3. Parallel sum:")
    total = veda.par_iter(range(1000)).sum()
    print(f"   Sum of range(1000): {total}\n")
    
    # Fold operation
    print("4. Parallel fold:")
    product = veda.par_iter([1, 2, 3, 4, 5]).fold(1, lambda acc, x: acc * x)
    print(f"   Product of [1,2,3,4,5]: {product}\n")


if __name__ == "__main__":
    main()
    veda.shutdown()
