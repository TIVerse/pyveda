"""GPU acceleration example."""

import pyveda as veda


@veda.gpu
def matrix_multiply(A, B):
    """Matrix multiplication with automatic GPU offload."""
    return A @ B


@veda.gpu
def vector_add(a, b):
    """Vector addition with GPU offload."""
    return a + b


def main():
    """Demonstrate GPU acceleration."""
    print("PyVeda - GPU Acceleration\n")
    
    # Initialize runtime with GPU support
    config = veda.Config.builder().gpu(True).build()
    veda.init(config)
    
    runtime = veda.get_runtime()
    
    if runtime.gpu and runtime.gpu.is_available():
        print(f"GPU backend: {runtime.gpu.backend}")
        print(f"GPU devices: {runtime.gpu.device_count}\n")
        
        try:
            import numpy as np
            
            # Small matrices (CPU execution)
            print("1. Small matrices (CPU):")
            A = np.ones((10, 10))
            B = np.ones((10, 10))
            C = matrix_multiply(A, B)
            print(f"   Result shape: {C.shape}\n")
            
            # Large matrices (GPU execution)
            print("2. Large matrices (GPU if beneficial):")
            A = np.ones((1000, 1000))
            B = np.ones((1000, 1000))
            C = matrix_multiply(A, B)
            print(f"   Result shape: {C.shape}")
            print(f"   Sum: {C.sum():.0f}\n")
            
        except ImportError:
            print("NumPy not installed, skipping matrix operations")
    else:
        print("GPU not available (install cupy or numba)")


if __name__ == "__main__":
    main()
    veda.shutdown()
