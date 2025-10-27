"""Example: Image Processing Pipeline with CPU+GPU Mix

This example demonstrates a real-world image processing pipeline that
intelligently mixes CPU and GPU operations for optimal performance.

Features demonstrated:
- CPU-bound preprocessing (file I/O, format conversion)
- GPU-accelerated transformations (filters, resizing)
- Automatic fallback when GPU unavailable
- Efficient batch processing
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

import vedart as veda

# Try to import PIL for image processing
try:
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  PIL not installed. Install with: pip install Pillow")


class ImageProcessor:
    """Image processing pipeline with CPU+GPU support."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and veda.get_runtime().gpu is not None

    def load_image(self, path: str) -> np.ndarray:
        """Load image from disk (CPU I/O-bound)."""
        if not HAS_PIL:
            # Simulate with random data
            return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def preprocess_cpu(self, img: np.ndarray) -> np.ndarray:
        """CPU-bound preprocessing operations."""
        # Simulate expensive CPU operations
        # In practice: format conversion, color space transforms, etc.
        img = img.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        return img

    @veda.gpu
    def apply_filter_gpu(self, img: np.ndarray) -> np.ndarray:
        """GPU-accelerated filter application.
        
        This decorator automatically:
        1. Detects if GPU is available
        2. Transfers data to GPU memory
        3. Executes on GPU if beneficial
        4. Falls back to CPU if needed
        """
        # Gaussian blur simulation
        # On GPU: uses CuPy's optimized kernels
        # On CPU: uses NumPy (automatic fallback)
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable filter (optimized on GPU)
        from scipy.ndimage import convolve1d
        img_blurred = convolve1d(img, kernel_1d, axis=0)
        img_blurred = convolve1d(img_blurred, kernel_1d, axis=1)
        
        return img_blurred

    @veda.gpu
    def resize_gpu(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """GPU-accelerated image resizing."""
        # In production, this would use GPU-optimized resize
        # For demo, we'll use simple nearest-neighbor
        h, w = target_size
        curr_h, curr_w = img.shape[:2]
        
        # Scale factors
        scale_h = curr_h / h
        scale_w = curr_w / w
        
        # Create coordinate grids
        coords_h = (np.arange(h) * scale_h).astype(np.int32)
        coords_w = (np.arange(w) * scale_w).astype(np.int32)
        
        # Index into original image
        resized = img[coords_h[:, None], coords_w[None, :]]
        
        return resized

    def postprocess_cpu(self, img: np.ndarray) -> np.ndarray:
        """CPU-bound postprocessing operations."""
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        
        # Clip and convert back to uint8
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        return img

    def save_image(self, img: np.ndarray, path: str) -> None:
        """Save image to disk (CPU I/O-bound)."""
        if HAS_PIL:
            Image.fromarray(img).save(path)


def process_single_image(processor: ImageProcessor, img_path: str, output_path: str) -> dict[str, Any]:
    """Process a single image through the pipeline."""
    start_time = time.perf_counter()
    
    # Step 1: Load (CPU I/O)
    img = processor.load_image(img_path)
    load_time = time.perf_counter() - start_time
    
    # Step 2: Preprocess (CPU)
    img = processor.preprocess_cpu(img)
    preprocess_time = time.perf_counter() - start_time - load_time
    
    # Step 3: Apply filter (GPU if available)
    img = processor.apply_filter_gpu(img)
    filter_time = time.perf_counter() - start_time - load_time - preprocess_time
    
    # Step 4: Resize (GPU if available)
    img = processor.resize_gpu(img, (256, 256))
    resize_time = time.perf_counter() - start_time - load_time - preprocess_time - filter_time
    
    # Step 5: Postprocess (CPU)
    img = processor.postprocess_cpu(img)
    postprocess_time = time.perf_counter() - start_time - load_time - preprocess_time - filter_time - resize_time
    
    # Step 6: Save (CPU I/O)
    processor.save_image(img, output_path)
    save_time = time.perf_counter() - start_time - load_time - preprocess_time - filter_time - resize_time - postprocess_time
    
    total_time = time.perf_counter() - start_time
    
    return {
        "path": img_path,
        "total_ms": total_time * 1000,
        "breakdown": {
            "load": load_time * 1000,
            "preprocess": preprocess_time * 1000,
            "filter_gpu": filter_time * 1000,
            "resize_gpu": resize_time * 1000,
            "postprocess": postprocess_time * 1000,
            "save": save_time * 1000,
        }
    }


def demo_parallel_batch_processing():
    """Demonstrate parallel batch image processing."""
    print("🖼️  Image Processing Pipeline Demo")
    print("=" * 60)
    
    # Create synthetic image dataset
    print("\n📂 Generating synthetic images...")
    image_paths = []
    output_dir = Path("/tmp/vedart_images")
    output_dir.mkdir(exist_ok=True)
    
    for i in range(20):
        # Create random images
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        path = output_dir / f"input_{i:03d}.png"
        
        if HAS_PIL:
            Image.fromarray(img).save(path)
        
        image_paths.append(str(path))
    
    print(f"✓ Created {len(image_paths)} images")
    
    # Initialize processor
    processor = ImageProcessor(use_gpu=True)
    gpu_status = "✓ GPU" if processor.use_gpu else "⚠️  CPU only"
    print(f"\n⚙️  Processor initialized ({gpu_status})")
    
    # Process images in parallel
    print(f"\n🚀 Processing {len(image_paths)} images in parallel...")
    start = time.perf_counter()
    
    results = (
        veda.par_iter(image_paths)
        .enumerate()
        .map(lambda idx_path: process_single_image(
            processor,
            idx_path[1],
            str(output_dir / f"output_{idx_path[0]:03d}.png")
        ))
        .collect()
    )
    
    elapsed = time.perf_counter() - start
    
    # Statistics
    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"   Throughput: {len(image_paths) / elapsed:.1f} images/sec")
    
    avg_times = {
        "load": np.mean([r["breakdown"]["load"] for r in results]),
        "preprocess": np.mean([r["breakdown"]["preprocess"] for r in results]),
        "filter_gpu": np.mean([r["breakdown"]["filter_gpu"] for r in results]),
        "resize_gpu": np.mean([r["breakdown"]["resize_gpu"] for r in results]),
        "postprocess": np.mean([r["breakdown"]["postprocess"] for r in results]),
        "save": np.mean([r["breakdown"]["save"] for r in results]),
    }
    
    print("\n📊 Average time per stage (ms):")
    for stage, time_ms in avg_times.items():
        bar = "█" * int(time_ms / 5)
        print(f"   {stage:12s}: {time_ms:6.2f}ms {bar}")
    
    # Telemetry
    if hasattr(veda, 'telemetry'):
        print("\n📈 Telemetry:")
        metrics = veda.telemetry.snapshot()
        print(f"   Tasks executed: {metrics.tasks_executed}")
        print(f"   Avg latency: {metrics.avg_latency_ms:.2f}ms")


def demo_cpu_vs_gpu_comparison():
    """Compare CPU vs GPU performance."""
    print("\n" + "=" * 60)
    print("⚡ CPU vs GPU Performance Comparison")
    print("=" * 60)
    
    # Create test image
    test_img = np.random.rand(1024, 1024, 3).astype(np.float32)
    
    # CPU-only processing
    print("\n🔹 CPU-only mode:")
    processor_cpu = ImageProcessor(use_gpu=False)
    
    start = time.perf_counter()
    for _ in range(10):
        result = processor_cpu.apply_filter_gpu(test_img)
    cpu_time = (time.perf_counter() - start) / 10
    
    print(f"   Avg filter time: {cpu_time * 1000:.2f}ms")
    
    # GPU mode (with fallback)
    print("\n🔹 GPU mode (auto-fallback):")
    processor_gpu = ImageProcessor(use_gpu=True)
    
    start = time.perf_counter()
    for _ in range(10):
        result = processor_gpu.apply_filter_gpu(test_img)
    gpu_time = (time.perf_counter() - start) / 10
    
    print(f"   Avg filter time: {gpu_time * 1000:.2f}ms")
    
    if processor_gpu.use_gpu and gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n🎉 GPU speedup: {speedup:.2f}x faster")
    elif not processor_gpu.use_gpu:
        print(f"\n⚠️  GPU not available - using CPU fallback")
    else:
        print(f"\n💡 For this workload, CPU is sufficient")


def main():
    """Run all demos."""
    try:
        demo_parallel_batch_processing()
        demo_cpu_vs_gpu_comparison()
        
        print("\n" + "=" * 60)
        print("✨ Demo complete!")
        print("\nKey takeaways:")
        print("  • CPU operations (I/O, preprocessing) run on thread pool")
        print("  • GPU operations (@veda.gpu) automatically offloaded")
        print("  • Seamless fallback when GPU unavailable")
        print("  • VedaRT handles all executor management")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
