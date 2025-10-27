"""Machine learning pipeline example."""

import time

import pyveda as veda


def load_data_batch(batch_id):
    """Simulate loading a data batch."""
    time.sleep(0.01)  # Simulate I/O
    return [{"id": i, "features": [i, i * 2]} for i in range(batch_id * 10, (batch_id + 1) * 10)]


def preprocess_sample(sample):
    """Preprocess a single sample."""
    # Normalize features
    features = sample["features"]
    normalized = [f / 100.0 for f in features]
    return {**sample, "features": normalized}


def compute_features(sample):
    """Compute additional features."""
    features = sample["features"]
    enhanced = features + [sum(features), max(features)]
    return {**sample, "features": enhanced}


def train_model(samples):
    """Simulate model training."""
    time.sleep(0.1)
    return {"weights": len(samples), "accuracy": 0.95}


def main():
    """Demonstrate ML training pipeline."""
    print("PyVeda - ML Pipeline Example\n")
    
    # Step 1: Parallel data loading
    print("Step 1: Load data batches (parallel I/O)")
    start = time.time()
    
    with veda.scope() as s:
        futures = [s.spawn(load_data_batch, i) for i in range(5)]
        batches = s.wait_all()
    
    all_samples = [sample for batch in batches for sample in batch]
    print(f"  Loaded {len(all_samples)} samples in {time.time() - start:.2f}s\n")
    
    # Step 2: Parallel preprocessing
    print("Step 2: Preprocess samples (parallel CPU)")
    start = time.time()
    
    preprocessed = (
        veda.par_iter(all_samples)
        .map(preprocess_sample)
        .map(compute_features)
        .collect()
    )
    
    print(f"  Preprocessed {len(preprocessed)} samples in {time.time() - start:.2f}s\n")
    
    # Step 3: Train model
    print("Step 3: Train model")
    start = time.time()
    
    model = train_model(preprocessed)
    
    print(f"  Training complete in {time.time() - start:.2f}s")
    print(f"  Model accuracy: {model['accuracy']:.2%}")
    print(f"  Sample features: {preprocessed[0]['features']}")


if __name__ == "__main__":
    main()
    veda.shutdown()
