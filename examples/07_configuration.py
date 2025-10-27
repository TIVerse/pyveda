"""Configuration and tuning example."""

import vedart as veda


def main():
    """Demonstrate runtime configuration."""
    print("VedaRT - Configuration Examples\n")
    
    # Default configuration
    print("1. Default configuration:")
    config1 = veda.Config.default()
    print(f"   Policy: {config1.scheduling_policy.value}")
    print(f"   Telemetry: {config1.enable_telemetry}")
    print(f"   GPU: {config1.enable_gpu}\n")
    
    # Builder pattern
    print("2. Custom configuration with builder:")
    config2 = (
        veda.Config.builder()
        .threads(4)
        .processes(2)
        .policy(veda.SchedulingPolicy.ADAPTIVE)
        .telemetry(True)
        .worker_limits(min_workers=1, max_workers=8)
        .build()
    )
    print(f"   Threads: {config2.num_threads}")
    print(f"   Processes: {config2.num_processes}")
    print(f"   Min workers: {config2.min_workers}")
    print(f"   Max workers: {config2.max_workers}\n")
    
    # Thread-only configuration
    print("3. Thread-only mode:")
    config3 = (
        veda.Config.builder()
        .threads(8)
        .policy(veda.SchedulingPolicy.THREAD_ONLY)
        .build()
    )
    print(f"   Policy: {config3.scheduling_policy.value}\n")
    
    # Initialize with custom config
    print("4. Initialize runtime with custom config:")
    veda.init(config2)
    runtime = veda.get_runtime()
    print(f"   Runtime initialized")
    print(f"   Scheduler type: {type(runtime.scheduler).__name__}")
    
    # Run some work
    result = veda.par_iter(range(100)).sum()
    print(f"   Test computation result: {result}")


if __name__ == "__main__":
    main()
    veda.shutdown()
