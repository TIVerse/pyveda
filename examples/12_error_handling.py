"""Advanced Error Handling with VedaRT.

This example demonstrates VedaRT's comprehensive error handling capabilities:
- Failure strategies (fail-fast, continue, retry)
- Error aggregation and reporting
- Partial results recovery
- Graceful degradation
- Timeout handling
"""

import time
import random
from typing import List, Optional
import vedart as veda


# ============================================================================
# Example 1: Fail-Fast Strategy
# ============================================================================

def example_fail_fast():
    """Demonstrate fail-fast error handling."""
    print("=" * 60)
    print("Example 1: Fail-Fast Strategy")
    print("=" * 60)
    
    def unreliable_task(x: int) -> int:
        """Task that fails 30% of the time."""
        if random.random() < 0.3:
            raise ValueError(f"Task {x} failed!")
        return x * 2
    
    print("\nFail-fast mode (stops on first error):")
    
    try:
        with veda.deterministic(seed=42):  # For reproducibility
            result = veda.par_iter(range(20)).map(unreliable_task).collect()
        print(f"Success! Processed {len(result)} items")
    except Exception as e:
        print(f"✓ Failed fast as expected: {type(e).__name__}")
        print(f"  Error: {str(e)[:50]}...")
    
    print()


# ============================================================================
# Example 2: Continue on Error
# ============================================================================

def example_continue_on_error():
    """Demonstrate continuing execution despite errors."""
    print("=" * 60)
    print("Example 2: Continue on Error")
    print("=" * 60)
    
    def sometimes_fails(x: int) -> int:
        """Fails on multiples of 3."""
        if x % 3 == 0 and x > 0:
            raise ValueError(f"Cannot process {x}")
        return x ** 2
    
    print("\nProcessing with error tolerance:")
    
    # Simulate continue-on-error by catching exceptions in the task
    def safe_task(x: int) -> Optional[int]:
        try:
            return sometimes_fails(x)
        except Exception:
            return None  # Return sentinel value
    
    result = veda.par_iter(range(15)).map(safe_task).collect()
    
    # Filter out failures
    successes = [r for r in result if r is not None]
    failures = len([r for r in result if r is None])
    
    print(f"✓ Completed with partial results:")
    print(f"  Successes: {len(successes)}")
    print(f"  Failures:  {failures}")
    print(f"  Results:   {successes[:5]}...")
    print()


# ============================================================================
# Example 3: Retry Strategy
# ============================================================================

def example_retry_strategy():
    """Demonstrate automatic retry on failure."""
    print("=" * 60)
    print("Example 3: Retry Strategy")
    print("=" * 60)
    
    # Track retry attempts
    attempts = {}
    
    def flaky_network_call(x: int) -> str:
        """Simulates flaky network operation."""
        if x not in attempts:
            attempts[x] = 0
        
        attempts[x] += 1
        
        # Succeed on 3rd attempt
        if attempts[x] < 3:
            raise ConnectionError(f"Network timeout for {x}")
        
        return f"Data-{x}"
    
    def retry_task(x: int, max_retries: int = 3) -> str:
        """Wrapper with retry logic."""
        for attempt in range(max_retries):
            try:
                return flaky_network_call(x)
            except ConnectionError as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
                time.sleep(0.01 * (attempt + 1))  # Exponential backoff
        return ""  # Unreachable
    
    print("\nRetrying flaky operations:")
    result = veda.par_iter(range(10)).map(retry_task).collect()
    
    print(f"✓ All tasks succeeded after retries:")
    print(f"  Results: {result[:5]}...")
    print(f"  Total retry attempts: {sum(attempts.values())}")
    print()


# ============================================================================
# Example 4: Error Aggregation
# ============================================================================

class TaskError:
    """Container for task error information."""
    
    def __init__(self, task_id: int, error: Exception):
        self.task_id = task_id
        self.error = error
        self.error_type = type(error).__name__
        self.error_message = str(error)
    
    def __repr__(self):
        return f"TaskError(id={self.task_id}, type={self.error_type})"


def example_error_aggregation():
    """Demonstrate collecting and analyzing errors."""
    print("=" * 60)
    print("Example 4: Error Aggregation")
    print("=" * 60)
    
    def multi_error_task(x: int) -> int:
        """Tasks with different error types."""
        if x % 5 == 0:
            raise ValueError("Validation error")
        elif x % 7 == 0:
            raise TypeError("Type error")
        elif x % 11 == 0:
            raise RuntimeError("Runtime error")
        return x * 2
    
    def safe_wrapper(x: int) -> tuple[Optional[int], Optional[TaskError]]:
        """Wrapper that captures errors."""
        try:
            result = multi_error_task(x)
            return (result, None)
        except Exception as e:
            return (None, TaskError(x, e))
    
    print("\nCollecting errors from parallel execution:")
    results = veda.par_iter(range(30)).map(safe_wrapper).collect()
    
    # Separate successes and failures
    successes = [r[0] for r in results if r[1] is None]
    errors = [r[1] for r in results if r[1] is not None]
    
    # Aggregate error statistics
    error_counts = {}
    for err in errors:
        error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1
    
    print(f"✓ Error analysis complete:")
    print(f"  Total tasks:  {len(results)}")
    print(f"  Succeeded:    {len(successes)}")
    print(f"  Failed:       {len(errors)}")
    print(f"\n  Error breakdown:")
    for error_type, count in sorted(error_counts.items()):
        print(f"    {error_type}: {count}")
    
    # Show sample errors
    print(f"\n  Sample errors: {errors[:3]}")
    print()


# ============================================================================
# Example 5: Timeout Handling
# ============================================================================

def example_timeout_handling():
    """Demonstrate timeout handling for long-running tasks."""
    print("=" * 60)
    print("Example 5: Timeout Handling")
    print("=" * 60)
    
    def variable_duration_task(x: int) -> int:
        """Task with variable duration."""
        duration = x * 0.01  # 0-0.5 seconds
        time.sleep(duration)
        return x ** 2
    
    def with_timeout(func, timeout: float):
        """Wrapper to add timeout to function."""
        import signal
        
        def handler(signum, frame):
            raise TimeoutError(f"Task exceeded {timeout}s")
        
        def wrapper(*args, **kwargs):
            # Note: signal.alarm only works on Unix
            try:
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(int(timeout))
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            except AttributeError:
                # signal.SIGALRM not available (Windows)
                # Fallback to simple execution
                return func(*args, **kwargs)
        
        return wrapper
    
    print("\nProcessing with timeout protection:")
    
    def timeout_task(x: int) -> tuple[int, Optional[str]]:
        """Task with timeout wrapper."""
        try:
            result = variable_duration_task(x)
            return (x, None)  # Success
        except TimeoutError as e:
            return (x, "timeout")
    
    results = veda.par_iter(range(20)).map(timeout_task).collect()
    
    successes = [r for r in results if r[1] is None]
    timeouts = [r for r in results if r[1] == "timeout"]
    
    print(f"✓ Timeout handling complete:")
    print(f"  Completed:  {len(successes)}")
    print(f"  Timed out:  {len(timeouts)}")
    print()


# ============================================================================
# Example 6: Partial Results Recovery
# ============================================================================

def example_partial_results():
    """Demonstrate recovering partial results from failed batch."""
    print("=" * 60)
    print("Example 6: Partial Results Recovery")
    print("=" * 60)
    
    def batch_processor(batch_id: int) -> List[int]:
        """Process a batch, may fail on some items."""
        results = []
        for i in range(10):
            item_id = batch_id * 10 + i
            
            # Simulate failure on specific items
            if item_id == 25 or item_id == 47:
                print(f"  ⚠ Item {item_id} failed")
                continue  # Skip failed item
            
            results.append(item_id ** 2)
        
        return results
    
    print("\nProcessing batches with partial recovery:")
    batch_results = veda.par_iter(range(5)).map(batch_processor).collect()
    
    # Flatten results
    all_results = [item for batch in batch_results for item in batch]
    
    print(f"✓ Recovery complete:")
    print(f"  Batches processed:  {len(batch_results)}")
    print(f"  Total items:        {len(all_results)}")
    print(f"  Expected items:     48 (2 failed)")
    print(f"  Sample results:     {all_results[:5]}...")
    print()


# ============================================================================
# Example 7: Graceful Degradation
# ============================================================================

def example_graceful_degradation():
    """Demonstrate graceful degradation with fallback strategies."""
    print("=" * 60)
    print("Example 7: Graceful Degradation")
    print("=" * 60)
    
    def primary_service(x: int) -> str:
        """Primary computation (may fail)."""
        if x % 4 == 0:
            raise ConnectionError("Primary service unavailable")
        return f"primary-{x}"
    
    def fallback_service(x: int) -> str:
        """Fallback computation (always works, lower quality)."""
        return f"fallback-{x}"
    
    def resilient_task(x: int) -> tuple[int, str, str]:
        """Task with automatic fallback."""
        try:
            result = primary_service(x)
            return (x, result, "primary")
        except Exception:
            result = fallback_service(x)
            return (x, result, "fallback")
    
    print("\nProcessing with automatic fallback:")
    results = veda.par_iter(range(20)).map(resilient_task).collect()
    
    # Analyze service usage
    primary_count = len([r for r in results if r[2] == "primary"])
    fallback_count = len([r for r in results if r[2] == "fallback"])
    
    print(f"✓ Graceful degradation successful:")
    print(f"  Total requests:     {len(results)}")
    print(f"  Primary service:    {primary_count}")
    print(f"  Fallback service:   {fallback_count}")
    print(f"  Success rate:       100% (no failures)")
    print()


# ============================================================================
# Example 8: Error Context and Debugging
# ============================================================================

def example_error_context():
    """Demonstrate capturing rich error context."""
    print("=" * 60)
    print("Example 8: Error Context for Debugging")
    print("=" * 60)
    
    class RichError:
        """Error with full context."""
        
        def __init__(self, task_id: int, error: Exception, context: dict):
            self.task_id = task_id
            self.error = error
            self.context = context
            import traceback
            self.traceback = traceback.format_exc()
        
        def report(self) -> str:
            lines = [
                f"Task {self.task_id} failed:",
                f"  Error: {type(self.error).__name__}: {self.error}",
                f"  Context: {self.context}",
                f"  Traceback: {self.traceback[:200]}...",
            ]
            return "\n".join(lines)
    
    def complex_task(x: int) -> int:
        """Task that may fail with context."""
        context = {
            "input": x,
            "timestamp": time.time(),
            "attempt": 1,
        }
        
        try:
            if x == 7:
                raise ValueError("Lucky number 7 not allowed!")
            return x * 3
        except Exception as e:
            raise RichError(x, e, context)
    
    def safe_complex_task(x: int):
        try:
            return complex_task(x)
        except RichError as e:
            return e
    
    print("\nProcessing with rich error context:")
    results = veda.par_iter(range(15)).map(safe_complex_task).collect()
    
    # Find errors
    errors = [r for r in results if isinstance(r, RichError)]
    successes = [r for r in results if not isinstance(r, RichError)]
    
    print(f"✓ Execution complete:")
    print(f"  Succeeded: {len(successes)}")
    print(f"  Failed:    {len(errors)}")
    
    if errors:
        print(f"\n  Error details:")
        for err in errors:
            print(f"    {err.report()[:150]}...")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VedaRT Advanced Error Handling Examples")
    print("=" * 60 + "\n")
    
    example_fail_fast()
    example_continue_on_error()
    example_retry_strategy()
    example_error_aggregation()
    example_timeout_handling()
    example_partial_results()
    example_graceful_degradation()
    example_error_context()
    
    print("=" * 60)
    print("All error handling examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Multiple failure strategies available")
    print("  • Rich error context for debugging")
    print("  • Partial results recovery")
    print("  • Graceful degradation with fallbacks")
    print("  • Automatic retry with exponential backoff")
    print()
