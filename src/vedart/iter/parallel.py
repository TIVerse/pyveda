"""Rayon-inspired parallel iterator for Python."""

import logging
import os
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

from vedart.core.task import Task

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def _process_chunk_with_ops(chunk: list, operations: list, chunk_idx: int) -> tuple:
    """Module-level helper for processing chunks (picklable for multiprocessing).

    Args:
        chunk: Data chunk to process
        operations: List of operations to apply
        chunk_idx: Index of this chunk

    Returns:
        Tuple of (chunk_idx, processed_result)
    """
    result = chunk
    for op in operations:
        result = op.apply(result)
    return (chunk_idx, result)


class Operation:
    """Base class for iterator operations."""

    def apply(self, items: list[Any]) -> list[Any]:
        """Apply operation to a list of items.

        Args:
            items: Input items

        Returns:
            Transformed items
        """
        raise NotImplementedError


class MapOp(Operation):
    """Map operation."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def apply(self, items: list[Any]) -> list[Any]:
        return [self.func(item) for item in items]


class FilterOp(Operation):
    """Filter operation."""

    def __init__(self, predicate: Callable[[Any], bool]) -> None:
        self.predicate = predicate

    def apply(self, items: list[Any]) -> list[Any]:
        return [item for item in items if self.predicate(item)]


class FlatMapOp(Operation):
    """Flat map operation."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def apply(self, items: list[Any]) -> list[Any]:
        result = []
        for item in items:
            mapped = self.func(item)
            if hasattr(mapped, "__iter__") and not isinstance(mapped, (str, bytes)):
                result.extend(mapped)
            else:
                result.append(mapped)
        return result


class AsyncMapOp(Operation):
    """Async map operation that runs async functions in sync context."""

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def apply(self, items: list[Any]) -> list[Any]:
        """Apply async function to items by running each in a new event loop."""
        import asyncio
        import inspect
        
        results = []
        for item in items:
            if inspect.iscoroutinefunction(self.func):
                # Run async function
                try:
                    result = asyncio.run(self.func(item))
                except RuntimeError:
                    # If there's already an event loop, this shouldn't happen in worker processes
                    # But as a fallback, just call the function and it will return a coroutine
                    coro = self.func(item)
                    result = asyncio.get_event_loop().run_until_complete(coro)
                results.append(result)
            else:
                # Regular function
                results.append(self.func(item))
        return results


class ParallelIterator(Generic[T]):
    """Rayon-inspired parallel iterator.

    Provides composable parallel operations on iterables with
    automatic chunking and distribution to workers.

    Example:
        result = (
            par_iter(range(1000))
            .map(lambda x: x * 2)
            .filter(lambda x: x > 100)
            .collect()
        )
    """

    def __init__(
        self,
        iterable: Iterable[T],
        chunk_size: int | None = None,
        ordered: bool = True,
    ) -> None:
        """Initialize parallel iterator.

        Args:
            iterable: Source iterable
            chunk_size: Chunk size for parallelism (None = auto)
            ordered: Preserve order of results
        """
        self._iterable = iterable
        self._chunk_size = chunk_size
        self._ordered = ordered
        self._operations: list[Operation] = []
        self._preserve_chunks = False  # Flag to preserve chunk boundaries in collect()

    def map(self, func: Callable[[T], U]) -> "ParallelIterator[U]":
        """Apply function to each element in parallel.

        Args:
            func: Transformation function

        Returns:
            New iterator with map operation
        """
        new_iter = ParallelIterator(
            self._iterable,
            self._chunk_size,
            self._ordered,
        )
        new_iter._operations = self._operations + [MapOp(func)]
        new_iter._preserve_chunks = self._preserve_chunks
        return new_iter

    def filter(self, predicate: Callable[[T], bool]) -> "ParallelIterator[T]":
        """Filter elements in parallel.

        Args:
            predicate: Filter predicate

        Returns:
            New iterator with filter operation
        """
        new_iter = ParallelIterator(
            self._iterable,
            self._chunk_size,
            self._ordered,
        )
        new_iter._operations = self._operations + [FilterOp(predicate)]
        new_iter._preserve_chunks = self._preserve_chunks
        return new_iter

    def flat_map(self, func: Callable[[T], Iterable[U]]) -> "ParallelIterator[U]":
        """Map and flatten results in parallel.

        Args:
            func: Function that returns an iterable

        Returns:
            New iterator with flat_map operation
        """
        new_iter = ParallelIterator(
            self._iterable,
            self._chunk_size,
            self._ordered,
        )
        new_iter._operations = self._operations + [FlatMapOp(func)]
        new_iter._preserve_chunks = self._preserve_chunks
        return new_iter

    def enumerate(self) -> "ParallelIterator[tuple[int, T]]":
        """Add indices to elements.

        Returns:
            New iterator with (index, element) tuples
        """
        # Materialize with sequential indices before applying operations
        items = list(self._iterable)
        enumerated = [(idx, item) for idx, item in enumerate(items)]
        new_iter = ParallelIterator(enumerated, self._chunk_size, self._ordered)
        new_iter._operations = self._operations.copy()
        return new_iter

    def zip(self, other: Iterable[U]) -> "ParallelIterator[tuple[T, U]]":
        """Zip with another iterable.

        Args:
            other: Iterable to zip with

        Returns:
            New iterator with tuples
        """
        items = list(self._iterable)
        other_items = list(other)
        zipped = list(zip(items, other_items, strict=False))
        new_iter = ParallelIterator(zipped, self._chunk_size, self._ordered)
        new_iter._operations = self._operations.copy()
        return new_iter

    def take(self, n: int) -> "ParallelIterator[T]":
        """Take first n elements.

        Args:
            n: Number of elements to take

        Returns:
            New iterator limited to n elements
        """
        # Convert iterable to list slice
        items = list(self._iterable)[:n]
        new_iter = ParallelIterator(items, self._chunk_size, self._ordered)
        new_iter._operations = self._operations.copy()
        return new_iter

    def skip(self, n: int) -> "ParallelIterator[T]":
        """Skip first n elements.

        Args:
            n: Number of elements to skip

        Returns:
            New iterator without first n elements
        """
        items = list(self._iterable)[n:]
        new_iter = ParallelIterator(items, self._chunk_size, self._ordered)
        new_iter._operations = self._operations.copy()
        return new_iter

    def chunk(self, size: int) -> "ParallelIterator[list[T]]":
        """Group elements into fixed-size chunks.

        Args:
            size: Chunk size

        Returns:
            New iterator yielding chunks
        """
        items = list(self._iterable)
        chunks = [items[i : i + size] for i in range(0, len(items), size)]
        new_iter = ParallelIterator(chunks, self._chunk_size, self._ordered)
        new_iter._operations = self._operations.copy()
        new_iter._preserve_chunks = True  # Set flag to preserve chunks in collect()
        return new_iter

    def gpu_map(self, func: Callable[[T], U]) -> "ParallelIterator[U]":
        """Apply function using GPU acceleration.

        Args:
            func: Function to apply (should be GPU-compatible)

        Returns:
            New iterator with GPU-accelerated map
        """
        # Mark items for GPU processing by wrapping the function
        from vedart.gpu.decorators import gpu

        gpu_func = gpu(func)
        return self.map(gpu_func)

    def async_map(self, func: Callable[[T], U]) -> "ParallelIterator[U]":
        """Apply async function to elements.

        Args:
            func: Async function to apply

        Returns:
            New iterator with async map
        """
        # Use AsyncMapOp which handles async execution in worker processes
        new_iter = ParallelIterator(
            self._iterable,
            self._chunk_size,
            self._ordered,
        )
        new_iter._operations = self._operations + [AsyncMapOp(func)]
        new_iter._preserve_chunks = self._preserve_chunks
        return new_iter

    def collect(self) -> list[T]:
        """Execute pipeline and collect results.

        Returns:
            List of results
        """
        from vedart.core.runtime import get_runtime

        runtime = get_runtime()

        # Convert to list and chunk
        items = list(self._iterable)
        if not items:
            return []

        # If preserve_chunks is set, items ARE the chunks - process each as a unit
        if self._preserve_chunks:
            # Each item is already a chunk, process them directly without re-chunking
            futures = []
            for idx, chunk in enumerate(items):
                task = Task(
                    func=_process_chunk_with_ops,
                    args=(chunk, self._operations, idx),
                    kwargs={},
                )
                future = runtime.scheduler.submit(task)
                futures.append(future)
            
            # Collect results
            results = [f.result() for f in futures]
            
            # Preserve order
            if self._ordered:
                results.sort(key=lambda x: x[0])
            
            # Return the processed chunks directly
            return [chunk_result for _, chunk_result in results]

        # Normal path: chunk for parallel processing, then flatten
        chunk_size = self._chunk_size or self._auto_chunk_size(len(items))
        chunks = self._chunk_iterable(items, chunk_size)

        # Submit tasks (using module-level function for picklability)
        futures = []
        for idx, chunk in enumerate(chunks):
            task = Task(
                func=_process_chunk_with_ops,
                args=(chunk, self._operations, idx),
                kwargs={},
            )
            future = runtime.scheduler.submit(task)
            futures.append(future)

        # Collect results
        results = [f.result() for f in futures]

        # Optionally preserve order
        if self._ordered:
            results.sort(key=lambda x: x[0])  # Sort by chunk index
        
        # Flatten results
        flattened = []
        for _, chunk_result in results:
            flattened.extend(chunk_result)

        return flattened

    def fold(self, identity: U, op: Callable[[U, T], U]) -> U:
        """Parallel fold/reduce with identity element.

        Args:
            identity: Identity element for the operation
            op: Binary associative operation

        Returns:
            Reduced value
        """
        chunks = self.collect()
        if not chunks:
            return identity

        # Tree reduction for associativity
        return self._tree_reduce(chunks, identity, op)

    def reduce(self, op: Callable[[T, T], T]) -> T | None:
        """Parallel reduce without identity.

        Args:
            op: Binary associative operation

        Returns:
            Reduced value or None if empty
        """
        chunks = self.collect()
        if not chunks:
            return None

        if len(chunks) == 1:
            return chunks[0]

        # Tree reduction
        while len(chunks) > 1:
            new_chunks = []
            for i in range(0, len(chunks), 2):
                if i + 1 < len(chunks):
                    new_chunks.append(op(chunks[i], chunks[i + 1]))
                else:
                    new_chunks.append(chunks[i])
            chunks = new_chunks

        return chunks[0]

    def sum(self) -> T:
        """Sum all elements (requires numeric type).

        Returns:
            Sum of all elements
        """
        return self.fold(0, lambda a, b: a + b)  # type: ignore

    def count(self) -> int:
        """Count elements.

        Returns:
            Number of elements
        """
        return len(self.collect())

    def min(self) -> T | None:
        """Find minimum element.

        Returns:
            Minimum element or None if empty
        """
        items = self.collect()
        return min(items) if items else None

    def max(self) -> T | None:
        """Find maximum element.

        Returns:
            Maximum element or None if empty
        """
        items = self.collect()
        return max(items) if items else None

    def any(self, pred: Callable[[T], bool]) -> bool:
        """Check if any element satisfies predicate.

        Args:
            pred: Predicate function

        Returns:
            True if any element satisfies predicate
        """
        # Short-circuit: could be optimized further
        items = self.collect()
        return any(pred(item) for item in items)

    def all(self, pred: Callable[[T], bool]) -> bool:
        """Check if all elements satisfy predicate.

        Args:
            pred: Predicate function

        Returns:
            True if all elements satisfy predicate
        """
        # Short-circuit: could be optimized further
        items = self.collect()
        return all(pred(item) for item in items)

    def for_each(self, func: Callable[[T], None]) -> None:
        """Execute function on each element for side effects.

        Args:
            func: Function to execute
        """
        self.map(func).collect()

    def to_dict(self, key_func: Callable[[T], Any]) -> dict[Any, T]:
        """Convert to dictionary using key function.

        Args:
            key_func: Function to extract key

        Returns:
            Dictionary mapping keys to values
        """
        items = self.collect()
        return {key_func(item): item for item in items}

    def _auto_chunk_size(self, total_items: int) -> int:
        """Compute optimal chunk size.

        Args:
            total_items: Total number of items

        Returns:
            Chunk size
        """
        # Heuristic: aim for num_workers * 4 chunks (load balancing)
        num_workers = os.cpu_count() or 4
        target_chunks = num_workers * 4
        chunk_size = max(1, total_items // target_chunks)
        return chunk_size

    def _chunk_iterable(self, items: list[T], chunk_size: int) -> list[list[T]]:
        """Split items into chunks.

        Args:
            items: Items to chunk
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i : i + chunk_size])
        return chunks

    def _tree_reduce(
        self,
        items: list[T],
        identity: U,
        op: Callable[[U, T], U],
    ) -> U:
        """Perform tree reduction for associative operations.

        Args:
            items: Items to reduce
            identity: Identity element
            op: Binary operation

        Returns:
            Reduced value
        """
        if not items:
            return identity

        result = identity
        for item in items:
            result = op(result, item)
        return result


def par_iter(
    iterable: Iterable[T], chunk_size: int | None = None
) -> ParallelIterator[T]:
    """Create a parallel iterator from an iterable.

    Args:
        iterable: Source iterable
        chunk_size: Chunk size for parallelism (None = auto)

    Returns:
        ParallelIterator instance
    """
    return ParallelIterator(iterable, chunk_size)
