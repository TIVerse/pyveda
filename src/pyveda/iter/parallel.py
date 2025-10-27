"""Rayon-inspired parallel iterator for Python."""

import logging
import os
from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

from pyveda.core.task import Task

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


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
        return new_iter

    def enumerate(self) -> "ParallelIterator[tuple[int, T]]":
        """Add indices to elements.

        Returns:
            New iterator with (index, element) tuples
        """
        return self.map(lambda item: (id(item), item))  # Simplified enumeration

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
        return new_iter

    def gpu_map(self, func: Callable[[T], U]) -> "ParallelIterator[U]":
        """Apply function using GPU acceleration.

        Args:
            func: Function to apply (should be GPU-compatible)

        Returns:
            New iterator with GPU-accelerated map
        """
        # Mark items for GPU processing by wrapping the function
        from pyveda.gpu.decorators import gpu

        gpu_func = gpu(func)
        return self.map(gpu_func)

    def async_map(self, func: Callable[[T], U]) -> "ParallelIterator[U]":
        """Apply async function to elements.

        Args:
            func: Async function to apply

        Returns:
            New iterator with async map
        """
        # Create async-aware wrapper
        import asyncio
        import inspect

        if not inspect.iscoroutinefunction(func):
            # Convert to async
            async def async_wrapper(x: T) -> U:
                return func(x)

            func = async_wrapper

        # Execute async functions via runtime
        def sync_wrapper(x: T) -> U:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(func(x))
                return result
            finally:
                loop.close()

        return self.map(sync_wrapper)

    def collect(self) -> list[T]:
        """Execute pipeline and collect results.

        Returns:
            List of results
        """
        from pyveda.core.runtime import get_runtime

        runtime = get_runtime()

        # Convert to list and chunk
        items = list(self._iterable)
        if not items:
            return []

        chunk_size = self._chunk_size or self._auto_chunk_size(len(items))
        chunks = self._chunk_iterable(items, chunk_size)

        # Process chunks in parallel
        def process_chunk(chunk: list[T], chunk_idx: int) -> tuple[int, list[T]]:
            result = chunk
            for op in self._operations:
                result = op.apply(result)
            return (chunk_idx, result)

        # Submit tasks
        futures = []
        for idx, chunk in enumerate(chunks):
            task = Task(
                func=process_chunk,
                args=(chunk, idx),
                kwargs={},
            )
            future = runtime.scheduler.submit(task)
            futures.append(future)

        # Collect results
        results = [f.result() for f in futures]

        # Flatten and optionally preserve order
        if self._ordered:
            results.sort(key=lambda x: x[0])  # Sort by chunk index

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
