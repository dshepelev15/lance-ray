# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import hashlib
import heapq
import logging
import math
import os
import pickle
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any, Literal, Optional, TypeAlias, Union

import lance
import pyarrow as pa
import ray
from lance.dataset import Index, IndexConfig, LanceDataset
from lance.indices import IndicesBuilder
from packaging import version
from ray.util.multiprocessing import Pool

from .field_path import resolve_arrow_field_path, resolve_dataset_field_path
from .utils import (
    get_namespace_kwargs,
    has_namespace_params,
    resolve_namespace_table,
    validate_uri_or_namespace,
)

logger = logging.getLogger(__name__)


_VectorIndexArtifact: TypeAlias = (
    pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray | None
)
_VectorIndexArtifactRef: TypeAlias = _VectorIndexArtifact | ray.ObjectRef
_VectorIndexArtifactRefs: TypeAlias = tuple[
    _VectorIndexArtifactRef, _VectorIndexArtifactRef
]


def _dataset_load_kwargs(
    storage_options: Optional[dict[str, Any]],
    namespace_kwargs: dict[str, Any],
    block_size: Optional[int],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "storage_options": storage_options,
        **namespace_kwargs,
    }
    if block_size is not None:
        kwargs["block_size"] = block_size
    return kwargs


def _index_exists(dataset: LanceDataset, name: str) -> bool:
    return any(index.name == name for index in dataset.describe_indices())


def _distribute_fragments_balanced(
    fragments: list[Any], num_workers: int, logger: logging.Logger
) -> list[list[int]]:
    """Distribute fragments across workers using a balanced algorithm.

    This function implements a greedy algorithm that assigns fragments to the
    worker with the currently smallest total workload, helping to balance the
    processing time across workers.

    Parameters
    ----------
    fragments : list
        List of Lance fragment objects.
    num_workers : int
        Number of workers to distribute fragments across.
    logger : logging.Logger
        Logger instance for debugging information.

    Returns
    -------
    list[list[int]]
        Each inner list contains fragment IDs for one worker.
    """
    if not fragments:
        return [[] for _ in range(num_workers)]

    fragment_info: list[dict[str, int]] = []
    for fragment in fragments:
        try:
            # Try to get fragment size information
            # fragment.count_rows() gives us the number of rows in the fragment
            row_count = fragment.count_rows()
            fragment_info.append({"id": fragment.fragment_id, "size": row_count})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Could not get size for fragment %s: %s. Using fragment_id as size estimate.",
                fragment.fragment_id,
                exc,
            )
            fragment_info.append(
                {"id": fragment.fragment_id, "size": fragment.fragment_id}
            )

    # Sort fragments by size in descending order (largest first)
    # This helps with better load balancing using the greedy algorithm
    fragment_info.sort(key=lambda x: x["size"], reverse=True)

    worker_batches: list[list[int]] = [[] for _ in range(num_workers)]
    worker_workloads = [0] * num_workers

    # Greedy assignment: assign each fragment to the worker with minimum workload
    for frag_info in fragment_info:
        # Find the worker with the minimum current workload
        min_workload_idx = min(range(num_workers), key=lambda i: worker_workloads[i])
        worker_batches[min_workload_idx].append(frag_info["id"])
        worker_workloads[min_workload_idx] += frag_info["size"]

    total_size = sum(info["size"] for info in fragment_info)
    logger.info("Fragment distribution statistics:")
    logger.info("  Total fragments: %d", len(fragment_info))
    logger.info("  Total size: %d", total_size)
    logger.info("  Workers: %d", num_workers)

    for i, (batch, workload) in enumerate(
        zip(worker_batches, worker_workloads, strict=False)
    ):
        percentage = (workload / total_size * 100) if total_size > 0 else 0
        logger.info(
            "  Worker %d: %d fragments, workload: %d (%.1f%%)",
            i,
            len(batch),
            workload,
            percentage,
        )

    non_empty_batches = [batch for batch in worker_batches if batch]
    return non_empty_batches


def _map_async_with_pool(
    create_fragment_handler: Callable[[], Any],
    fragment_batches: list[Any],
    *,
    num_workers: int,
    ray_remote_args: Optional[dict[str, Any]],
    error_prefix: str,
    max_task_retries: int = 0,
    on_success: Optional[Callable[[Any, dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Run fragment tasks in a Ray-backed multiprocessing Pool.

    This helper encapsulates the common Pool submission + error wrapping logic
    so that both scalar and vector distributed index builders can share the
    same implementation.

    When ``max_task_retries`` is positive, task payloads whose handler returned
    a ``{"status": "error"}`` result — or whose task failed at the Ray level
    (Ray's Pool yields the exception *object* as the imap value when a task or
    its actor dies) — are resubmitted (handlers must therefore be idempotent).
    Note that Ray Pool actors are not restarted after an actor death by
    default; pass ``{"max_restarts": ...}`` in ``ray_remote_args`` to make
    retries survive worker-process crashes. ``on_success`` is invoked with
    ``(payload, result)`` as soon as a task result with
    ``status == "success"`` is observed, which allows callers to checkpoint
    completed work before the whole map finishes.
    """
    pool = Pool(processes=num_workers, ray_remote_args=ray_remote_args)
    try:
        fragment_handler = create_fragment_handler()

        if max_task_retries == 0 and on_success is None:
            rst_futures = pool.map_async(
                fragment_handler,
                fragment_batches,
                chunksize=1,
            )
            return rst_futures.get()

        results: dict[int, dict[str, Any]] = {}
        pending = list(enumerate(fragment_batches))
        for attempt in range(max_task_retries + 1):
            round_payloads = [payload for _, payload in pending]
            still_failing: list[tuple[int, Any]] = []
            for (payload_index, payload), result in zip(
                pending,
                pool.imap(fragment_handler, round_payloads, chunksize=1),
                strict=True,
            ):
                if isinstance(result, Exception):
                    # Ray's Pool delivers task/actor failures as exception
                    # objects in the imap stream (PoolTaskError wraps the
                    # original exception in .underlying).
                    underlying = getattr(result, "underlying", result)
                    result = {
                        "status": "error",
                        "error": f"{type(underlying).__name__}: {underlying}",
                    }
                results[payload_index] = result
                if result.get("status") == "success":
                    if on_success is not None:
                        on_success(payload, result)
                else:
                    still_failing.append((payload_index, payload))
            pending = still_failing
            if not pending:
                break
            if attempt < max_task_retries:
                logger.warning(
                    "%d task(s) failed; retrying (attempt %d of %d)",
                    len(pending),
                    attempt + 2,
                    max_task_retries + 1,
                )
        return [results[i] for i in range(len(fragment_batches))]
    except Exception as exc:  # pragma: no cover - exercised via integration tests
        raise RuntimeError(f"{error_prefix}: {exc}") from exc
    finally:
        pool.close()
        pool.join()


def _is_ray_object_ref(value: Any) -> bool:
    object_ref_type = getattr(ray, "ObjectRef", None)
    return object_ref_type is not None and isinstance(value, object_ref_type)


def _ray_put_index_artifact(value: Any) -> _VectorIndexArtifactRef:
    if value is None or _is_ray_object_ref(value):
        return value
    return ray.put(value)


def _ray_get_index_artifact(value: Any) -> _VectorIndexArtifact:
    if _is_ray_object_ref(value):
        return ray.get(value)
    return value


def _put_vector_index_artifacts_in_object_store(
    ivf_centroids: pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray | None,
    pq_codebook: pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray | None,
) -> _VectorIndexArtifactRefs:
    return (
        _ray_put_index_artifact(ivf_centroids),
        _ray_put_index_artifact(pq_codebook),
    )


def _build_rabitq_model(*, dimension: int, num_bits: int = 1) -> str:
    """Build one shared RaBitQ rotation model for distributed IVF_RQ shards.

    ``dimension`` is the vector width and must satisfy Lance's IVF_RQ
    requirement that it is divisible by 8. ``num_bits`` controls the number of
    RaBitQ code bits per vector dimension and defaults to Lance's IVF_RQ default
    of 1; supported values are validated by ``lance.lance.indices.build_rq_model``.
    """
    from lance.lance import indices

    return indices.build_rq_model(dimension=dimension, num_bits=num_bits)


_SCALAR_SEGMENT_INDEX_TYPES = {"BTREE", "BITMAP", "INVERTED", "FTS"}


def _scalar_index_type_name(index_type: str | IndexConfig) -> str | None:
    if isinstance(index_type, str):
        return index_type.upper()
    if isinstance(index_type, IndexConfig):
        return index_type.index_type.upper()
    return None


def _handle_scalar_segment_index(
    dataset_uri: str,
    column: str,
    index_type: str | IndexConfig,
    name: str,
    train: bool,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
    **kwargs: Any,
):
    """Create a fragment handler closure for scalar segment index builds."""

    def func(fragment_ids: list[int]) -> dict[str, Any]:
        try:
            if not fragment_ids:
                raise ValueError("fragment_ids cannot be empty")

            for fragment_id in fragment_ids:
                if fragment_id < 0 or fragment_id > 0xFFFFFFFF:
                    raise ValueError(f"Invalid fragment_id: {fragment_id}")

            namespace_kwargs = get_namespace_kwargs(
                namespace_impl, namespace_properties, table_id
            )
            dataset = LanceDataset(
                dataset_uri,
                **_dataset_load_kwargs(storage_options, namespace_kwargs, block_size),
            )

            available_fragments = {f.fragment_id for f in dataset.get_fragments()}
            invalid_fragments = set(fragment_ids) - available_fragments
            if invalid_fragments:
                raise ValueError(f"Fragment IDs {invalid_fragments} do not exist")

            logger.info(
                "Building distributed scalar segment index for fragments %s using "
                "create_index_uncommitted",
                fragment_ids,
            )

            segment_index = dataset.create_index_uncommitted(
                column=column,
                index_type=index_type,
                name=name,
                replace=False,
                train=train,
                storage_options=storage_options,
                fragment_ids=fragment_ids,
                **kwargs,
            )

            logger.info(
                "Fragment scalar segment index created successfully for fragments %s",
                fragment_ids,
            )

            return {
                "status": "success",
                "fragment_ids": fragment_ids,
                "segment_index": segment_index,
            }

        except Exception as exc:  # pragma: no cover - exercised via integration tests
            logger.error(
                "Fragment scalar segment index task failed for fragments %s: %s",
                fragment_ids,
                exc,
            )
            return {
                "status": "error",
                "fragment_ids": fragment_ids,
                "error": str(exc),
            }

    return func


def _handle_fragment_index(
    dataset_uri: str,
    column: str,
    index_type: str | IndexConfig,
    name: str,
    index_uuid: str,
    replace: bool,
    train: bool,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
    **kwargs: Any,
):
    """Create a fragment handler closure for scalar index builds.

    The returned callable can be used with :func:`Pool.map_async` to build
    indices for specific fragments.
    """

    def func(fragment_ids: list[int]) -> dict[str, Any]:
        try:
            if not fragment_ids:
                raise ValueError("fragment_ids cannot be empty")

            for fragment_id in fragment_ids:
                if fragment_id < 0 or fragment_id > 0xFFFFFFFF:
                    raise ValueError(f"Invalid fragment_id: {fragment_id}")

            namespace_kwargs = get_namespace_kwargs(
                namespace_impl, namespace_properties, table_id
            )

            # Load dataset
            dataset = LanceDataset(
                dataset_uri,
                **_dataset_load_kwargs(storage_options, namespace_kwargs, block_size),
            )

            available_fragments = {f.fragment_id for f in dataset.get_fragments()}
            invalid_fragments = set(fragment_ids) - available_fragments
            if invalid_fragments:
                raise ValueError(f"Fragment IDs {invalid_fragments} do not exist")

            logger.info(
                "Building distributed scalar index for fragments %s using create_scalar_index",
                fragment_ids,
            )

            dataset.create_scalar_index(
                column=column,
                index_type=index_type,
                name=name,
                replace=replace,
                train=train,
                index_uuid=index_uuid,
                fragment_ids=fragment_ids,
                **kwargs,
            )

            field_id = resolve_dataset_field_path(dataset, column).field_id

            logger.info(
                "Fragment scalar index created successfully for fragments %s",
                fragment_ids,
            )

            return {
                "status": "success",
                "fragment_ids": fragment_ids,
                "fields": [field_id],
                "uuid": index_uuid,
            }

        except Exception as exc:  # pragma: no cover - exercised via integration tests
            logger.error(
                "Fragment scalar index task failed for fragments %s: %s",
                fragment_ids,
                exc,
            )
            return {
                "status": "error",
                "fragment_ids": fragment_ids,
                "error": str(exc),
            }

    return func


def merge_index_metadata_compat(dataset, index_id, index_type, **kwargs):
    """Call ``merge_index_metadata`` with backwards compatible signature."""
    try:
        return dataset.merge_index_metadata(
            index_id, index_type, batch_readhead=kwargs.get("batch_readhead")
        )
    except TypeError:
        return dataset.merge_index_metadata(index_id)


def create_scalar_index(
    uri: Optional[str] = None,
    *,
    column: str,
    index_type: Literal["BTREE"]
    | Literal["BITMAP"]
    | Literal["LABEL_LIST"]
    | Literal["INVERTED"]
    | Literal["FTS"]
    | Literal["NGRAM"]
    | Literal["ZONEMAP"]
    | IndexConfig,
    table_id: Optional[list[str]] = None,
    name: Optional[str] = None,
    replace: bool = True,
    train: bool = True,
    fragment_ids: Optional[list[int]] = None,
    index_uuid: Optional[str] = None,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> "lance.LanceDataset":
    """Build scalar indices with Ray in a distributed workflow.

    Args:
        uri: The URI of the Lance dataset to build index on. Either uri OR
            (namespace_impl + table_id) must be provided.
        column: Column name to index.
        index_type: Type of index to build ("BTREE", "BITMAP", "LABEL_LIST",
            "INVERTED", "FTS", "NGRAM", "ZONEMAP") or IndexConfig object.
        table_id: The table identifier as a list of strings. Must be provided
            together with namespace_impl.
        name: Name of the index (generated if None).
        replace: Whether to replace existing index with the same name (default: True).
        train: Whether to train the index (default: True).
        fragment_ids: Optional list of fragment IDs to build index on.
        index_uuid: Optional fragment UUID for distributed indexing.
        num_workers: Number of Ray workers to use (keyword-only).
        storage_options: Storage options for the dataset (keyword-only).
        block_size: Block size in bytes to use when loading the dataset (keyword-only).
        namespace_impl: The namespace implementation type (e.g., "rest", "dir").
            Used together with table_id for resolving the dataset location and
            credentials vending in distributed workers.
        namespace_properties: Properties for connecting to the namespace.
            Used together with namespace_impl and table_id.
        ray_remote_args: Options for Ray tasks (e.g., num_cpus, resources) (keyword-only).
        **kwargs: Additional arguments to pass to create_scalar_index.

    Returns:
        Updated Lance dataset with the index created.

    Raises:
        ValueError: If input parameters are invalid.
        TypeError: If column type is not string.
        RuntimeError: If index building fails or pylance version is incompatible.
    """
    # Check pylance version compatibility
    try:
        lance_version = version.parse(lance.__version__)
        min_required_version = version.parse("0.36.0")

        if lance_version < min_required_version:
            raise RuntimeError(
                "Distributed indexing requires pylance >= 0.36.0, but found "
                f"{lance.__version__}. The distribute-related interfaces are "
                "not available in older versions. Please upgrade pylance by "
                "running: pip install --upgrade pylance"
            )

        logger.info("Pylance version check passed: %s >= 0.36.0", lance.__version__)

    except AttributeError as err:  # pragma: no cover - defensive
        raise RuntimeError(
            "Cannot determine pylance version. Distributed indexing requires "
            "pylance >= 0.36.0. Please upgrade pylance by running: "
            "pip install --upgrade pylance"
        ) from err

    index_id = str(uuid.uuid4())
    logger.info("Starting distributed scalar index build with ID: %s", index_id)

    # Validate uri or namespace params
    validate_uri_or_namespace(uri, namespace_impl, table_id)

    if not column:
        raise ValueError("Column name cannot be empty")

    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    if block_size is not None and block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if isinstance(index_type, str):
        valid_index_types = [
            "BTREE",
            "BITMAP",
            "LABEL_LIST",
            "INVERTED",
            "FTS",
            "NGRAM",
            "ZONEMAP",
        ]
        if index_type not in valid_index_types:
            raise ValueError(
                f"Index type must be one of {valid_index_types}, not '{index_type}'"
            )

        supported_distributed_types = {"INVERTED", "FTS", "BTREE", "BITMAP"}
        if index_type not in supported_distributed_types:
            raise ValueError(
                "Distributed indexing currently supports "
                f"{sorted(supported_distributed_types)} index types, "
                f"not '{index_type}'"
            )
    elif not isinstance(index_type, IndexConfig):
        raise ValueError(
            "index_type must be a string literal or IndexConfig object, got "
            f"{type(index_type)}"
        )

    # Note: Ray initialization is now handled by the Pool, following the pattern from io.py
    # This removes the need for explicit ray.init() calls

    # Resolve URI and get storage options from namespace if provided
    uri, merged_storage_options = resolve_namespace_table(
        uri, storage_options, namespace_impl, namespace_properties, table_id
    )

    namespace_kwargs = get_namespace_kwargs(
        namespace_impl, namespace_properties, table_id
    )

    # Load dataset
    dataset = LanceDataset(
        uri,
        **_dataset_load_kwargs(merged_storage_options, namespace_kwargs, block_size),
    )

    try:
        resolved_column = resolve_dataset_field_path(dataset, column)
    except KeyError as exc:
        available_columns = [field.name for field in dataset.schema]
        raise ValueError(
            f"Column '{column}' not found. Available: {available_columns}"
        ) from exc
    column = resolved_column.path
    field = resolved_column.field

    # Check column type according to index type
    value_type = field.type
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
        value_type = field.type.value_type

    if isinstance(index_type, str):
        match index_type:
            case "INVERTED" | "FTS":
                if not (
                    pa.types.is_string(value_type)
                    or pa.types.is_large_string(value_type)
                ):
                    raise TypeError(
                        f"Column {column} must be string type for {index_type} "
                        f"index, got {value_type}"
                    )
            case "BTREE":
                is_supported = (
                    pa.types.is_integer(value_type)
                    or pa.types.is_floating(value_type)
                    or pa.types.is_string(value_type)
                )
                if not is_supported:
                    raise TypeError(
                        f"Column {column} must be numeric or string type for BTREE "
                        f"index, got {value_type}"
                    )
            case _:
                # For other index types, skip strict validation to maintain compatibility
                pass

    use_segment_workflow = (
        _scalar_index_type_name(index_type) in _SCALAR_SEGMENT_INDEX_TYPES
    )

    if name is None:
        name = f"{column}_idx"

    if replace:
        if _index_exists(dataset, name):
            # Lance 4.0.0: fragment_ids + replace=True may hit an unimplemented path.
            # Implement replace semantics at the driver by dropping the index first.
            dataset.drop_index(name)
            dataset = LanceDataset(
                uri,
                **_dataset_load_kwargs(
                    merged_storage_options, namespace_kwargs, block_size
                ),
            )

    else:
        if _index_exists(dataset, name):
            raise ValueError(
                f"Index with name '{name}' already exists. Set replace=True "
                "to replace it."
            )

    fragments = dataset.get_fragments()
    if not fragments:
        raise ValueError("Dataset contains no fragments")

    if fragment_ids is not None:
        available_fragment_ids = {f.fragment_id for f in fragments}
        invalid_fragments = set(fragment_ids) - available_fragment_ids
        if invalid_fragments:
            raise ValueError(
                f"Fragment IDs {invalid_fragments} do not exist in dataset"
            )
        fragments = [f for f in fragments if f.fragment_id in fragment_ids]
        fragment_ids_to_use = fragment_ids
    else:
        fragment_ids_to_use = [fragment.fragment_id for fragment in fragments]

    if num_workers > len(fragment_ids_to_use):
        num_workers = len(fragment_ids_to_use)
        logger.info("Adjusted num_workers to %d to match fragment count", num_workers)

    fragment_batches = _distribute_fragments_balanced(fragments, num_workers, logger)

    def create_fragment_handler() -> Any:
        if use_segment_workflow:
            return _handle_scalar_segment_index(
                dataset_uri=uri,
                column=column,
                index_type=index_type,
                name=name,
                train=train,
                storage_options=merged_storage_options,
                block_size=block_size,
                namespace_impl=namespace_impl,
                namespace_properties=namespace_properties,
                table_id=table_id,
                **kwargs,
            )

        return _handle_fragment_index(
            dataset_uri=uri,
            column=column,
            index_type=index_type,
            name=name,
            index_uuid=index_id,
            replace=False,
            train=train,
            storage_options=merged_storage_options,
            block_size=block_size,
            namespace_impl=namespace_impl,
            namespace_properties=namespace_properties,
            table_id=table_id,
            **kwargs,
        )

    logger.info(
        "Phase 1: Distributing scalar index build across %d workers for %d fragments",
        len(fragment_batches),
        len(fragment_ids_to_use),
    )

    results = _map_async_with_pool(
        create_fragment_handler=create_fragment_handler,
        fragment_batches=fragment_batches,
        num_workers=num_workers,
        ray_remote_args=ray_remote_args,
        error_prefix="Failed to complete distributed index building",
    )

    failed_results = [r for r in results if r["status"] == "error"]
    if failed_results:
        error_messages = [r["error"] for r in failed_results]
        raise RuntimeError(f"Index building failed: {'; '.join(error_messages)}")

    # Reload dataset to get the latest state after fragment index creation
    dataset = LanceDataset(
        uri,
        **_dataset_load_kwargs(merged_storage_options, namespace_kwargs, block_size),
    )

    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        raise RuntimeError("No successful index creation results found")

    if use_segment_workflow:
        logger.info(
            "Phase 2: Committing scalar distributed index segments for index '%s'",
            name,
        )

        updated_dataset = dataset.commit_existing_index_segments(
            index_name=name,
            column=column,
            segments=[r["segment_index"] for r in successful_results],
        )

        logger.info(
            "Successfully created distributed scalar segment index '%s'",
            name,
        )
        logger.info(
            "Fragments: %d, Workers: %d",
            len(fragment_ids_to_use),
            len(fragment_batches),
        )
        return updated_dataset

    logger.info("Phase 2: Merging index metadata for index ID: %s", index_id)
    # Convert IndexConfig to string for merge_index_metadata which expects a string
    # (lance's create_scalar_index converts IndexConfig to "scalar" internally)
    index_type_str = "scalar" if isinstance(index_type, IndexConfig) else index_type
    merge_index_metadata_compat(dataset, index_id, index_type=index_type_str, **kwargs)

    logger.info("Phase 3: Creating and committing scalar index '%s'", name)

    fields = successful_results[0]["fields"]

    index = Index(
        uuid=index_id,
        name=name,
        fields=fields,
        dataset_version=dataset.version,
        fragment_ids=set(fragment_ids_to_use),
        index_version=0,
    )

    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    updated_dataset = lance.LanceDataset.commit(
        uri,
        create_index_op,
        read_version=dataset.version,
        storage_options=merged_storage_options,
        **namespace_kwargs,
    )

    logger.info(
        "Successfully created distributed scalar index '%s' with three-phase workflow",
        name,
    )
    logger.info(
        "Index ID: %s, Fragments: %d, Workers: %d",
        index_id,
        len(fragment_ids_to_use),
        len(fragment_batches),
    )
    return updated_dataset


# ---------------------------------------------------------------------------
# Distributed vector index support (IVF_* and IVF_HNSW_* families)
# ---------------------------------------------------------------------------

# Vector index types supported by the distributed merge pipeline.
_VECTOR_INDEX_TYPES = {
    "IVF_FLAT",
    "IVF_PQ",
    "IVF_RQ",
    "IVF_SQ",
    "IVF_HNSW_FLAT",
    "IVF_HNSW_PQ",
    "IVF_HNSW_SQ",
}

# Lance's IVF shuffler stores partition offsets in a spill file as 32-bit
# values, so a single uncommitted index segment cannot cover 2^32 or more
# rows. Fragment batches are capped below this limit.
_MAX_ROWS_PER_SEGMENT = (1 << 32) - 1


def _fragment_sizes(
    fragments: list[Any], logger: logging.Logger
) -> tuple[dict[int, int], list[int]]:
    """Return a fragment_id -> row count map plus fragments of unknown size."""
    sizes: dict[int, int] = {}
    unknown: list[int] = []
    for fragment in fragments:
        try:
            sizes[fragment.fragment_id] = fragment.count_rows()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Could not get row count for fragment %s: %s. It will be "
                "placed in its own batch.",
                fragment.fragment_id,
                exc,
            )
            unknown.append(fragment.fragment_id)
    return sizes, unknown


def _plan_fragment_batches(
    fragments: list[Any],
    *,
    num_workers: int,
    rows_per_task: Optional[int],
    logger: logging.Logger,
) -> tuple[list[list[int]], dict[int, int]]:
    """Pack fragments into batches; each batch becomes one index segment.

    Batches are balanced by row count with a greedy largest-first algorithm
    over a min-heap (O(F log B)). ``rows_per_task`` sets a target batch size
    so that large builds produce many more tasks than workers (per-task
    retries stay cheap and stragglers are amortized); when omitted the
    fragments are spread evenly over ``num_workers`` batches. Every batch is
    additionally kept below Lance's hard per-segment limit of 2^32 rows;
    fragments whose row count cannot be determined are isolated in singleton
    batches (a single Lance fragment cannot exceed the per-segment limit).
    """
    sizes, unknown_size_fragments = _fragment_sizes(fragments, logger)
    if not sizes and not unknown_size_fragments:
        return [], {}

    total_rows = sum(sizes.values())
    if rows_per_task is not None:
        target_rows = rows_per_task
    elif total_rows > 0:
        target_rows = math.ceil(total_rows / num_workers)
    else:
        target_rows = 1
    target_rows = min(max(target_rows, 1), _MAX_ROWS_PER_SEGMENT)
    num_batches = num_workers
    if total_rows > 0:
        num_batches = max(num_workers, math.ceil(total_rows / target_rows))

    ordered = sorted(sizes.items(), key=lambda item: (-item[1], item[0]))
    batches: list[list[int]] = [[] for _ in range(num_batches)]
    # Min-heap of (workload, batch_index); ties resolve to the lowest index.
    heap: list[tuple[int, int]] = [(0, i) for i in range(num_batches)]

    for fragment_id, size in ordered:
        if size > _MAX_ROWS_PER_SEGMENT:
            raise ValueError(
                f"Fragment {fragment_id} has {size} rows, which exceeds the "
                f"maximum of {_MAX_ROWS_PER_SEGMENT} rows a single index "
                "segment can cover. Split the fragment (e.g. via compaction) "
                "before building a distributed vector index."
            )
        workload, batch_index = heap[0]
        if workload + size > _MAX_ROWS_PER_SEGMENT:
            # The least-loaded batch cannot fit this fragment, so no batch
            # can: open a new one.
            batches.append([])
            batch_index = len(batches) - 1
            heapq.heappush(heap, (size, batch_index))
        else:
            heapq.heapreplace(heap, (workload + size, batch_index))
        batches[batch_index].append(fragment_id)

    # Fragments whose size is unknown must not be packed with others, or the
    # per-segment row cap could be silently exceeded.
    for fragment_id in unknown_size_fragments:
        batches.append([fragment_id])

    non_empty = [batch for batch in batches if batch]
    logger.info(
        "Planned %d fragment batches over %d fragments (%d total rows, "
        "target %d rows per batch)",
        len(non_empty),
        len(sizes) + len(unknown_size_fragments),
        total_rows,
        target_rows,
    )
    return non_empty, sizes


class _VectorIndexBuildCheckpoint:
    """Driver-local checkpoint store for distributed vector index builds.

    Persists the Phase 1 training artifacts and every successfully built
    (still uncommitted) index segment so that a failed build can be resumed
    without retraining or rebuilding completed segments. Entries are keyed by
    a fingerprint of the build configuration and dataset version; a checkpoint
    written for a different configuration is ignored.
    """

    _PHASE1_FILE = "phase1.pkl"

    def __init__(self, root: str, fingerprint: str):
        self._fingerprint = fingerprint
        self._dir = os.path.join(root, fingerprint[:16])
        self._segments_dir = os.path.join(self._dir, "segments")
        self._merges_dir = os.path.join(self._dir, "merges")
        os.makedirs(self._segments_dir, exist_ok=True)
        os.makedirs(self._merges_dir, exist_ok=True)

    @staticmethod
    def build_fingerprint(**parts: Any) -> str:
        canonical = repr(sorted(parts.items()))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _task_key(parts: Iterable[Any]) -> str:
        canonical = repr(tuple(sorted(str(part) for part in parts)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def _write(self, path: str, payload: Any) -> None:
        # Checkpointing is a resume optimization, not required state: never
        # fail the build because a checkpoint file could not be written.
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
            with open(tmp_path, "wb") as fh:
                pickle.dump({"fingerprint": self._fingerprint, "payload": payload}, fh)
            os.replace(tmp_path, path)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Failed to write checkpoint file %s: %s", path, exc)

    def _read(self, path: str) -> Optional[dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:  # pragma: no cover - corrupted checkpoint
            logger.warning("Ignoring unreadable checkpoint file %s: %s", path, exc)
            return None
        if not isinstance(data, dict) or data.get("fingerprint") != self._fingerprint:
            return None
        return {"payload": data.get("payload")}

    def load_phase1(self) -> Optional[dict[str, Any]]:
        entry = self._read(os.path.join(self._dir, self._PHASE1_FILE))
        return entry["payload"] if entry is not None else None

    def save_phase1(self, payload: dict[str, Any]) -> None:
        self._write(os.path.join(self._dir, self._PHASE1_FILE), payload)

    def _segment_path(self, fragment_ids: Iterable[int]) -> str:
        return os.path.join(self._segments_dir, f"{self._task_key(fragment_ids)}.pkl")

    def load_segment(self, fragment_ids: Iterable[int]) -> Any:
        entry = self._read(self._segment_path(fragment_ids))
        return entry["payload"] if entry is not None else None

    def save_segment(self, fragment_ids: Iterable[int], segment: Any) -> None:
        self._write(self._segment_path(fragment_ids), segment)

    def _merge_path(self, member_keys: Iterable[Any]) -> str:
        return os.path.join(self._merges_dir, f"{self._task_key(member_keys)}.pkl")

    def load_merge(self, member_keys: Iterable[Any]) -> Any:
        entry = self._read(self._merge_path(member_keys))
        return entry["payload"] if entry is not None else None

    def save_merge(self, member_keys: Iterable[Any], segment: Any) -> None:
        self._write(self._merge_path(member_keys), segment)

    def clear(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)


def _artifact_digest(artifact: Any) -> Optional[str]:
    """Content digest of a precomputed training artifact for fingerprinting.

    Distinguishes different centroids/codebooks/models supplied across resume
    attempts so that a checkpoint written for one artifact is never reused
    for another.
    """
    if artifact is None:
        return None
    try:
        if isinstance(artifact, pa.ChunkedArray):
            artifact = artifact.combine_chunks()
        if isinstance(artifact, pa.Array):
            digest = hashlib.sha256(
                f"{artifact.type}:{len(artifact)}:{artifact.offset}".encode()
            )
            for buf in artifact.buffers():
                if buf is not None:
                    digest.update(buf)
            return digest.hexdigest()
        if isinstance(artifact, (str, bytes)):
            data = artifact.encode() if isinstance(artifact, str) else artifact
            return hashlib.sha256(data).hexdigest()
        tobytes = getattr(artifact, "tobytes", None)
        if callable(tobytes):  # numpy-like
            data = tobytes()
            if isinstance(data, bytes):
                digest = hashlib.sha256(
                    f"{getattr(artifact, 'dtype', '')}:{getattr(artifact, 'shape', '')}".encode()
                )
                digest.update(data)
                return digest.hexdigest()
        return hashlib.sha256(pickle.dumps(artifact)).hexdigest()
    except Exception:  # pragma: no cover - defensive
        return "unhashable"


def _segment_files_exist(dataset_uri: str, segment: Any) -> bool:
    """Best-effort probe that a checkpointed segment's files still exist.

    Uncommitted segment directories under ``_indices/<uuid>/`` can be
    reclaimed by ``cleanup_old_versions`` between a failed build and its
    resume; committing metadata that points at reclaimed files would produce
    a broken index. When the storage cannot be probed the checkpoint is
    trusted.
    """
    segment_uuid = getattr(segment, "uuid", None)
    if not segment_uuid:
        return True
    try:
        if os.path.isdir(dataset_uri):
            index_dir = os.path.join(dataset_uri, "_indices", str(segment_uuid))
            return os.path.isdir(index_dir) and bool(os.listdir(index_dir))
        from pyarrow import fs as pa_fs

        filesystem, base_path = pa_fs.FileSystem.from_uri(dataset_uri)
        index_dir = f"{base_path.rstrip('/')}/_indices/{segment_uuid}"
        infos = filesystem.get_file_info(
            pa_fs.FileSelector(index_dir, allow_not_found=True)
        )
        return len(infos) > 0
    except Exception:  # pragma: no cover - environment dependent
        return True


def _vector_dimension(field: pa.Field) -> int:
    if pa.types.is_fixed_size_list(field.type):
        return field.type.list_size
    if isinstance(field.type, pa.FixedShapeTensorType) and len(field.type.shape) == 1:
        return field.type.shape[0]
    raise TypeError(
        f"Vector column must be FixedSizeListArray or 1-dimensional "
        f"FixedShapeTensorArray, got {field.type}"
    )


def _validate_vector_value_type(field: pa.Field) -> None:
    value_type = field.type.value_type
    if not (
        pa.types.is_floating(value_type) or pa.types.is_unsigned_integer(value_type)
    ):
        raise TypeError(
            "Vector column must have floating or unsigned integer value type, "
            f"got {value_type}"
        )


def _schema_names(schema: pa.Schema) -> list[str]:
    names = getattr(schema, "names", None)
    if names is not None:
        return list(names)
    return [field.name for field in schema]


class _NestedVectorIndicesBuilder:
    def __init__(self, dataset: LanceDataset, column: str, field: pa.Field):
        self.dataset = dataset
        self.column = column
        self.dimension = _vector_dimension(field)
        _validate_vector_value_type(field)

    def train_ivf(
        self,
        num_partitions: Optional[int] = None,
        *,
        distance_type: str = "l2",
        sample_rate: int = 256,
        max_iters: int = 50,
        fragment_ids: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("accelerator") is not None:
            raise NotImplementedError(
                "Nested vector IVF training does not support accelerator training"
            )

        from lance.indices.ivf import IvfModel
        from lance.lance import indices

        num_rows = _count_rows_for_fragments(self.dataset, fragment_ids)
        partition_count = _determine_num_partitions(num_partitions, num_rows)
        _verify_ivf_sample_rate(sample_rate, partition_count, num_rows)
        distance_type = _normalize_distance_type(distance_type)
        _verify_num_partitions(partition_count)

        centroids = indices.train_ivf_model(
            self.dataset._ds,
            self.column,
            self.dimension,
            partition_count,
            distance_type,
            sample_rate,
            max_iters,
            fragment_ids,
        )
        return IvfModel(centroids, distance_type)

    def train_pq(
        self,
        ivf_model: Any,
        num_subvectors: Optional[int] = None,
        *,
        sample_rate: int = 256,
        max_iters: int = 50,
        fragment_ids: Optional[list[int]] = None,
    ) -> Any:
        from lance.indices.pq import PqModel
        from lance.lance import indices

        num_rows = _count_rows_for_fragments(self.dataset, fragment_ids)
        num_subvectors = _normalize_pq_params(num_subvectors, self.dimension)
        _verify_pq_sample_rate(num_rows, sample_rate)
        codebook = indices.train_pq_model(
            self.dataset._ds,
            self.column,
            self.dimension,
            num_subvectors,
            ivf_model.distance_type,
            sample_rate,
            max_iters,
            ivf_model.centroids,
            fragment_ids,
        )
        return PqModel(num_subvectors, codebook)


def _count_rows_for_fragments(
    dataset: LanceDataset,
    fragment_ids: Optional[list[int]],
) -> int:
    if fragment_ids is None:
        return dataset.count_rows()

    row_count = 0
    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")
        row_count += fragment.count_rows()
    return row_count


def _determine_num_partitions(num_partitions: Optional[int], num_rows: int) -> int:
    if num_partitions is None:
        return round(math.sqrt(num_rows))
    return num_partitions


def _verify_base_sample_rate(sample_rate: int) -> None:
    if not isinstance(sample_rate, int) or sample_rate < 2:
        raise ValueError(
            f"The sample_rate must be an int greater than 1, got {sample_rate}"
        )


def _verify_ivf_sample_rate(
    sample_rate: int,
    num_partitions: int,
    num_rows: int,
) -> None:
    _verify_base_sample_rate(sample_rate)
    if num_partitions * sample_rate > num_rows:
        raise ValueError(
            "There are not enough rows in the dataset to create IVF centroids with"
            f" {num_partitions} partitions and a sample rate of {sample_rate}."
            f" {sample_rate * num_partitions} rows needed and there are {num_rows}"
        )


def _verify_num_partitions(num_partitions: int) -> None:
    if not isinstance(num_partitions, int):
        raise TypeError(f"num_partitions must be int, got {type(num_partitions)}")


def _normalize_distance_type(distance_type: str) -> str:
    if not isinstance(distance_type, str) or distance_type.lower() not in [
        "l2",
        "cosine",
        "euclidean",
        "dot",
        "hamming",
    ]:
        raise ValueError(f"Distance type {distance_type} not supported.")
    return distance_type.lower()


def _normalize_pq_params(num_subvectors: Optional[int], dimension: int) -> int:
    if num_subvectors is None:
        if dimension % 16 == 0:
            return dimension // 16
        if dimension % 8 == 0:
            return dimension // 8
        raise ValueError(
            f"vector dimension {dimension} is not divisible by 16 or 8."
            " PQ performance will be poor.  Please specify num_subvectors manually."
        )
    if not isinstance(num_subvectors, int):
        raise ValueError("num_subvectors must be an int")
    if num_subvectors < 1:
        raise ValueError("num_subvectors must be greater than 0")
    if num_subvectors > dimension:
        raise ValueError(
            "num_subvectors must be less than or equal to the dimension of the vectors"
        )
    if dimension % num_subvectors != 0:
        raise ValueError(
            f"dimension ({dimension}) must be divisible by num_subvectors "
            f"({num_subvectors}) without remainder"
        )
    return num_subvectors


def _verify_pq_sample_rate(num_rows: int, sample_rate: int) -> None:
    _verify_base_sample_rate(sample_rate)
    if 256 * sample_rate > num_rows:
        raise ValueError(
            "There are not enough rows in the dataset to create PQ codebook with "
            f"a sample rate of {sample_rate}. {sample_rate * 256} rows needed and "
            f"there are {num_rows}"
        )


def _indices_builder_for_field_path(
    dataset: LanceDataset,
    column: str,
    field: pa.Field,
) -> IndicesBuilder | _NestedVectorIndicesBuilder:
    if column in _schema_names(dataset.schema):
        return IndicesBuilder(dataset, column)

    return _NestedVectorIndicesBuilder(dataset, column, field)


def _train_pq_for_field_path(
    builder: IndicesBuilder | _NestedVectorIndicesBuilder,
    ivf_model: Any,
    *,
    num_subvectors: Optional[int],
    sample_rate: int,
    max_iters: int = 50,
) -> Any:
    return builder.train_pq(
        ivf_model,
        num_subvectors=num_subvectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
    )


def _coerce_ivf_centroids(ivf_centroids: Any, dimension: int) -> Any:
    """Normalize precomputed IVF centroids to a FixedSizeListArray.

    Accepts a FixedSizeListArray / 1-D FixedShapeTensorArray of width
    ``dimension``, a flat numeric ``pa.Array`` whose length is a multiple of
    ``dimension``, or a 2-D numpy array of shape ``(num_partitions,
    dimension)``.
    """
    if isinstance(ivf_centroids, pa.FixedShapeTensorArray):
        shape = ivf_centroids.type.shape
        if len(shape) != 1 or shape[0] != dimension:
            raise ValueError(
                f"ivf_centroids tensor shape {shape} does not match the vector "
                f"dimension {dimension}"
            )
        return ivf_centroids.storage

    if isinstance(ivf_centroids, pa.ChunkedArray):
        ivf_centroids = ivf_centroids.combine_chunks()

    if isinstance(ivf_centroids, pa.Array):
        if pa.types.is_fixed_size_list(ivf_centroids.type):
            if ivf_centroids.type.list_size != dimension:
                raise ValueError(
                    f"ivf_centroids list size {ivf_centroids.type.list_size} "
                    f"does not match the vector dimension {dimension}"
                )
            return ivf_centroids
        if len(ivf_centroids) == 0 or len(ivf_centroids) % dimension != 0:
            raise ValueError(
                f"Flat ivf_centroids length {len(ivf_centroids)} is not a "
                f"positive multiple of the vector dimension {dimension}"
            )
        return pa.FixedSizeListArray.from_arrays(ivf_centroids, dimension)

    shape = getattr(ivf_centroids, "shape", None)
    if shape is not None and len(shape) == 2:
        if shape[1] != dimension:
            raise ValueError(
                f"ivf_centroids shape {tuple(shape)} does not match the "
                f"vector dimension {dimension}"
            )
        values = pa.array(ivf_centroids.reshape(-1))
        return pa.FixedSizeListArray.from_arrays(values, dimension)

    raise TypeError(
        "ivf_centroids must be a FixedSizeListArray, a 1-D "
        "FixedShapeTensorArray, a flat pyarrow Array, or a 2-D numpy array; "
        f"got {type(ivf_centroids)}"
    )


def _num_partitions_from_centroids(ivf_centroids: Any) -> int:
    """Derive the IVF partition count from normalized centroids."""
    num_partitions = len(ivf_centroids)
    if num_partitions < 1:
        raise ValueError("ivf_centroids must contain at least one centroid")
    return num_partitions


def _validate_pq_codebook(pq_codebook: Any, dimension: int) -> Any:
    """Validate the shape of a precomputed PQ codebook where possible.

    The PqModel layout is a FixedSizeListArray whose list size equals the
    full vector dimension (one row per PQ code word). Shapes that cannot be
    checked here are passed through to Lance, which performs its own
    validation.
    """
    if isinstance(pq_codebook, pa.ChunkedArray):
        pq_codebook = pq_codebook.combine_chunks()
    if isinstance(pq_codebook, pa.Array) and pa.types.is_fixed_size_list(
        pq_codebook.type
    ):
        if pq_codebook.type.list_size != dimension:
            raise ValueError(
                f"pq_codebook list size {pq_codebook.type.list_size} does not "
                f"match the vector dimension {dimension}; expected the "
                "PqModel layout with one full-dimension row per code word"
            )
        return pq_codebook
    shape = getattr(pq_codebook, "shape", None)
    if shape is not None and len(shape) == 2 and shape[1] != dimension:
        raise ValueError(
            f"pq_codebook shape {tuple(shape)} does not match the vector "
            f"dimension {dimension}"
        )
    return pq_codebook


def _normalize_index_type(index_type: Any) -> str:
    """Normalize index type to upper-case string and validate support.

    Parameters
    ----------
    index_type : str or enum-like
        Vector index type. Must be one of the precise distributed vector
        types supported by Lance.
    """

    if hasattr(index_type, "value") and isinstance(index_type.value, str):
        index_type_name = index_type.value.upper()
    elif isinstance(index_type, str):
        index_type_name = index_type.upper()
    else:
        raise TypeError(
            "index_type must be a string or an enum-like object with a string 'value' "
            f"attribute, got {type(index_type)}"
        )

    if index_type_name not in _VECTOR_INDEX_TYPES:
        raise ValueError(
            "Distributed vector indexing only supports the following index types: "
            f"{sorted(_VECTOR_INDEX_TYPES)}, not '{index_type_name}'"
        )

    return index_type_name


def _check_pylance_version() -> None:
    """Ensure pylance (lance) provides distributed vector APIs."""

    try:
        lance_version = version.parse(lance.__version__)
        min_required_version = version.parse("0.36.0")

        if lance_version < min_required_version:
            raise RuntimeError(
                "Distributed vector indexing requires pylance >= 0.36.0, but found "
                f"{lance.__version__}. The distributed vector interfaces are not "
                "available in older versions. Please upgrade pylance by running: "
                "pip install --upgrade pylance"
            )

        logger.info("Pylance version check passed: %s >= 0.36.0", lance.__version__)

    except AttributeError as err:  # pragma: no cover - defensive
        raise RuntimeError(
            "Cannot determine pylance version. Distributed vector indexing requires "
            "pylance >= 0.36.0. Please upgrade pylance by running: "
            "pip install --upgrade pylance"
        ) from err


def _validate_metric(metric: str) -> str:
    """Normalize and validate the distance metric string."""

    if not isinstance(metric, str):
        raise TypeError(f"Metric must be a string, got {type(metric)}")

    metric_lower = metric.lower()
    valid_metrics = {"l2", "cosine", "euclidean", "dot", "hamming"}
    if metric_lower not in valid_metrics:
        raise ValueError(
            f"Metric {metric} not supported. Valid: {sorted(valid_metrics)}"
        )
    return metric_lower


def _dataset_load_kwargs_with_manifest(
    storage_options: Optional[dict[str, Any]],
    namespace_kwargs: dict[str, Any],
    block_size: Optional[int],
    dataset_version: Optional[int],
    serialized_manifest: Any,
) -> dict[str, Any]:
    """Build dataset load kwargs, reusing a broadcast manifest when available.

    Passing the driver's serialized manifest (and pinning its version) lets
    every worker skip re-reading the manifest from object storage and
    guarantees the workers see the same fragment set the driver planned
    against.
    """
    load_kwargs = _dataset_load_kwargs(storage_options, namespace_kwargs, block_size)
    manifest = _ray_get_index_artifact(serialized_manifest)
    if manifest is not None:
        load_kwargs["serialized_manifest"] = manifest
        if dataset_version is not None:
            load_kwargs["version"] = dataset_version
    return load_kwargs


# Default scan batch (rows) for worker-side index builds. Large enough to
# keep object-store reads and SIMD pipelines busy, and far below the arrow-rs
# FixedSizeList take() overflow threshold for any realistic dimension.
_DEFAULT_INDEX_BUILD_BATCH_ROWS = 131072
# Cap batches so rows * dim stays under 2^31 — half the u32 threshold at
# which arrow-rs take() panics on (or silently corrupts) FixedSizeList data.
_SAFE_FSL_TAKE_ELEMENTS = 1 << 31


def _safe_index_build_batch_rows(dimension: Optional[int]) -> int:
    batch_rows = _DEFAULT_INDEX_BUILD_BATCH_ROWS
    if dimension and dimension > 0:
        batch_rows = min(batch_rows, _SAFE_FSL_TAKE_ELEMENTS // dimension)
    return max(batch_rows, 1024)


def _handle_vector_fragment_index(
    dataset_uri: str,
    column: str,
    index_type: str,
    name: str,
    index_uuid: str,
    replace: bool,
    metric: str,
    num_partitions: Optional[int],
    num_sub_vectors: Optional[int],
    ivf_centroids: pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray | None,
    pq_codebook: pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray | None,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
    dataset_version: Optional[int] = None,
    serialized_manifest: Any = None,
    dimension: Optional[int] = None,
    **kwargs: Any,
):
    """Create a fragment handler closure for vector index builds."""

    def func(fragment_ids: list[int]) -> dict[str, Any]:
        try:
            # Guard against an arrow-rs u32 overflow in FixedSizeList take():
            # a large dataset block_size inflates the default scan batch to
            # block_size/4 rows, and any batch whose rows * dim exceeds 2^32
            # (e.g. >5.59M rows at dim=768) panics — or, past the wrap
            # point, silently corrupts vectors — inside Lance's
            # KeepFiniteVectors transform during the index build.
            os.environ.setdefault(
                "LANCE_DEFAULT_BATCH_SIZE",
                str(_safe_index_build_batch_rows(dimension)),
            )

            if not fragment_ids:
                raise ValueError("fragment_ids cannot be empty")

            for fragment_id in fragment_ids:
                if fragment_id < 0 or fragment_id > 0xFFFFFFFF:
                    raise ValueError(f"Invalid fragment_id: {fragment_id}")

            namespace_kwargs = get_namespace_kwargs(
                namespace_impl, namespace_properties, table_id
            )
            dataset = LanceDataset(
                dataset_uri,
                **_dataset_load_kwargs_with_manifest(
                    storage_options,
                    namespace_kwargs,
                    block_size,
                    dataset_version,
                    serialized_manifest,
                ),
            )
            available_fragments = {f.fragment_id for f in dataset.get_fragments()}
            invalid_fragments = set(fragment_ids) - available_fragments
            if invalid_fragments:
                raise ValueError(f"Fragment IDs {invalid_fragments} do not exist")

            logger.info(
                "Building distributed vector index for fragments %s using "
                "LanceDataset.create_index",
                fragment_ids,
            )

            resolved_ivf_centroids = _ray_get_index_artifact(ivf_centroids)
            resolved_pq_codebook = _ray_get_index_artifact(pq_codebook)

            segment_index = dataset.create_index_uncommitted(
                column=column,
                index_type=index_type,
                name=name,
                metric=metric,
                replace=replace,
                num_partitions=num_partitions,
                ivf_centroids=resolved_ivf_centroids,
                pq_codebook=resolved_pq_codebook,
                num_sub_vectors=num_sub_vectors,
                storage_options=storage_options,
                train=True,
                fragment_ids=fragment_ids,
                **kwargs,
            )

            logger.info(
                "Fragment vector index created successfully for fragments %s",
                fragment_ids,
            )

            return {
                "status": "success",
                "fragment_ids": fragment_ids,
                "segment_index": segment_index,
                "uuid": getattr(segment_index, "uuid", index_uuid),
            }

        except Exception as exc:  # pragma: no cover - exercised via integration tests
            logger.error(
                "Fragment vector index task failed for fragments %s: %s",
                fragment_ids,
                exc,
            )
            return {
                "status": "error",
                "fragment_ids": fragment_ids,
                "error": str(exc),
            }

    return func


def _handle_merge_segments(
    dataset_uri: str,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
    dataset_version: Optional[int] = None,
    serialized_manifest: Any = None,
):
    """Create a handler that physically merges one group of index segments.

    Segments built with shared IVF/PQ/RaBitQ artifacts can be merged into a
    single physical segment before commit; running the merges on Ray workers
    parallelizes what would otherwise be a single-node post-processing step.
    """

    def func(segment_group: list[Any]) -> dict[str, Any]:
        try:
            segments = list(segment_group)
            if not segments:
                raise ValueError("segment group cannot be empty")

            namespace_kwargs = get_namespace_kwargs(
                namespace_impl, namespace_properties, table_id
            )
            dataset = LanceDataset(
                dataset_uri,
                **_dataset_load_kwargs_with_manifest(
                    storage_options,
                    namespace_kwargs,
                    block_size,
                    dataset_version,
                    serialized_manifest,
                ),
            )

            merged = dataset.merge_existing_index_segments(segments)
            logger.info(
                "Merged %d index segments into segment %s",
                len(segments),
                getattr(merged, "uuid", "<unknown>"),
            )
            return {"status": "success", "segment_index": merged}

        except Exception as exc:  # pragma: no cover - exercised via integration tests
            logger.error(
                "Index segment merge task failed for %d segments: %s",
                len(segment_group),
                exc,
            )
            return {"status": "error", "error": str(exc)}

    return func


def _plan_segment_merge_groups(
    results: list[dict[str, Any]],
    fragment_sizes: dict[int, int],
    target_num_segments: int,
) -> list[list[Any]]:
    """Greedily balance built segments into ``target_num_segments`` groups."""
    entries = []
    for result in results:
        segment = result["segment_index"]
        rows = sum(
            fragment_sizes.get(fragment_id, 0)
            for fragment_id in result.get("fragment_ids", [])
        )
        entries.append((rows, str(getattr(segment, "uuid", "")), segment))
    entries.sort(key=lambda entry: (-entry[0], entry[1]))

    groups: list[list[Any]] = [[] for _ in range(target_num_segments)]
    workloads = [0] * target_num_segments
    for rows, _, segment in entries:
        group_index = min(range(target_num_segments), key=lambda i: workloads[i])
        groups[group_index].append(segment)
        workloads[group_index] += rows
    return [group for group in groups if group]


_INDEX_COMMIT_MAX_ATTEMPTS = 3
_INDEX_COMMIT_RETRY_DELAY_S = 1.0


class _IndexCommitCoverageError(RuntimeError):
    """The dataset changed between segment build and commit; not retryable."""


def _commit_index_segments_with_retry(
    load_dataset: Callable[[], Any],
    *,
    index_name: str,
    column: str,
    segments: list[Any],
    covered_fragment_ids: Optional[set[int]] = None,
    built_version: Optional[int] = None,
) -> Any:
    """Commit index segments, retrying transient failures with backoff.

    The dataset handle is reloaded before every attempt so that a retry
    commits against the latest version (e.g. after a concurrent-writer
    conflict). When ``covered_fragment_ids`` is given, each attempt first
    verifies that every fragment the segments were built against still exists
    in the commit-target version — segments are built against a pinned
    dataset version, and committing them after those fragments were deleted
    or compacted away would publish an index over vanished data.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_INDEX_COMMIT_MAX_ATTEMPTS):
        try:
            dataset = load_dataset()
            if covered_fragment_ids is not None:
                current_fragment_ids = {
                    fragment.fragment_id for fragment in dataset.get_fragments()
                }
                missing = covered_fragment_ids - current_fragment_ids
                if missing:
                    raise _IndexCommitCoverageError(
                        f"Dataset changed between index build (version "
                        f"{built_version}) and commit (version "
                        f"{getattr(dataset, 'version', 'unknown')}): fragments "
                        f"{sorted(missing)[:10]} no longer exist. Rebuild the "
                        "index against the current dataset version."
                    )
            return dataset.commit_existing_index_segments(
                index_name=index_name,
                column=column,
                segments=segments,
            )
        except _IndexCommitCoverageError:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < _INDEX_COMMIT_MAX_ATTEMPTS - 1:
                delay = _INDEX_COMMIT_RETRY_DELAY_S * (2**attempt)
                logger.warning(
                    "commit_existing_index_segments failed (attempt %d of %d): "
                    "%s; retrying in %.1fs",
                    attempt + 1,
                    _INDEX_COMMIT_MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
    if last_exc is None:  # pragma: no cover - loop always sets last_exc
        raise RuntimeError("commit_existing_index_segments failed")
    raise last_exc


def _train_vector_index_models(
    dataset: LanceDataset,
    *,
    column: str,
    index_type_name: str,
    metric: str,
    num_partitions: Optional[int],
    num_sub_vectors: Optional[int],
    ivf_centroids: Any,
    pq_codebook: Any,
    ivf_sample_rate: int,
    pq_sample_rate: int,
    max_iters: int,
    accelerator: Any,
    build_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Phase 1: resolve the IVF/PQ/RaBitQ artifacts shared by every segment.

    Precomputed artifacts pass through untouched; the rest are trained on
    ``dataset``. ``build_kwargs`` are the extra :func:`create_index` keywords
    (``rabitq_model``, ``num_bits``). Runs on the driver, or as a Ray task
    when :func:`create_index` is given ``training_ray_remote_args``: a
    ``LanceDataset`` pickles with its version and manifest, so the task sees
    exactly the data the build was planned on.
    """
    rabitq_model = build_kwargs.get("rabitq_model")
    num_bits = build_kwargs.get("num_bits", 1)
    field = resolve_arrow_field_path(dataset.schema, column).field
    builder = _indices_builder_for_field_path(dataset, column, field)
    dimension = builder.dimension

    if ivf_centroids is not None:
        num_partitions = _num_partitions_from_centroids(ivf_centroids)
        logger.info(
            "Phase 1: Using precomputed IVF centroids "
            "(num_partitions=%d); skipping IVF training",
            num_partitions,
        )
    else:
        logger.info(
            "Phase 1: Training IVF centroids (index_type=%s, metric=%s, "
            "requested_num_partitions=%s, num_rows=%d, dimension=%d, "
            "sample_rate=%d, max_iters=%d, accelerator=%s)",
            index_type_name,
            metric,
            num_partitions,
            dataset.count_rows(),
            dimension,
            ivf_sample_rate,
            max_iters,
            accelerator,
        )
        train_ivf_kwargs: dict[str, Any] = {
            "num_partitions": num_partitions,
            "distance_type": metric,
            "sample_rate": ivf_sample_rate,
            "max_iters": max_iters,
        }
        if accelerator is not None:
            train_ivf_kwargs["accelerator"] = accelerator
        ivf_model = builder.train_ivf(**train_ivf_kwargs)
        ivf_centroids = ivf_model.centroids
        num_partitions = ivf_model.num_partitions
        logger.info("IVF training completed: num_partitions=%d", num_partitions)

    if index_type_name in {"IVF_PQ", "IVF_HNSW_PQ"}:
        if pq_codebook is not None:
            if num_sub_vectors is None:
                raise ValueError(
                    "num_sub_vectors must be provided together with a "
                    "precomputed pq_codebook"
                )
            num_sub_vectors = _normalize_pq_params(num_sub_vectors, dimension)
            logger.info(
                "Phase 1: Using precomputed PQ codebook "
                "(num_sub_vectors=%d); skipping PQ training",
                num_sub_vectors,
            )
        else:
            from lance.indices.ivf import IvfModel

            ivf_model = IvfModel(ivf_centroids, metric)
            logger.info(
                "Training PQ codebook: requested_num_sub_vectors=%s, sample_rate=%d",
                num_sub_vectors,
                pq_sample_rate,
            )
            pq_model = _train_pq_for_field_path(
                builder,
                ivf_model,
                num_subvectors=num_sub_vectors,
                sample_rate=pq_sample_rate,
                max_iters=max_iters,
            )
            pq_codebook = pq_model.codebook
            num_sub_vectors = pq_model.num_subvectors
            logger.info("PQ training completed: num_sub_vectors=%d", num_sub_vectors)

    if index_type_name == "IVF_RQ" and rabitq_model is None:
        logger.info(
            "Building shared RaBitQ model: dimension=%d, num_bits=%s",
            dimension,
            num_bits,
        )
        rabitq_model = _build_rabitq_model(dimension=dimension, num_bits=num_bits)

    return {
        "ivf_centroids": ivf_centroids,
        "pq_codebook": pq_codebook,
        "num_partitions": num_partitions,
        "num_sub_vectors": num_sub_vectors,
        "rabitq_model": rabitq_model,
    }


def create_index(
    uri: Optional[Union[str, "lance.LanceDataset"]] = None,
    column: str = "",
    index_type: str | Any = "",
    name: Optional[str] = None,
    *,
    replace: bool = True,
    num_workers: int = 4,
    storage_options: Optional[dict[str, str]] = None,
    block_size: Optional[int] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    table_id: Optional[list[str]] = None,
    ray_remote_args: Optional[dict[str, Any]] = None,
    metric: str = "l2",
    num_partitions: Optional[int] = None,
    num_sub_vectors: Optional[int] = None,
    sample_rate: int = 256,
    ivf_sample_rate: Optional[int] = None,
    pq_sample_rate: Optional[int] = None,
    max_iters: int = 50,
    accelerator: Optional[Any] = None,
    training_ray_remote_args: Optional[dict[str, Any]] = None,
    ivf_centroids: Optional[
        pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray
    ] = None,
    pq_codebook: Optional[
        pa.Array | pa.FixedSizeListArray | pa.FixedShapeTensorArray
    ] = None,
    rabitq_model: Optional[str] = None,
    rows_per_task: Optional[int] = None,
    max_task_retries: int = 2,
    checkpoint_dir: Optional[str] = None,
    target_num_segments: Optional[int] = None,
    **kwargs: Any,
) -> "lance.LanceDataset":
    """Build distributed vector indices with Ray.

    This function mirrors :func:`create_scalar_index` but targets the precise
    vector index families supported by Lance's distributed merge pipeline.

    Args:
        uri: Lance dataset or URI to build index on
        column: Column name to index
        index_type: Type of index to build (e.g., "IVF_PQ", "IVF_RQ",
            "IVF_HNSW_PQ")
        name: Name of the index (generated if None)
        replace: Whether to replace existing index with the same name (default: True)
        num_workers: Number of Ray workers to use (keyword-only)
        storage_options: Storage options for the dataset (keyword-only)
        block_size: Block size in bytes to use when loading the dataset (keyword-only)
        ray_remote_args: Options for Ray tasks (keyword-only)
        metric: Distance metric to use (default: "l2")
        num_partitions: Number of IVF partitions (optional)
        num_sub_vectors: Number of PQ sub-vectors (optional). Required when a
            precomputed ``pq_codebook`` is supplied.
        sample_rate: Number of rows sampled per IVF partition and PQ centroid
            (default: 256)
        ivf_sample_rate: Overrides ``sample_rate`` for IVF centroid training.
            The IVF training sample is ``ivf_sample_rate * num_partitions``
            rows, so lowering this bounds driver memory on large partition
            counts (optional)
        pq_sample_rate: Overrides ``sample_rate`` for PQ codebook training
            (optional)
        max_iters: Maximum k-means iterations for IVF and PQ training
            (default: 50)
        accelerator: Accelerator for Phase 1 IVF centroid training, e.g.
            ``"cuda"`` or ``"mps"`` (optional). Applies to Phase 1 training
            only; it is intentionally not forwarded to fragment workers
            because torch inputs disable Lance's distributed build path.
        training_ray_remote_args: When set, Phase 1 (IVF/PQ training and the
            RaBitQ model) runs as a Ray task with these options instead of on
            the driver, e.g. ``{"num_cpus": 16, "memory": 64 * 2**30}`` to
            keep a resource-poor head node out of k-means, or
            ``{"num_gpus": 1}`` together with ``accelerator="cuda"`` to train
            on a GPU node. The task receives the dataset pinned to the
            version the build was planned on (optional)
        ivf_centroids: Pre-computed IVF centroids. When provided, driver-side
            IVF training is skipped and ``num_partitions`` is derived from the
            centroids. Accepts a FixedSizeListArray, a 1-D
            FixedShapeTensorArray, a flat pyarrow Array, or a 2-D numpy array.
        pq_codebook: Pre-computed PQ codebook. When provided together with
            ``num_sub_vectors``, driver-side PQ training is skipped. The
            codebook must have been trained against the same IVF centroids
            and metric used for this build.
        rabitq_model: Pre-built RaBitQ model for IVF_RQ. If omitted, Lance-Ray
            builds one shared model on the driver and sends it to every worker.
            The model dimension is the vector column width and must be divisible
            by 8. The ``num_bits`` keyword controls RaBitQ bits per vector
            dimension and defaults to 1.
        rows_per_task: Target number of rows per fragment batch. Each batch
            becomes one Ray task and one uncommitted index segment; setting
            this well below ``total_rows / num_workers`` yields many small
            tasks, which improves retry granularity and straggler behavior on
            large builds. Batches are always capped below Lance's hard limit
            of 2^32 rows per segment (optional)
        max_task_retries: How many times failed fragment batches (and merge
            groups) are resubmitted before the build fails (default: 2)
        checkpoint_dir: Driver-local directory used to persist Phase 1
            training artifacts and completed segments. Re-running the same
            build with the same checkpoint_dir resumes from the checkpoint
            instead of retraining/rebuilding; the checkpoint is removed after
            a successful commit (optional)
        target_num_segments: When set and fewer than the number of built
            segments, segments are physically merged on Ray workers into this
            many segments before commit. Fewer committed segments make
            queries cheaper (query cost grows with the number of index
            segments), at the price of a merge pass over the quantized codes
            (optional)
        **kwargs: Additional arguments to pass to the fragment index build entrypoint.

    Returns:
        Updated Lance dataset with the index created
    """

    _check_pylance_version()

    if not column:
        raise ValueError("Column name cannot be empty")

    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    for rate_name, rate_value in (
        ("ivf_sample_rate", ivf_sample_rate),
        ("pq_sample_rate", pq_sample_rate),
    ):
        if rate_value is not None and rate_value <= 0:
            raise ValueError(f"{rate_name} must be positive, got {rate_value}")

    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")

    if rows_per_task is not None and rows_per_task <= 0:
        raise ValueError(f"rows_per_task must be positive, got {rows_per_task}")

    if max_task_retries < 0:
        raise ValueError(
            f"max_task_retries must be non-negative, got {max_task_retries}"
        )

    if target_num_segments is not None and target_num_segments < 1:
        raise ValueError(
            f"target_num_segments must be at least 1, got {target_num_segments}"
        )

    if pq_codebook is not None and ivf_centroids is None:
        raise ValueError(
            "ivf_centroids must be provided when pq_codebook is provided; a "
            "PQ codebook is only valid for the IVF centroids it was trained "
            "against"
        )

    if block_size is not None and block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    index_type_name = _normalize_index_type(index_type)
    metric_lower = _validate_metric(metric)

    index_id = str(uuid.uuid4())
    logger.info("Starting distributed vector index build with ID: %s", index_id)

    merged_storage_options: dict[str, Any] = {}
    if storage_options:
        merged_storage_options.update(storage_options)

    if isinstance(uri, str | type(None)):
        # URI or namespace mode
        validate_uri_or_namespace(uri, namespace_impl, table_id)

        # Resolve URI and storage options from namespace if provided
        uri, merged_storage_options = resolve_namespace_table(
            uri, storage_options, namespace_impl, namespace_properties, table_id
        )
        dataset_uri = uri
        namespace_kwargs = get_namespace_kwargs(
            namespace_impl, namespace_properties, table_id
        )
        dataset_obj = LanceDataset(
            dataset_uri,
            **_dataset_load_kwargs(
                merged_storage_options, namespace_kwargs, block_size
            ),
        )
    else:
        # LanceDataset object passed directly
        dataset_obj = uri
        dataset_uri = dataset_obj.uri
        if not merged_storage_options:
            merged_storage_options = (
                getattr(dataset_obj, "_storage_options", None) or {}
            )
        namespace_kwargs = {}

    try:
        resolved_column = resolve_arrow_field_path(dataset_obj.schema, column)
    except KeyError as exc:
        available_columns = [field.name for field in dataset_obj.schema]
        raise ValueError(
            f"Column '{column}' not found. Available: {available_columns}"
        ) from exc
    column = resolved_column.path
    field = resolved_column.field

    if name is None:
        name = f"{column}_idx"

    if not replace and _index_exists(dataset_obj, name):
        raise ValueError(
            f"Index with name '{name}' already exists. Set replace=True to replace it."
        )

    fragments = dataset_obj.get_fragments()
    if not fragments:
        raise ValueError("Dataset contains no fragments")

    fragment_ids_to_use = [fragment.fragment_id for fragment in fragments]

    if num_workers > len(fragment_ids_to_use):
        num_workers = len(fragment_ids_to_use)
        logger.info("Adjusted num_workers to %d to match fragment count", num_workers)

    index_build_kwargs = dict(kwargs)
    if rabitq_model is not None:
        index_build_kwargs["rabitq_model"] = rabitq_model

    effective_ivf_sample_rate = (
        ivf_sample_rate if ivf_sample_rate is not None else sample_rate
    )
    effective_pq_sample_rate = (
        pq_sample_rate if pq_sample_rate is not None else sample_rate
    )

    builder = _indices_builder_for_field_path(dataset_obj, column, field)
    dimension = builder.dimension
    dataset_version = getattr(dataset_obj, "version", None)

    if ivf_centroids is not None:
        ivf_centroids = _coerce_ivf_centroids(ivf_centroids, dimension)
    if pq_codebook is not None:
        pq_codebook = _validate_pq_codebook(pq_codebook, dimension)

    checkpoint: Optional[_VectorIndexBuildCheckpoint] = None
    if checkpoint_dir is not None:
        fingerprint = _VectorIndexBuildCheckpoint.build_fingerprint(
            dataset_uri=dataset_uri,
            dataset_version=dataset_version,
            column=column,
            index_type=index_type_name,
            name=name,
            metric=metric_lower,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            ivf_sample_rate=effective_ivf_sample_rate,
            pq_sample_rate=effective_pq_sample_rate,
            max_iters=max_iters,
            extra_kwargs=repr(sorted((k, repr(v)) for k, v in kwargs.items())),
            ivf_centroids_digest=_artifact_digest(ivf_centroids),
            pq_codebook_digest=_artifact_digest(pq_codebook),
            rabitq_model_digest=_artifact_digest(rabitq_model),
        )
        checkpoint = _VectorIndexBuildCheckpoint(checkpoint_dir, fingerprint)

    # Global training artifacts are resolved once so that all shards share
    # the same centroids/codebook/rotation; the Ray entrypoint owns the
    # lifecycle of these artifacts and distributes them to workers.
    phase1_state = checkpoint.load_phase1() if checkpoint is not None else None
    if phase1_state is not None:
        logger.info(
            "Phase 1: Reusing IVF/PQ training artifacts from checkpoint at %s",
            checkpoint_dir,
        )
    else:
        training_kwargs: dict[str, Any] = {
            "column": column,
            "index_type_name": index_type_name,
            "metric": metric_lower,
            "num_partitions": num_partitions,
            "num_sub_vectors": num_sub_vectors,
            "ivf_centroids": ivf_centroids,
            "pq_codebook": pq_codebook,
            "ivf_sample_rate": effective_ivf_sample_rate,
            "pq_sample_rate": effective_pq_sample_rate,
            "max_iters": max_iters,
            "accelerator": accelerator,
            "build_kwargs": index_build_kwargs,
        }
        if training_ray_remote_args is None:
            phase1_state = _train_vector_index_models(dataset_obj, **training_kwargs)
        else:
            logger.info(
                "Phase 1: Training on a Ray worker with options %s",
                training_ray_remote_args,
            )
            train_task = ray.remote(_train_vector_index_models).options(
                **training_ray_remote_args
            )
            phase1_state = ray.get(train_task.remote(dataset_obj, **training_kwargs))
        if checkpoint is not None:
            checkpoint.save_phase1(phase1_state)

    ivf_centroids_artifact = phase1_state["ivf_centroids"]
    pq_codebook_artifact = phase1_state["pq_codebook"]
    num_partitions = phase1_state["num_partitions"]
    num_sub_vectors = phase1_state["num_sub_vectors"]
    if phase1_state["rabitq_model"] is not None:
        index_build_kwargs["rabitq_model"] = phase1_state["rabitq_model"]

    fragment_batches, fragment_sizes = _plan_fragment_batches(
        fragments,
        num_workers=num_workers,
        rows_per_task=rows_per_task,
        logger=logger,
    )

    # Broadcast the driver's manifest so workers skip re-reading it from
    # object storage and build against the exact version the batches were
    # planned on. Namespace-managed tables keep the standard load path.
    serialized_manifest: Optional[bytes] = None
    if not namespace_kwargs:
        manifest_fn = getattr(
            getattr(dataset_obj, "_ds", None), "serialized_manifest", None
        )
        if manifest_fn is not None:
            try:
                serialized_manifest = manifest_fn()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Could not serialize dataset manifest for workers: %s", exc
                )

    cached_results: list[dict[str, Any]] = []
    batches_to_build: list[list[int]] = []
    for batch in fragment_batches:
        segment = checkpoint.load_segment(batch) if checkpoint is not None else None
        if segment is not None and not _segment_files_exist(dataset_uri, segment):
            logger.warning(
                "Checkpointed segment %s for fragments %s no longer exists "
                "in storage; rebuilding the batch.",
                getattr(segment, "uuid", "<unknown>"),
                batch,
            )
            segment = None
        if segment is not None:
            cached_results.append(
                {
                    "status": "success",
                    "fragment_ids": list(batch),
                    "segment_index": segment,
                    "uuid": getattr(segment, "uuid", None),
                }
            )
        else:
            batches_to_build.append(batch)

    if cached_results:
        logger.info(
            "Phase 2: Reusing %d of %d segments from checkpoint at %s",
            len(cached_results),
            len(fragment_batches),
            checkpoint_dir,
        )

    logger.info(
        "Phase 2: Distributing vector index build across %d workers for "
        "%d fragment batches (%d fragments)",
        num_workers,
        len(batches_to_build),
        len(fragment_ids_to_use),
    )

    def create_fragment_handler() -> Any:
        shared_ivf_centroids, shared_pq_codebook = (
            _put_vector_index_artifacts_in_object_store(
                ivf_centroids_artifact,
                pq_codebook_artifact,
            )
        )
        shared_manifest = _ray_put_index_artifact(serialized_manifest)
        return _handle_vector_fragment_index(
            dataset_uri=dataset_uri,
            column=column,
            index_type=index_type_name,
            name=name,
            index_uuid=index_id,
            replace=replace,
            metric=metric_lower,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            ivf_centroids=shared_ivf_centroids,
            pq_codebook=shared_pq_codebook,
            storage_options=merged_storage_options,
            block_size=block_size,
            namespace_impl=namespace_impl,
            namespace_properties=namespace_properties,
            table_id=table_id,
            dataset_version=dataset_version,
            serialized_manifest=shared_manifest,
            dimension=dimension,
            **index_build_kwargs,
        )

    on_segment_built: Optional[Callable[[list[int], dict[str, Any]], None]] = None
    if checkpoint is not None:
        segment_checkpoint = checkpoint

        def _save_segment_on_success(batch: list[int], result: dict[str, Any]) -> None:
            segment_checkpoint.save_segment(batch, result["segment_index"])

        on_segment_built = _save_segment_on_success

    if batches_to_build:
        built_results = _map_async_with_pool(
            create_fragment_handler=create_fragment_handler,
            fragment_batches=batches_to_build,
            num_workers=min(num_workers, len(batches_to_build)),
            ray_remote_args=ray_remote_args,
            error_prefix="Failed to complete distributed vector index building",
            max_task_retries=max_task_retries,
            on_success=on_segment_built,
        )
    else:
        built_results = []

    results = cached_results + built_results

    failed_results = [r for r in results if r.get("status") == "error"]
    if failed_results:
        error_messages = [r["error"] for r in failed_results if "error" in r]
        completed_uuids = [
            str(r.get("uuid") or getattr(r.get("segment_index"), "uuid", None))
            for r in results
            if r.get("status") == "success"
        ]
        if checkpoint is not None:
            logger.error(
                "Vector index build failed after %d retries; %d completed "
                "segments are checkpointed at %s — re-run the same "
                "create_index call to resume.",
                max_task_retries,
                len(completed_uuids),
                checkpoint_dir,
            )
        elif completed_uuids:
            logger.error(
                "Vector index build failed; %d already-built uncommitted "
                "segments are orphaned under _indices/ (uuids: %s) and will "
                "be reclaimed by cleanup_old_versions. Pass checkpoint_dir "
                "to make failed builds resumable.",
                len(completed_uuids),
                ", ".join(completed_uuids),
            )
        raise RuntimeError("Vector index building failed: " + "; ".join(error_messages))

    successful_results = [r for r in results if r.get("status") == "success"]
    if not successful_results:
        raise RuntimeError("No successful vector index creation results found")

    segments = [r["segment_index"] for r in successful_results]

    if target_num_segments is not None and len(segments) > target_num_segments:
        merge_groups = _plan_segment_merge_groups(
            successful_results, fragment_sizes, target_num_segments
        )
        passthrough_segments = [group[0] for group in merge_groups if len(group) == 1]
        multi_groups = [group for group in merge_groups if len(group) > 1]

        logger.info(
            "Phase 2.5: Merging %d segments into %d segments on workers",
            len(segments),
            len(merge_groups),
        )

        merged_segments: list[Any] = []
        groups_to_merge: list[list[Any]] = []
        for group in multi_groups:
            member_keys = [getattr(segment, "uuid", None) for segment in group]
            cached_merge = (
                checkpoint.load_merge(member_keys) if checkpoint is not None else None
            )
            if cached_merge is not None and not _segment_files_exist(
                dataset_uri, cached_merge
            ):
                logger.warning(
                    "Checkpointed merged segment %s no longer exists in "
                    "storage; re-merging the group.",
                    getattr(cached_merge, "uuid", "<unknown>"),
                )
                cached_merge = None
            if cached_merge is not None:
                merged_segments.append(cached_merge)
            else:
                groups_to_merge.append(group)

        if groups_to_merge:

            def create_merge_handler() -> Any:
                shared_manifest = _ray_put_index_artifact(serialized_manifest)
                return _handle_merge_segments(
                    dataset_uri=dataset_uri,
                    storage_options=merged_storage_options,
                    block_size=block_size,
                    namespace_impl=namespace_impl,
                    namespace_properties=namespace_properties,
                    table_id=table_id,
                    dataset_version=dataset_version,
                    serialized_manifest=shared_manifest,
                )

            on_group_merged: Optional[Callable[[list[Any], dict[str, Any]], None]] = (
                None
            )
            if checkpoint is not None:
                merge_checkpoint = checkpoint

                def _save_merge_on_success(
                    group: list[Any], result: dict[str, Any]
                ) -> None:
                    merge_checkpoint.save_merge(
                        [getattr(segment, "uuid", None) for segment in group],
                        result["segment_index"],
                    )

                on_group_merged = _save_merge_on_success

            merge_results = _map_async_with_pool(
                create_fragment_handler=create_merge_handler,
                fragment_batches=groups_to_merge,
                num_workers=min(num_workers, len(groups_to_merge)),
                ray_remote_args=ray_remote_args,
                error_prefix="Failed to merge vector index segments",
                max_task_retries=max_task_retries,
                on_success=on_group_merged,
            )
            merge_errors = [
                r["error"] for r in merge_results if r.get("status") == "error"
            ]
            if merge_errors:
                raise RuntimeError(
                    "Vector index segment merge failed: " + "; ".join(merge_errors)
                )
            merged_segments.extend(r["segment_index"] for r in merge_results)

        segments = passthrough_segments + merged_segments

    logger.info(
        "Phase 3: Committing %d index segments for vector index '%s'",
        len(segments),
        name,
    )

    def load_commit_dataset() -> Any:
        return LanceDataset(
            dataset_uri,
            **_dataset_load_kwargs(
                merged_storage_options, namespace_kwargs, block_size
            ),
        )

    covered_fragment_ids: set[int] = set()
    for result in successful_results:
        covered_fragment_ids.update(result.get("fragment_ids", []))

    updated_dataset = _commit_index_segments_with_retry(
        load_commit_dataset,
        index_name=name,
        column=column,
        segments=segments,
        covered_fragment_ids=covered_fragment_ids,
        built_version=dataset_version,
    )

    if checkpoint is not None:
        checkpoint.clear()

    logger.info(
        "Successfully created distributed vector index '%s'",
        name,
    )
    logger.info(
        "Index ID: %s, Fragments: %d, Batches: %d, Segments committed: %d",
        index_id,
        len(fragment_ids_to_use),
        len(fragment_batches),
        len(segments),
    )

    return updated_dataset


def optimize_indices(
    uri: Optional[str] = None,
    *,
    table_id: Optional[list[str]] = None,
    indices: Optional[list[str]] = None,
    num_indices_to_merge: int = 1,
    retrain: bool = False,
    storage_options: Optional[dict[str, str]] = None,
    namespace_impl: Optional[str] = None,
    namespace_properties: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> "lance.LanceDataset":
    """Optimize indices for newly added data (incremental index update).

    As new data arrives it is not added to existing indexes automatically.
    This function adds the new data to existing indexes, restoring search
    performance. It does not retrain the index by default; it only assigns
    the new data to existing partitions, so the update is quicker than
    retraining but may have less accuracy if the new data has different
    patterns.

    Delegates to ``dataset.optimize.optimize_indices()`` from the lance library.

    Args:
        uri: The URI of the Lance dataset. Either uri OR
            (namespace_impl + table_id) must be provided.
        table_id: The table identifier as a list of strings. Must be provided
            together with namespace_impl.
        indices: Optional list of index names to optimize. If None, all indices
            on the dataset are optimized. Passed to lance as ``index_names``.
            When the dataset has both scalar and vector columns, specifying
            only the index names you need (e.g. ``["label_btree"]``) can avoid
            internal errors on list/vector fields in some lance versions.
            If you still get an error about ``vector.item`` / ``List(Float64)``,
            this is a known lance bug: use a dataset with only scalar columns
            for scalar index optimization until lance fixes it.
        num_indices_to_merge: Number of delta indices to merge (default 1).
            If set to 0, a new delta index will be created instead of merging.
        retrain: If True, retrain the whole index from current data; all indices
            are merged into one and ``num_indices_to_merge`` is ignored.
            Use when data distribution has changed significantly (default False).
        storage_options: Storage options for the dataset.
        namespace_impl: The namespace implementation type (e.g., "rest", "dir").
            Used together with table_id for resolving the dataset location.
        namespace_properties: Properties for connecting to the namespace.
        **kwargs: Additional arguments passed through to the underlying
            ``DatasetOptimizer.optimize_indices`` API.

    Returns:
        The Lance dataset instance (optimization is applied in-place on storage).

    Raises:
        ValueError: If input parameters are invalid.
        RuntimeError: If optimize_indices is not supported by the current
            lance version or if the operation fails.

    Example:
        >>> import lance_ray
        >>> ds = lance_ray.optimize_indices("path/to/dataset")
        >>> ds = lance_ray.optimize_indices(
        ...     "path/to/dataset",
        ...     indices=["vec_idx", "scalar_idx"],
        ...     num_indices_to_merge=2,
        ... )
    """
    logger.info(
        "Starting optimize_indices: uri=%s, indices=%s, num_indices_to_merge=%s, retrain=%s",
        uri if uri else "(from namespace)",
        indices,
        num_indices_to_merge,
        retrain,
    )
    validate_uri_or_namespace(uri, namespace_impl, table_id)

    resolve_from_namespace = uri is None and has_namespace_params(
        namespace_impl, table_id
    )
    uri, merged_storage_options = resolve_namespace_table(
        uri, storage_options, namespace_impl, namespace_properties, table_id
    )
    if resolve_from_namespace:
        logger.info(
            "Resolved dataset URI from namespace (table_id=%s): %s",
            table_id,
            uri,
        )

    namespace_kwargs = get_namespace_kwargs(
        namespace_impl, namespace_properties, table_id
    )

    dataset = LanceDataset(
        uri,
        storage_options=merged_storage_options,
        **namespace_kwargs,
    )
    logger.info(
        "Loaded dataset: uri=%s, version=%s",
        uri,
        getattr(dataset, "version", "unknown"),
    )

    if not hasattr(dataset, "optimize"):
        raise RuntimeError(
            "LanceDataset has no 'optimize' property. Please ensure "
            "lance is installed with a version that provides DatasetOptimizer."
        )
    optimizer = dataset.optimize
    if not hasattr(optimizer, "optimize_indices"):
        raise RuntimeError(
            "optimize_indices is not available on DatasetOptimizer. Please ensure "
            "lance is installed with a version that provides optimize_indices."
        )

    call_kw: dict[str, Any] = {
        "num_indices_to_merge": num_indices_to_merge,
        "retrain": retrain,
        **kwargs,
    }
    if indices is not None:
        call_kw["index_names"] = indices

    logger.info(
        "Calling DatasetOptimizer.optimize_indices with: %s",
        {k: v for k, v in call_kw.items() if k != "storage_options"},
    )
    optimizer.optimize_indices(**call_kw)
    logger.info(
        "optimize_indices completed successfully for dataset uri=%s",
        uri,
    )
    return dataset
