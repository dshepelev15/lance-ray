"""Tests for distributed vector index option handling."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_index_module_with_stubs():
    """Load lance_ray.index when the native pylance extension is unavailable."""

    repo_root = Path(__file__).resolve().parents[1]
    package = ModuleType("lance_ray")
    package.__path__ = [str(repo_root / "lance_ray")]

    lance = ModuleType("lance")
    lance.__version__ = "6.0.0"

    lance_dataset = ModuleType("lance.dataset")
    lance_dataset.Index = type("Index", (), {})
    lance_dataset.IndexConfig = type("IndexConfig", (), {})
    lance_dataset.LanceDataset = object

    lance_indices = ModuleType("lance.indices")
    lance_indices.IndicesBuilder = object

    ray = ModuleType("ray")
    ray.ObjectRef = type("ObjectRef", (), {})
    ray_util = ModuleType("ray.util")
    ray_multiprocessing = ModuleType("ray.util.multiprocessing")
    ray_multiprocessing.Pool = object

    sys.modules["lance_ray"] = package
    sys.modules["lance"] = lance
    sys.modules["lance.dataset"] = lance_dataset
    sys.modules["lance.indices"] = lance_indices
    sys.modules["ray"] = ray
    sys.modules["ray.util"] = ray_util
    sys.modules["ray.util.multiprocessing"] = ray_multiprocessing

    spec = importlib.util.spec_from_file_location(
        "lance_ray.index",
        repo_root / "lance_ray" / "index.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["lance_ray.index"] = module
    spec.loader.exec_module(module)
    return module


try:
    from lance_ray import index as index_mod
except ImportError:  # pragma: no cover - environment dependent
    index_mod = _load_index_module_with_stubs()


class _FakeField:
    def __init__(self, name, field_type=None):
        self.name = name
        self.type = field_type or index_mod.pa.float32()


class _FakeLanceField:
    def id(self):
        return 7


class _FakeLanceSchema:
    def field(self, column):
        if column not in {"value", "text"}:
            raise KeyError(column)
        return _FakeLanceField()


class _FakeSchema:
    def field(self, column):
        if column == "vector":
            return _FakeField(column)
        if column == "value":
            return _FakeField(column, index_mod.pa.int64())
        if column == "text":
            return _FakeField(column, index_mod.pa.string())
        else:
            raise KeyError(column)

    def __iter__(self):
        return iter(
            [
                _FakeField("vector"),
                _FakeField("value", index_mod.pa.int64()),
                _FakeField("text", index_mod.pa.string()),
            ]
        )


class _FakeFragment:
    def __init__(self, fragment_id, rows):
        self.fragment_id = fragment_id
        self._rows = rows

    def count_rows(self):
        return self._rows


class _FakeDataset:
    uri = "memory://fake"
    schema = _FakeSchema()
    lance_schema = _FakeLanceSchema()
    version = 1

    def get_fragments(self):
        return [_FakeFragment(0, 100), _FakeFragment(1, 100)]

    def count_rows(self):
        return 200

    def describe_indices(self):
        return []

    def create_scalar_index(self, **kwargs):
        self.scalar_index_kwargs = kwargs

    def create_index_uncommitted(self, **kwargs):
        self.vector_index_kwargs = kwargs
        return "segment"

    def commit_existing_index_segments(self, **kwargs):
        self.commit_kwargs = kwargs
        return self


def test_map_async_with_pool_closes_and_joins_pool(monkeypatch):
    """The Ray Pool should be joined after close so actors finish cleanup."""

    events = []

    class FakeAsyncResult:
        def get(self):
            events.append("get")
            return [{"status": "success"}]

    class FakePool:
        def __init__(self, processes, ray_remote_args):
            events.append(("init", processes, ray_remote_args))

        def map_async(self, fragment_handler, fragment_batches, chunksize):
            events.append(("map_async", fragment_batches, chunksize))
            return FakeAsyncResult()

        def close(self):
            events.append("close")

        def join(self):
            events.append("join")

    def create_fragment_handler():
        events.append("create_handler")
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    monkeypatch.setattr(index_mod, "Pool", FakePool)

    assert index_mod._map_async_with_pool(
        create_fragment_handler=create_fragment_handler,
        fragment_batches=[[0, 1]],
        num_workers=2,
        ray_remote_args={"num_cpus": 1},
        error_prefix="failed",
    ) == [{"status": "success"}]
    assert events == [
        ("init", 2, {"num_cpus": 1}),
        "create_handler",
        ("map_async", [[0, 1]], 1),
        "get",
        "close",
        "join",
    ]


def test_create_index_uses_sample_rate_for_global_training(monkeypatch):
    """The public sample_rate option should drive both IVF and PQ training."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            captured["builder_dataset"] = dataset
            captured["builder_column"] = column

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq_ivf_model"] = ivf_model
            captured["train_pq"] = kwargs
            return SimpleNamespace(codebook="pq_codebook", num_subvectors=4)

    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        captured["put_artifacts"] = (ivf_centroids, pq_codebook)
        return "ivf_ref", "pq_ref"

    def fake_map_async_with_pool(**kwargs):
        captured["map_kwargs"] = kwargs
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": [0, 1],
                "segment_index": "segment",
            }
        ]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    updated_dataset = index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        num_workers=2,
        num_partitions=4,
        num_sub_vectors=4,
        sample_rate=8,
    )

    assert updated_dataset is fake_dataset
    assert captured["train_ivf"]["sample_rate"] == 8
    assert captured["train_pq"]["sample_rate"] == 8
    assert captured["put_artifacts"] == ("ivf_centroids", "pq_codebook")
    assert captured["fragment_handler_kwargs"]["ivf_centroids"] == "ivf_ref"
    assert captured["fragment_handler_kwargs"]["pq_codebook"] == "pq_ref"
    assert "sample_rate" not in captured["fragment_handler_kwargs"]
    assert fake_dataset.commit_kwargs["segments"] == ["segment"]


def test_create_index_supports_ivf_rq(monkeypatch):
    """IVF_RQ should build and share a RaBitQ model when one is not provided."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            captured["builder_dataset"] = dataset
            captured["builder_column"] = column

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq"] = kwargs
            raise AssertionError("IVF_RQ should not train a PQ codebook")

    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {
            "status": "success",
            "fragment_ids": fragment_ids,
            "segment_index": "segment",
        }

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        captured["put_artifacts"] = (ivf_centroids, pq_codebook)
        return "ivf_ref", None

    def fake_build_rabitq_model(*, dimension, num_bits):
        captured["build_rabitq_model"] = {
            "dimension": dimension,
            "num_bits": num_bits,
        }
        return "auto-rq-model"

    def fake_map_async_with_pool(**kwargs):
        captured["map_kwargs"] = kwargs
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": [0, 1],
                "segment_index": "segment",
            }
        ]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_build_rabitq_model", fake_build_rabitq_model)
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    updated_dataset = index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_RQ",
        name="vector_idx",
        num_workers=2,
        num_partitions=4,
        sample_rate=8,
        num_bits=2,
    )

    assert updated_dataset is fake_dataset
    assert captured["train_ivf"]["sample_rate"] == 8
    assert "train_pq" not in captured
    assert captured["build_rabitq_model"] == {"dimension": 16, "num_bits": 2}
    assert captured["put_artifacts"] == ("ivf_centroids", None)
    assert captured["fragment_handler_kwargs"]["index_type"] == "IVF_RQ"
    assert captured["fragment_handler_kwargs"]["ivf_centroids"] == "ivf_ref"
    assert captured["fragment_handler_kwargs"]["pq_codebook"] is None
    assert captured["fragment_handler_kwargs"]["num_bits"] == 2
    assert captured["fragment_handler_kwargs"]["rabitq_model"] == "auto-rq-model"
    assert fake_dataset.commit_kwargs["segments"] == ["segment"]


def test_create_index_uses_provided_ivf_rq_model(monkeypatch):
    """A caller-provided RaBitQ model should be shared without rebuilding it."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            captured["builder_dataset"] = dataset
            captured["builder_column"] = column

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            raise AssertionError("IVF_RQ should not train a PQ codebook")

    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {
            "status": "success",
            "fragment_ids": fragment_ids,
            "segment_index": "segment",
        }

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        captured["put_artifacts"] = (ivf_centroids, pq_codebook)
        return "ivf_ref", None

    def fake_build_rabitq_model(**kwargs):
        raise AssertionError("provided rabitq_model should be reused")

    def fake_map_async_with_pool(**kwargs):
        captured["map_kwargs"] = kwargs
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": [0, 1],
                "segment_index": "segment",
            }
        ]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_build_rabitq_model", fake_build_rabitq_model)
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    updated_dataset = index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_RQ",
        name="vector_idx",
        num_workers=2,
        num_partitions=4,
        sample_rate=8,
        rabitq_model="shared-rq-model",
    )

    assert updated_dataset is fake_dataset
    assert captured["fragment_handler_kwargs"]["rabitq_model"] == "shared-rq-model"
    assert captured["fragment_handler_kwargs"]["index_type"] == "IVF_RQ"
    assert fake_dataset.commit_kwargs["segments"] == ["segment"]


def test_create_index_rejects_non_positive_sample_rate(monkeypatch):
    """Invalid sample rates should fail before training starts."""

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)

    with pytest.raises(ValueError, match="sample_rate must be positive, got 0"):
        index_mod.create_index(
            uri=_FakeDataset(),
            column="vector",
            index_type="IVF_PQ",
            sample_rate=0,
        )


@pytest.mark.parametrize("index_type", ["BTREE", "BITMAP", "INVERTED", "FTS"])
def test_create_scalar_index_uses_segment_path(monkeypatch, index_type):
    """Migrated scalar indexes should use Lance's segment workflow."""

    captured = {"loads": []}
    fake_dataset = _FakeDataset()

    def fake_lance_dataset(*args, **kwargs):
        captured["loads"].append(kwargs)
        return fake_dataset

    def fake_handle_scalar_segment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {
            "status": "success",
            "fragment_ids": fragment_ids,
            "segment_index": "segment",
        }

    def fake_map_async_with_pool(**kwargs):
        captured["map_kwargs"] = kwargs
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": [0, 1],
                "segment_index": "segment",
            }
        ]

    monkeypatch.setattr(index_mod, "LanceDataset", fake_lance_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_scalar_segment_index",
        fake_handle_scalar_segment_index,
    )
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    column = "text" if index_type in {"INVERTED", "FTS"} else "value"
    updated_dataset = index_mod.create_scalar_index(
        uri="memory://fake",
        column=column,
        index_type=index_type,
        num_workers=2,
        block_size=4096,
    )

    assert updated_dataset is fake_dataset
    assert [load["block_size"] for load in captured["loads"]] == [4096, 4096]
    assert captured["fragment_handler_kwargs"]["index_type"] == index_type
    assert captured["fragment_handler_kwargs"]["block_size"] == 4096
    assert fake_dataset.commit_kwargs["segments"] == ["segment"]


def test_create_index_passes_block_size_to_loads_and_handler(monkeypatch):
    """The vector index path should use block_size for driver and worker loads."""

    captured = {"loads": []}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            captured["builder_dataset"] = dataset
            captured["builder_column"] = column

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq_ivf_model"] = ivf_model
            captured["train_pq"] = kwargs
            return SimpleNamespace(codebook="pq_codebook", num_subvectors=4)

    def fake_lance_dataset(*args, **kwargs):
        captured["loads"].append(kwargs)
        return fake_dataset

    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        captured["put_artifacts"] = (ivf_centroids, pq_codebook)
        return "ivf_ref", "pq_ref"

    def fake_map_async_with_pool(**kwargs):
        captured["map_kwargs"] = kwargs
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": [0, 1],
                "segment_index": "segment",
            }
        ]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", fake_lance_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    updated_dataset = index_mod.create_index(
        uri="memory://fake",
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        num_workers=2,
        num_partitions=4,
        num_sub_vectors=4,
        block_size=8192,
    )

    assert updated_dataset is fake_dataset
    assert [load["block_size"] for load in captured["loads"]] == [8192, 8192]
    assert captured["fragment_handler_kwargs"]["block_size"] == 8192
    assert captured["put_artifacts"] == ("ivf_centroids", "pq_codebook")
    assert fake_dataset.commit_kwargs["segments"] == ["segment"]


def test_fragment_handlers_pass_block_size_to_dataset_load(monkeypatch):
    """Worker-side scalar and vector handlers should load datasets with block_size."""

    captured = {"loads": []}
    fake_dataset = _FakeDataset()

    def fake_lance_dataset(*args, **kwargs):
        captured["loads"].append(kwargs)
        return fake_dataset

    monkeypatch.setattr(index_mod, "LanceDataset", fake_lance_dataset)

    scalar_handler = index_mod._handle_fragment_index(
        dataset_uri="memory://fake",
        column="value",
        index_type="LABEL_LIST",
        name="value_idx",
        index_uuid="scalar-index",
        replace=False,
        train=True,
        block_size=4096,
    )
    vector_handler = index_mod._handle_vector_fragment_index(
        dataset_uri="memory://fake",
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        index_uuid="vector-index",
        replace=False,
        metric="l2",
        num_partitions=4,
        num_sub_vectors=4,
        ivf_centroids="ivf_centroids",
        pq_codebook="pq_codebook",
        block_size=8192,
    )

    assert scalar_handler([0])["status"] == "success"
    assert vector_handler([0])["status"] == "success"
    assert [load["block_size"] for load in captured["loads"]] == [4096, 8192]


def test_vector_fragment_handler_resolves_shared_artifact_refs(monkeypatch):
    """Workers should dereference shared training artifacts before Lance calls."""

    class FakeObjectRef:
        def __init__(self, value):
            self.value = value

    fake_dataset = _FakeDataset()
    captured = {"gets": []}

    def fake_get(ref):
        captured["gets"].append(ref)
        return ref.value

    monkeypatch.setattr(index_mod.ray, "ObjectRef", FakeObjectRef, raising=False)
    monkeypatch.setattr(index_mod.ray, "get", fake_get, raising=False)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)

    ivf_ref = FakeObjectRef("ivf_centroids")
    pq_ref = FakeObjectRef("pq_codebook")
    vector_handler = index_mod._handle_vector_fragment_index(
        dataset_uri="memory://fake",
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        index_uuid="vector-index",
        replace=False,
        metric="l2",
        num_partitions=4,
        num_sub_vectors=4,
        ivf_centroids=ivf_ref,
        pq_codebook=pq_ref,
    )

    assert vector_handler([0])["status"] == "success"
    assert captured["gets"] == [ivf_ref, pq_ref]
    assert fake_dataset.vector_index_kwargs["ivf_centroids"] == "ivf_centroids"
    assert fake_dataset.vector_index_kwargs["pq_codebook"] == "pq_codebook"


# ---------------------------------------------------------------------------
# Scale-oriented behavior: precomputed artifacts, training knobs, batching,
# retries, checkpointing, segment merging, commit retry, manifest broadcast.
# ---------------------------------------------------------------------------


class _FourFragmentDataset(_FakeDataset):
    def get_fragments(self):
        return [_FakeFragment(i, 100) for i in range(4)]

    def count_rows(self):
        return 400


def _make_centroids(num_partitions, dimension):
    values = index_mod.pa.array(
        [float(i) for i in range(num_partitions * dimension)],
        type=index_mod.pa.float32(),
    )
    return index_mod.pa.FixedSizeListArray.from_arrays(values, dimension)


def _patch_common(monkeypatch, fake_dataset, captured, builder_cls):
    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        captured["put_artifacts"] = (ivf_centroids, pq_codebook)
        return "ivf_ref", "pq_ref"

    def fake_map_async_with_pool(**kwargs):
        captured.setdefault("map_calls", []).append(kwargs)
        kwargs["create_fragment_handler"]()
        return [
            {
                "status": "success",
                "fragment_ids": list(batch),
                "segment_index": "segment",
            }
            for batch in kwargs["fragment_batches"]
        ]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", builder_cls)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)


def test_create_index_honors_precomputed_ivf_centroids(monkeypatch):
    """Precomputed centroids should skip IVF training entirely."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            raise AssertionError("IVF training should be skipped")

        def train_pq(self, ivf_model, **kwargs):
            raise AssertionError("PQ training should not run for IVF_FLAT")

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    centroids = _make_centroids(4, 16)
    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_FLAT",
        name="vector_idx",
        num_workers=2,
        ivf_centroids=centroids,
    )

    assert captured["fragment_handler_kwargs"]["num_partitions"] == 4
    assert captured["put_artifacts"][0] is centroids
    assert fake_dataset.commit_kwargs["segments"] == ["segment", "segment"]


def test_create_index_honors_precomputed_pq_artifacts(monkeypatch):
    """Precomputed centroids + codebook should skip both training phases."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            raise AssertionError("IVF training should be skipped")

        def train_pq(self, ivf_model, **kwargs):
            raise AssertionError("PQ training should be skipped")

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    centroids = _make_centroids(4, 16)
    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        num_workers=2,
        ivf_centroids=centroids,
        pq_codebook="codebook",
        num_sub_vectors=4,
    )

    assert captured["fragment_handler_kwargs"]["num_partitions"] == 4
    assert captured["fragment_handler_kwargs"]["num_sub_vectors"] == 4
    assert captured["put_artifacts"] == (centroids, "codebook")


def test_create_index_requires_num_sub_vectors_with_precomputed_codebook(monkeypatch):
    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            raise AssertionError("IVF training should be skipped")

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    with pytest.raises(ValueError, match="num_sub_vectors must be provided"):
        index_mod.create_index(
            uri=fake_dataset,
            column="vector",
            index_type="IVF_PQ",
            num_workers=2,
            ivf_centroids=_make_centroids(4, 16),
            pq_codebook="codebook",
        )


def test_create_index_trains_pq_against_precomputed_centroids(monkeypatch):
    """PQ training must reuse the provided centroids via an IvfModel wrapper."""

    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            raise AssertionError("IVF training should be skipped")

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq_ivf_model"] = ivf_model
            captured["train_pq"] = kwargs
            return SimpleNamespace(codebook="pq_codebook", num_subvectors=4)

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    centroids = _make_centroids(4, 16)
    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_PQ",
        num_workers=2,
        ivf_centroids=centroids,
        metric="cosine",
    )

    assert captured["train_pq_ivf_model"].centroids is centroids
    assert captured["train_pq_ivf_model"].distance_type == "cosine"
    assert captured["fragment_handler_kwargs"]["num_sub_vectors"] == 4


def test_create_index_forwards_accelerator_and_max_iters(monkeypatch):
    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq"] = kwargs
            return SimpleNamespace(codebook="pq_codebook", num_subvectors=4)

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_PQ",
        num_workers=2,
        num_partitions=4,
        accelerator="cuda",
        max_iters=7,
    )

    assert captured["train_ivf"]["accelerator"] == "cuda"
    assert captured["train_ivf"]["max_iters"] == 7
    assert captured["train_pq"]["max_iters"] == 7
    assert "accelerator" not in captured["fragment_handler_kwargs"]


def test_create_index_uses_separate_ivf_and_pq_sample_rates(monkeypatch):
    captured = {}
    fake_dataset = _FakeDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            captured["train_ivf"] = kwargs
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

        def train_pq(self, ivf_model, **kwargs):
            captured["train_pq"] = kwargs
            return SimpleNamespace(codebook="pq_codebook", num_subvectors=4)

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)

    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_PQ",
        num_workers=2,
        num_partitions=4,
        sample_rate=256,
        ivf_sample_rate=4,
        pq_sample_rate=16,
    )

    assert captured["train_ivf"]["sample_rate"] == 4
    assert captured["train_pq"]["sample_rate"] == 16


def test_plan_fragment_batches_respects_rows_per_task():
    fragments = [_FakeFragment(i, 100) for i in range(10)]
    batches, sizes = index_mod._plan_fragment_batches(
        fragments,
        num_workers=2,
        rows_per_task=100,
        logger=index_mod.logger,
    )
    assert len(batches) == 10
    assert sorted(fid for batch in batches for fid in batch) == list(range(10))
    assert sizes == {i: 100 for i in range(10)}


def test_plan_fragment_batches_defaults_to_worker_count():
    fragments = [_FakeFragment(i, 100) for i in range(10)]
    batches, _ = index_mod._plan_fragment_batches(
        fragments,
        num_workers=2,
        rows_per_task=None,
        logger=index_mod.logger,
    )
    assert len(batches) == 2
    assert {len(batch) for batch in batches} == {5}


def test_plan_fragment_batches_enforces_segment_row_cap(monkeypatch):
    monkeypatch.setattr(index_mod, "_MAX_ROWS_PER_SEGMENT", 150)

    fragments = [_FakeFragment(i, 100) for i in range(3)]
    batches, _ = index_mod._plan_fragment_batches(
        fragments,
        num_workers=1,
        rows_per_task=None,
        logger=index_mod.logger,
    )
    # No two 100-row fragments fit under the 150-row cap together.
    assert len(batches) == 3

    with pytest.raises(ValueError, match="exceeds the maximum"):
        index_mod._plan_fragment_batches(
            [_FakeFragment(0, 200)],
            num_workers=1,
            rows_per_task=None,
            logger=index_mod.logger,
        )


def test_map_async_with_pool_retries_failed_tasks(monkeypatch):
    attempts = {}
    successes = []

    class FakePool:
        def __init__(self, processes, ray_remote_args):
            pass

        def imap(self, handler, payloads, chunksize):
            return (handler(payload) for payload in payloads)

        def close(self):
            pass

        def join(self):
            pass

    def handler(fragment_ids):
        key = tuple(fragment_ids)
        attempts[key] = attempts.get(key, 0) + 1
        if key == (1,) and attempts[key] == 1:
            return {"status": "error", "fragment_ids": fragment_ids, "error": "boom"}
        return {"status": "success", "fragment_ids": fragment_ids}

    monkeypatch.setattr(index_mod, "Pool", FakePool)

    results = index_mod._map_async_with_pool(
        create_fragment_handler=lambda: handler,
        fragment_batches=[[0], [1], [2]],
        num_workers=2,
        ray_remote_args=None,
        error_prefix="failed",
        max_task_retries=1,
        on_success=lambda batch, result: successes.append(tuple(batch)),
    )

    assert [r["status"] for r in results] == ["success"] * 3
    assert attempts == {(0,): 1, (1,): 2, (2,): 1}
    assert sorted(successes) == [(0,), (1,), (2,)]


def test_map_async_with_pool_reports_exhausted_retries(monkeypatch):
    class FakePool:
        def __init__(self, processes, ray_remote_args):
            pass

        def imap(self, handler, payloads, chunksize):
            return (handler(payload) for payload in payloads)

        def close(self):
            pass

        def join(self):
            pass

    def handler(fragment_ids):
        return {"status": "error", "fragment_ids": fragment_ids, "error": "boom"}

    monkeypatch.setattr(index_mod, "Pool", FakePool)

    results = index_mod._map_async_with_pool(
        create_fragment_handler=lambda: handler,
        fragment_batches=[[0]],
        num_workers=1,
        ray_remote_args=None,
        error_prefix="failed",
        max_task_retries=2,
    )

    assert results == [{"status": "error", "fragment_ids": [0], "error": "boom"}]


def test_vector_index_build_checkpoint_roundtrip(tmp_path):
    checkpoint = index_mod._VectorIndexBuildCheckpoint(str(tmp_path), "abc123")

    assert checkpoint.load_phase1() is None
    checkpoint.save_phase1({"num_partitions": 4})
    assert checkpoint.load_phase1() == {"num_partitions": 4}

    assert checkpoint.load_segment([1, 2]) is None
    checkpoint.save_segment([1, 2], "segment-a")
    assert checkpoint.load_segment([2, 1]) == "segment-a"

    checkpoint.save_merge(["u1", "u2"], "merged")
    assert checkpoint.load_merge(["u2", "u1"]) == "merged"

    # A different fingerprint must not observe the other checkpoint's data.
    other = index_mod._VectorIndexBuildCheckpoint(str(tmp_path), "different")
    assert other.load_phase1() is None
    assert other.load_segment([1, 2]) is None

    checkpoint.clear()
    fresh = index_mod._VectorIndexBuildCheckpoint(str(tmp_path), "abc123")
    assert fresh.load_phase1() is None
    assert fresh.load_segment([1, 2]) is None


def test_create_index_resumes_from_checkpoint(monkeypatch, tmp_path):
    """A failed build should resume: reuse phase-1 artifacts and built segments."""

    captured = {}
    fake_dataset = _FakeDataset()
    train_calls = {"ivf": 0}

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            train_calls["ivf"] += 1
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

    def fake_handle_vector_fragment_index(**kwargs):
        captured["fragment_handler_kwargs"] = kwargs
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        return "ivf_ref", None

    def failing_map(**kwargs):
        kwargs["create_fragment_handler"]()
        results = []
        for batch in kwargs["fragment_batches"]:
            if batch == [0]:
                result = {
                    "status": "success",
                    "fragment_ids": list(batch),
                    "segment_index": "segment-0",
                }
                kwargs["on_success"](batch, result)
            else:
                result = {
                    "status": "error",
                    "fragment_ids": list(batch),
                    "error": "boom",
                }
            results.append(result)
        return results

    def succeeding_map(**kwargs):
        captured["resume_batches"] = list(kwargs["fragment_batches"])
        kwargs["create_fragment_handler"]()
        results = []
        for batch in kwargs["fragment_batches"]:
            result = {
                "status": "success",
                "fragment_ids": list(batch),
                "segment_index": "segment-1",
            }
            kwargs["on_success"](batch, result)
            results.append(result)
        return results

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )

    checkpoint_dir = str(tmp_path / "ckpt")
    common_kwargs = dict(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_FLAT",
        name="vector_idx",
        num_workers=2,
        checkpoint_dir=checkpoint_dir,
    )

    monkeypatch.setattr(index_mod, "_map_async_with_pool", failing_map)
    with pytest.raises(RuntimeError, match="Vector index building failed"):
        index_mod.create_index(**common_kwargs)

    assert train_calls["ivf"] == 1

    monkeypatch.setattr(index_mod, "_map_async_with_pool", succeeding_map)
    index_mod.create_index(**common_kwargs)

    # Phase 1 was reused from the checkpoint, and only the failed batch reran.
    assert train_calls["ivf"] == 1
    assert captured["resume_batches"] == [[1]]
    assert sorted(fake_dataset.commit_kwargs["segments"]) == [
        "segment-0",
        "segment-1",
    ]
    # A successful commit clears the checkpoint.
    fingerprint_dirs = list((tmp_path / "ckpt").iterdir())
    assert fingerprint_dirs == []


def test_create_index_merges_segments_to_target(monkeypatch):
    """target_num_segments should trigger a distributed merge before commit."""

    captured = {"map_calls": []}
    fake_dataset = _FourFragmentDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

    def fake_handle_vector_fragment_index(**kwargs):
        return lambda fragment_ids: {"status": "success", "fragment_ids": fragment_ids}

    def fake_handle_merge_segments(**kwargs):
        captured["merge_handler_kwargs"] = kwargs
        return lambda group: {
            "status": "success",
            "segment_index": f"merged({len(group)})",
        }

    def fake_put_vector_index_artifacts(ivf_centroids, pq_codebook):
        return "ivf_ref", None

    def fake_map_async_with_pool(**kwargs):
        captured["map_calls"].append(kwargs)
        handler = kwargs["create_fragment_handler"]()
        if len(captured["map_calls"]) == 1:
            return [
                {
                    "status": "success",
                    "fragment_ids": list(batch),
                    "segment_index": SimpleNamespace(uuid=f"seg-{batch[0]}"),
                }
                for batch in kwargs["fragment_batches"]
            ]
        return [handler(group) for group in kwargs["fragment_batches"]]

    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)
    monkeypatch.setattr(index_mod, "IndicesBuilder", FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.setattr(
        index_mod,
        "_handle_vector_fragment_index",
        fake_handle_vector_fragment_index,
    )
    monkeypatch.setattr(index_mod, "_handle_merge_segments", fake_handle_merge_segments)
    monkeypatch.setattr(
        index_mod,
        "_put_vector_index_artifacts_in_object_store",
        fake_put_vector_index_artifacts,
    )
    monkeypatch.setattr(index_mod, "_map_async_with_pool", fake_map_async_with_pool)

    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_FLAT",
        num_workers=2,
        rows_per_task=100,
        target_num_segments=2,
    )

    assert len(captured["map_calls"]) == 2
    merge_groups = captured["map_calls"][1]["fragment_batches"]
    assert [len(group) for group in merge_groups] == [2, 2]
    assert fake_dataset.commit_kwargs["segments"] == ["merged(2)", "merged(2)"]


def test_plan_segment_merge_groups_balances_by_rows():
    results = [
        {"segment_index": SimpleNamespace(uuid=f"seg-{i}"), "fragment_ids": [i]}
        for i in range(4)
    ]
    sizes = {0: 400, 1: 100, 2: 100, 3: 100}
    groups = index_mod._plan_segment_merge_groups(results, sizes, 2)
    assert len(groups) == 2
    # The 400-row segment stays alone; the three 100-row segments group together.
    group_sizes = sorted(len(group) for group in groups)
    assert group_sizes == [1, 3]


def test_handle_merge_segments_success_and_error(monkeypatch):
    class FakeMergeDataset:
        def __init__(self):
            self.merged_groups = []

        def merge_existing_index_segments(self, segments):
            self.merged_groups.append(segments)
            return SimpleNamespace(uuid="merged")

    fake_dataset = FakeMergeDataset()
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)

    handler = index_mod._handle_merge_segments(dataset_uri="memory://fake")

    result = handler([SimpleNamespace(uuid="a"), SimpleNamespace(uuid="b")])
    assert result["status"] == "success"
    assert result["segment_index"].uuid == "merged"
    assert len(fake_dataset.merged_groups[0]) == 2

    empty_result = handler([])
    assert empty_result["status"] == "error"


def test_commit_index_segments_with_retry_recovers(monkeypatch):
    attempts = {"count": 0}

    class FlakyDataset:
        def commit_existing_index_segments(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient commit conflict")
            attempts["kwargs"] = kwargs
            return "committed"

    monkeypatch.setattr(index_mod.time, "sleep", lambda _: None)

    result = index_mod._commit_index_segments_with_retry(
        lambda: FlakyDataset(),
        index_name="vector_idx",
        column="vector",
        segments=["segment"],
    )

    assert result == "committed"
    assert attempts["count"] == 2
    assert attempts["kwargs"]["segments"] == ["segment"]


def test_create_index_broadcasts_serialized_manifest(monkeypatch):
    captured = {}

    class ManifestDataset(_FakeDataset):
        _ds = SimpleNamespace(serialized_manifest=lambda: b"manifest-bytes")

    fake_dataset = ManifestDataset()

    class FakeIndicesBuilder:
        dimension = 16

        def __init__(self, dataset, column):
            pass

        def train_ivf(self, **kwargs):
            return SimpleNamespace(centroids="ivf_centroids", num_partitions=4)

    _patch_common(monkeypatch, fake_dataset, captured, FakeIndicesBuilder)
    monkeypatch.setattr(index_mod, "_ray_put_index_artifact", lambda value: value)

    index_mod.create_index(
        uri=fake_dataset,
        column="vector",
        index_type="IVF_FLAT",
        num_workers=2,
        num_partitions=4,
    )

    handler_kwargs = captured["fragment_handler_kwargs"]
    assert handler_kwargs["serialized_manifest"] == b"manifest-bytes"
    assert handler_kwargs["dataset_version"] == 1


def test_vector_fragment_handler_uses_manifest_and_version(monkeypatch):
    captured = {"loads": []}
    fake_dataset = _FakeDataset()

    def fake_lance_dataset(*args, **kwargs):
        captured["loads"].append(kwargs)
        return fake_dataset

    monkeypatch.setattr(index_mod, "LanceDataset", fake_lance_dataset)

    handler = index_mod._handle_vector_fragment_index(
        dataset_uri="memory://fake",
        column="vector",
        index_type="IVF_FLAT",
        name="vector_idx",
        index_uuid="vector-index",
        replace=False,
        metric="l2",
        num_partitions=4,
        num_sub_vectors=None,
        ivf_centroids="ivf_centroids",
        pq_codebook=None,
        dataset_version=7,
        serialized_manifest=b"manifest-bytes",
    )

    assert handler([0])["status"] == "success"
    load_kwargs = captured["loads"][0]
    assert load_kwargs["serialized_manifest"] == b"manifest-bytes"
    assert load_kwargs["version"] == 7


def test_map_async_with_pool_retries_exception_objects_from_imap(monkeypatch):
    """Ray Pool yields exception objects on actor death; they must be retried."""

    attempts = {"count": 0}

    class FakePool:
        def __init__(self, processes, ray_remote_args):
            pass

        def imap(self, handler, payloads, chunksize):
            return (handler(payload) for payload in payloads)

        def close(self):
            pass

        def join(self):
            pass

    class FakePoolTaskError(Exception):
        def __init__(self, underlying):
            super().__init__(str(underlying))
            self.underlying = underlying

    def handler(fragment_ids):
        attempts["count"] += 1
        if attempts["count"] == 1:
            # Simulate ray's Pool delivering the failure as a value.
            return FakePoolTaskError(RuntimeError("actor died"))
        return {"status": "success", "fragment_ids": fragment_ids}

    monkeypatch.setattr(index_mod, "Pool", FakePool)

    results = index_mod._map_async_with_pool(
        create_fragment_handler=lambda: handler,
        fragment_batches=[[0]],
        num_workers=1,
        ray_remote_args=None,
        error_prefix="failed",
        max_task_retries=1,
    )

    assert results[0]["status"] == "success"
    assert attempts["count"] == 2

    # With the retry budget exhausted, the exception surfaces as an error dict.
    attempts["count"] = -10
    exhausted = index_mod._map_async_with_pool(
        create_fragment_handler=lambda: (
            lambda fragment_ids: FakePoolTaskError(RuntimeError("still dead"))
        ),
        fragment_batches=[[0]],
        num_workers=1,
        ray_remote_args=None,
        error_prefix="failed",
        max_task_retries=1,
    )
    assert exhausted[0]["status"] == "error"
    assert "still dead" in exhausted[0]["error"]


def test_create_index_rejects_pq_codebook_without_centroids(monkeypatch):
    monkeypatch.setattr(index_mod, "_check_pylance_version", lambda: None)

    with pytest.raises(ValueError, match="ivf_centroids must be provided"):
        index_mod.create_index(
            uri=_FakeDataset(),
            column="vector",
            index_type="IVF_PQ",
            pq_codebook="codebook",
            num_sub_vectors=4,
        )


def test_artifact_digest_distinguishes_contents():
    a = _make_centroids(4, 16)
    values = index_mod.pa.array(
        [float(i) + 1.0 for i in range(4 * 16)], type=index_mod.pa.float32()
    )
    b = index_mod.pa.FixedSizeListArray.from_arrays(values, 16)

    assert index_mod._artifact_digest(None) is None
    assert index_mod._artifact_digest(a) == index_mod._artifact_digest(
        _make_centroids(4, 16)
    )
    assert index_mod._artifact_digest(a) != index_mod._artifact_digest(b)
    assert index_mod._artifact_digest("model-a") != index_mod._artifact_digest(
        "model-b"
    )


def test_validate_pq_codebook_rejects_dimension_mismatch():
    codebook = _make_centroids(4, 8)
    with pytest.raises(ValueError, match="does not match the vector dimension"):
        index_mod._validate_pq_codebook(codebook, 16)

    valid = _make_centroids(4, 16)
    assert index_mod._validate_pq_codebook(valid, 16) is valid
    # Opaque objects are passed through for Lance to validate.
    assert index_mod._validate_pq_codebook("codebook", 16) == "codebook"


def test_plan_fragment_batches_isolates_unknown_size_fragments():
    class BrokenFragment:
        def __init__(self, fragment_id):
            self.fragment_id = fragment_id

        def count_rows(self):
            raise RuntimeError("cannot count")

    fragments = [_FakeFragment(0, 100), _FakeFragment(1, 100), BrokenFragment(2)]
    batches, sizes = index_mod._plan_fragment_batches(
        fragments,
        num_workers=1,
        rows_per_task=None,
        logger=index_mod.logger,
    )
    assert [2] in batches
    assert 2 not in sizes
    assert sorted(fid for batch in batches for fid in batch) == [0, 1, 2]


def test_commit_retry_validates_fragment_coverage(monkeypatch):
    """A commit against a dataset missing built fragments must fail fast."""

    attempts = {"count": 0}

    class ShrunkDataset:
        version = 9

        def get_fragments(self):
            return [_FakeFragment(0, 100)]

        def commit_existing_index_segments(self, **kwargs):
            attempts["count"] += 1
            return self

    monkeypatch.setattr(index_mod.time, "sleep", lambda _: None)

    with pytest.raises(index_mod._IndexCommitCoverageError, match="no longer exist"):
        index_mod._commit_index_segments_with_retry(
            lambda: ShrunkDataset(),
            index_name="vector_idx",
            column="vector",
            segments=["segment"],
            covered_fragment_ids={0, 1},
            built_version=5,
        )

    # No commit attempt was made, and the coverage error was not retried.
    assert attempts["count"] == 0

    result = index_mod._commit_index_segments_with_retry(
        lambda: ShrunkDataset(),
        index_name="vector_idx",
        column="vector",
        segments=["segment"],
        covered_fragment_ids={0},
        built_version=5,
    )
    assert isinstance(result, ShrunkDataset)


def test_checkpoint_write_failures_are_non_fatal(tmp_path):
    checkpoint = index_mod._VectorIndexBuildCheckpoint(str(tmp_path), "abc123")

    class Unpicklable:
        def __reduce__(self):
            raise TypeError("cannot pickle")

    # Must not raise even though the payload cannot be serialized.
    checkpoint.save_segment([1], Unpicklable())
    assert checkpoint.load_segment([1]) is None


def test_segment_files_exist_local(tmp_path):
    dataset_dir = tmp_path / "ds.lance"
    seg_dir = dataset_dir / "_indices" / "seg-1"
    seg_dir.mkdir(parents=True)
    (seg_dir / "index.idx").write_bytes(b"data")

    present = SimpleNamespace(uuid="seg-1")
    missing = SimpleNamespace(uuid="seg-2")
    assert index_mod._segment_files_exist(str(dataset_dir), present)
    assert not index_mod._segment_files_exist(str(dataset_dir), missing)
    # Unknown storage schemes are trusted rather than rebuilt.
    assert index_mod._segment_files_exist("memory://fake", missing)


def test_vector_fragment_handler_guards_scan_batch_size(monkeypatch):
    """Worker builds must default LANCE_DEFAULT_BATCH_SIZE to a safe value.

    Without the guard, a large dataset block_size inflates the scan batch to
    block_size/4 rows; batches with rows * dim > 2^32 hit an arrow-rs
    FixedSizeList take() overflow (panic or silent vector corruption).
    """
    fake_dataset = _FakeDataset()
    monkeypatch.setattr(index_mod, "LanceDataset", lambda *args, **kwargs: fake_dataset)
    monkeypatch.delenv("LANCE_DEFAULT_BATCH_SIZE", raising=False)

    handler = index_mod._handle_vector_fragment_index(
        dataset_uri="memory://fake",
        column="vector",
        index_type="IVF_PQ",
        name="vector_idx",
        index_uuid="vector-index",
        replace=False,
        metric="l2",
        num_partitions=4,
        num_sub_vectors=4,
        ivf_centroids="ivf_centroids",
        pq_codebook="pq_codebook",
        block_size=100 * 1024 * 1024,
    )
    assert handler([0])["status"] == "success"
    assert index_mod.os.environ["LANCE_DEFAULT_BATCH_SIZE"] == "8192"

    # An explicit operator-provided value is respected.
    monkeypatch.setenv("LANCE_DEFAULT_BATCH_SIZE", "4096")
    assert handler([0])["status"] == "success"
    assert index_mod.os.environ["LANCE_DEFAULT_BATCH_SIZE"] == "4096"
