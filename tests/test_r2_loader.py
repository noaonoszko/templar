# ruff: noqa
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from dotenv import load_dotenv
import io
import asyncio
import pyarrow as pa
import pyarrow.parquet as pq

# Find and load the correct .env file
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"Required .env file not found at {env_path}")

# Load environment variables before any other imports
load_dotenv(env_path, override=True)

# Verify environment variables are loaded
required_vars = [
    "R2_GRADIENTS_ACCOUNT_ID",
    "R2_GRADIENTS_BUCKET_NAME",
    "R2_GRADIENTS_READ_ACCESS_KEY_ID",
    "R2_GRADIENTS_READ_SECRET_ACCESS_KEY",
    "R2_GRADIENTS_WRITE_ACCESS_KEY_ID",
    "R2_GRADIENTS_WRITE_SECRET_ACCESS_KEY",
    "R2_DATASET_ACCOUNT_ID",
    "R2_DATASET_BUCKET_NAME",
    "R2_DATASET_READ_ACCESS_KEY_ID",
    "R2_DATASET_READ_SECRET_ACCESS_KEY",
    "R2_DATASET_WRITE_ACCESS_KEY_ID",
    "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables in .env file: {', '.join(missing_vars)}"
    )

# Only import after environment variables are loaded and verified
import pytest
from transformers import AutoTokenizer
from tplr.logging import logger, debug, T
from tplr.r2_dataset import R2DatasetLoader
from tplr.hparams import load_hparams
import torch
import random
from neurons.validator import retry_call
import s3fs
from tplr.config import BUCKET_SECRETS
import threading
import time
import concurrent.futures


# Enable debug logging for tests
debug()


@pytest.mark.asyncio
async def test_local_parquet_loader():
    """
    Simple integration test to ensure R2DatasetLoader can fetch pages from your R2 parquet data.
    Adjust environment variables & the code below to point to your actual dataset, then run:
        pytest tests/test_local_parquet_loader.py
    """

    start_time = T()
    logger.info("Starting test_local_parquet_loader")

    # Make sure the required R2 environment variables are set
    missing_vars = []
    for var in [
        "R2_DATASET_ACCOUNT_ID",
        "R2_DATASET_BUCKET_NAME",
        "R2_DATASET_READ_ACCESS_KEY_ID",
        "R2_DATASET_READ_SECRET_ACCESS_KEY",
        "R2_DATASET_WRITE_ACCESS_KEY_ID",
        "R2_DATASET_WRITE_SECRET_ACCESS_KEY",
    ]:
        if not os.environ.get(var):
            missing_vars.append(var)

    if missing_vars:
        pytest.skip(f"Missing environment variables: {', '.join(missing_vars)}")

    # Instantiate a tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer
    logger.info(f"Tokenizer loaded ({T() - start_time:.2f}s)")

    # Prepare test parameters
    offset = 0
    n_pages = 2  # The number of random pages to fetch
    seed = "my-test-seed"  # Arbitrary seed for reproducibility
    batch_size = 2
    sequence_length = 128

    # 1. Generate random pages
    pages = await R2DatasetLoader.next_pages(offset=offset, n_pages=n_pages, seed=seed)
    logger.info(f"Random pages selected: {pages} ({T() - start_time:.2f}s)")

    # 2. Create loader
    loader = await R2DatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages,
        tokenizer=tokenizer,
        pack_samples=False,
    )
    logger.info(f"Loader created ({T() - start_time:.2f}s)")

    # 3. Iterate over the loader a few batches
    batch_count = 0
    try:
        for batch in loader:
            logger.info(f"[cyan]Batch {batch_count} shape: {batch.shape}[/cyan]")

            # Decode each sequence in the batch
            for i, sequence in enumerate(batch):
                # Convert to tokens, skip padding tokens
                tokens = sequence[sequence != tokenizer.pad_token_id].tolist()
                text = tokenizer.decode(tokens)
                logger.info(f"Sequence {i}:")
                logger.info(f"First 50 tokens: {tokens[:50]}...")
                logger.info(f"Text: {text[:200]}...")
                logger.info("[dim]" + "-" * 80 + "[/dim]")

            batch_count += 1
            if batch_count >= 2:  # Look at first 2 batches
                break
    except Exception as e:
        logger.error(f"[red]Error during iteration: {str(e)}[/red]", exc_info=True)

    # Basic assertion: We expect at least 1 batch if pages > 0
    assert batch_count > 0, "No batches were produced by the R2DatasetLoader"
    logger.info(
        f"[green]Test completed successfully. Processed {batch_count} batches ({T() - start_time:.2f}s)[/green]"
    )


@pytest.mark.asyncio
async def test_large_page_offset_handling():
    """
    Test that the loader correctly handles large page offsets that might exceed row group bounds.
    This specifically tests the fix for the row group index calculation.
    """
    start_time = T()
    logger.info("Starting test_large_page_offset_handling")

    # Load tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer

    # Get dataset configs to find maximum rows
    configs_data = await R2DatasetLoader.fetch_dataset_configs()

    # Find a config with the most rows to test boundary conditions
    max_rows_config = max(configs_data.items(), key=lambda x: x[1]["num_rows"])
    config_name = max_rows_config[0]
    num_rows = max_rows_config[1]["num_rows"]

    # Test cases with different offsets
    test_cases = [
        (0, "start of dataset"),
        (num_rows // 2, "middle of dataset"),
        (num_rows - 200, "near end of dataset"),  # Leave room for page size
    ]

    for offset, description in test_cases:
        logger.info(f"\nTesting {description} (offset: {offset})")

        # Create a single-page test with specific offset
        pages = [(config_name, offset, "train")]

        try:
            # Create loader with test page
            loader = await R2DatasetLoader.create(
                batch_size=2,
                sequence_length=128,
                pages_info=pages,
                tokenizer=tokenizer,
                pack_samples=False,
            )

            # Verify we can get at least one batch
            batch = next(iter(loader))

            # Basic validation
            assert batch is not None, f"Failed to get batch for offset {offset}"
            assert batch.shape == (2, 128), f"Unexpected batch shape: {batch.shape}"

            # Verify the batch contains valid token IDs
            for sequence in batch:
                valid_tokens = sequence[sequence != tokenizer.pad_token_id]
                assert len(valid_tokens) > 0, "Sequence contains no valid tokens"

                # Decode to verify we got meaningful text
                text = tokenizer.decode(valid_tokens)
                assert len(text.strip()) > 0, "Decoded text is empty"

            logger.info(
                f"[green]Successfully processed batch for offset {offset}[/green]"
            )

        except Exception as e:
            logger.error(
                f"[red]Error processing offset {offset}: {str(e)}[/red]", exc_info=True
            )
            raise

    logger.info(
        f"[green]All offset tests completed successfully ({T() - start_time:.2f}s)[/green]"
    )


@pytest.mark.asyncio
async def test_seed_consistency():
    """
    Test that R2DatasetLoader consistently returns the same pages for the same seed
    and different pages for different seeds.
    """
    start_time = T()
    logger.info("Starting test_seed_consistency")

    # Load tokenizer
    hparams = load_hparams()
    tokenizer = hparams.tokenizer

    # Test parameters
    offset = 1000  # Arbitrary offset
    n_pages = 2
    batch_size = 2
    sequence_length = 128

    # Test same seed returns same pages
    seed1 = 42
    seed2 = 42
    seed3 = 43  # Different seed

    # Get pages with same seed
    pages1 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed1
    )
    pages2 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed2
    )

    # Get pages with different seed
    pages3 = await R2DatasetLoader.next_pages(
        offset=offset, n_pages=n_pages, seed=seed3
    )

    # Test same seed produces same pages
    assert pages1 == pages2, "Same seed should produce identical pages"

    # Test different seeds produce different pages
    assert pages1 != pages3, "Different seeds should produce different pages"

    # Test page content consistency
    loader1 = await R2DatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages1,
        tokenizer=tokenizer,
        pack_samples=False,
    )

    loader2 = await R2DatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages2,
        tokenizer=tokenizer,
        pack_samples=False,
    )

    # Get first batch from each loader and convert to tensors
    batch1 = torch.tensor(next(iter(loader1)))
    batch2 = torch.tensor(next(iter(loader2)))

    # Test content consistency
    assert torch.equal(batch1, batch2), (
        "Same seed should produce identical batch content"
    )

    # Test seed range
    seeds = [random.randint(0, 10000) for _ in range(10)]
    unique_pages = set()

    for seed in seeds:
        pages = await R2DatasetLoader.next_pages(
            offset=offset, n_pages=n_pages, seed=seed
        )
        page_tuple = tuple(
            [(p[0], p[1]) for p in pages]
        )  # Convert to tuple for hashing
        unique_pages.add(page_tuple)

    # Check if we got different pages for different seeds
    assert len(unique_pages) > 1, "Random seeds should produce variety of pages"

    logger.success(
        f"[green]Seed consistency test completed successfully ({T() - start_time:.2f}s)[/green]"
    )


# --- Test: Transient failures recover (retry succeeds) ---
@pytest.mark.asyncio
async def test_retry_mechanism_success(monkeypatch):
    """
    Ensure that transient errors during fs.open are retried and eventually succeed.
    Edge case: The dummy filesystem fails a fixed number of times (2), then returns valid data.
    """
    # Create a minimal parquet file in memory
    table = pa.table({"text": ["Testing retry", "More testing"]})
    parquet_buffer = io.BytesIO()
    pq.write_table(table, parquet_buffer)
    parquet_bytes = parquet_buffer.getvalue()

    # Define a buffer factory that returns a new BytesIO each time
    def buffer_factory():
        return io.BytesIO(parquet_bytes)

    # DummyFS simulates transient failures before finally succeeding.
    class DummyFS:
        def __init__(self, real_fs, fail_times=2, buffer_factory=None):
            self.real_fs = real_fs
            self.fail_times = fail_times
            self.call_count = 0
            self.buffer_factory = buffer_factory

        def open(self, *args, **kwargs):
            if self.call_count < self.fail_times:
                self.call_count += 1
                raise Exception("Simulated transient error")
            return self.buffer_factory()

        def __getattr__(self, attr):
            return getattr(self.real_fs, attr)

    # Patch R2DatasetLoader._get_fs to return our DummyFS; note the lambda accepts self
    real_fs = R2DatasetLoader._get_fs()
    dummy_fs = DummyFS(real_fs, fail_times=2, buffer_factory=buffer_factory)
    monkeypatch.setattr(R2DatasetLoader, "_get_fs", lambda self: dummy_fs)

    # Setup dummy metadata to bypass actual R2 calls.
    dummy_config = "dummy_config"
    dummy_shard = {"path": "dummy/path", "num_rows": 2}

    async def dummy_load_r2_metadata(self):
        return (
            {
                dummy_config: {
                    "total_rows": 2,
                    "split": "train",
                    "shards": [dummy_shard],
                }
            },
            {"configs": [{"config_name": dummy_config}]},
        )

    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", dummy_load_r2_metadata)

    # Initialize a basic tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a loader instance.
    loader = R2DatasetLoader(
        batch_size=1, sequence_length=10, tokenizer=tokenizer, pack_samples=False
    )
    loader.num_rows_per_page = 2

    # Trigger processing a page.
    tokens = await loader._process_page(
        (dummy_config, 0, "train"), asyncio.Semaphore(1)
    )
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) > 0, "Tokens list should not be empty"
    # The DummyFS should have failed exactly 2 times before succeeding.
    assert dummy_fs.call_count == 2, (
        "DummyFS did not simulate the expected number of transient errors"
    )


# --- Test: Persistent failure raises exception ---
@pytest.mark.asyncio
async def test_retry_mechanism_failure(monkeypatch):
    """
    Ensure that persistent errors in fs.open eventually raise an exception after max retries.
    Edge case: The dummy filesystem always fails.
    """

    # AlwaysFailFS simulates persistent errors by always raising an Exception.
    class AlwaysFailFS:
        def open(self, *args, **kwargs):
            raise Exception("Persistent transient error")

        def __getattr__(self, attr):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(R2DatasetLoader, "_get_fs", lambda self: AlwaysFailFS())

    # Setup dummy metadata (with "self" parameter).
    dummy_config = "dummy_config"
    dummy_shard = {"path": "dummy/path", "num_rows": 2}

    async def dummy_load_r2_metadata(self):
        return (
            {
                dummy_config: {
                    "total_rows": 2,
                    "split": "train",
                    "shards": [dummy_shard],
                }
            },
            {"configs": [{"config_name": dummy_config}]},
        )

    monkeypatch.setattr(R2DatasetLoader, "_load_r2_metadata", dummy_load_r2_metadata)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = R2DatasetLoader(
        batch_size=1, sequence_length=10, tokenizer=tokenizer, pack_samples=False
    )
    loader.num_rows_per_page = 2

    with pytest.raises(Exception, match="Persistent transient error"):
        await loader._process_page((dummy_config, 0, "train"), asyncio.Semaphore(1))


# --- New Tests for the retry_call helper ---


@pytest.mark.asyncio
async def test_retry_call_immediate_success():
    """
    Test that retry_call returns immediately if the provided async function succeeds on the first attempt.
    """
    call_count = 0

    async def immediate_func(val):
        nonlocal call_count
        call_count += 1
        return val + 1

    result = await retry_call(
        immediate_func, 5, attempts=3, delay=0.1, context="immediate_func"
    )
    assert result == 6
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_call_success_after_failures():
    """
    Test that retry_call eventually succeeds after a few transient failures.
    """
    call_count = 0

    async def flaky_func(val):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Transient error")
        return val * 2

    result = await retry_call(
        flaky_func, 5, attempts=5, delay=0.1, context="flaky_func"
    )
    assert result == 10
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_call_exhaust_failures():
    """
    Test that retry_call returns None after exhausting all attempts when the function always fails.
    """
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise Exception("Persistent error")

    result = await retry_call(always_fail, attempts=3, delay=0.1, context="always_fail")
    assert result is None
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_in_next_pages(monkeypatch):
    """
    Test integration of retry_call with R2DatasetLoader.next_pages.
    Simulate transient failures before success.
    """
    call_count = 0

    async def dummy_next_pages(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Simulated transient failure in next_pages")
        return [("dummy_config", 0, "train")]

    # Monkey-patch R2DatasetLoader.next_pages to use our dummy function.
    from tplr.r2_dataset import R2DatasetLoader

    original_next_pages = R2DatasetLoader.next_pages
    R2DatasetLoader.next_pages = dummy_next_pages

    result = await retry_call(
        R2DatasetLoader.next_pages,
        offset=0,
        n_pages=1,
        seed=42,
        attempts=4,
        delay=0.1,
        context="dummy next pages",
    )
    assert result == [("dummy_config", 0, "train")]

    # Restore the original method to avoid side effects.
    R2DatasetLoader.next_pages = original_next_pages


# Dummy S3FileSystem for testing
class DummyS3FileSystem:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_round_robin_sequential(monkeypatch):
    # Setup: Configure BUCKET_SECRETS with a "multiple" key containing two endpoints.
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "accountA",
                "name": "bucketA",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                    "write": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                },
            },
            {
                "account_id": "accountB",
                "name": "bucketB",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                    "write": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                },
            },
        ]
    }
    # Override the dataset configuration in the global BUCKET_SECRETS.
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset round robin counter and fs cache
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Monkey-patch s3fs.S3FileSystem to use our DummyS3FileSystem
    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    # Action: Call _get_fs() three times
    fs1 = R2DatasetLoader._get_fs()  # Expect accountA
    fs2 = R2DatasetLoader._get_fs()  # Expect accountB
    fs3 = (
        R2DatasetLoader._get_fs()
    )  # Expect cycle back to accountA (likely returning the cached instance)

    # Expectations:
    # First call should return instance configured for accountA.
    endpoint1 = fs1.kwargs["client_kwargs"]["endpoint_url"]
    assert endpoint1 == "https://accountA.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountA, got {endpoint1}"
    )

    # Second call should return instance configured for accountB.
    endpoint2 = fs2.kwargs["client_kwargs"]["endpoint_url"]
    assert endpoint2 == "https://accountB.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountB, got {endpoint2}"
    )

    # Third call should cycle back to accountA.
    endpoint3 = fs3.kwargs["client_kwargs"]["endpoint_url"]
    assert endpoint3 == "https://accountA.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountA on cycle, got {endpoint3}"
    )
    # Verify that the fs instance is cached: fs1 and fs3 should be the same object.
    assert fs1 is fs3, (
        "Expected the same cached S3FileSystem instance for repeated accountA selection"
    )


def test_round_robin_single_entry(monkeypatch):
    # Setup: Configure BUCKET_SECRETS["dataset"] with a "multiple" key containing a single endpoint (endpoint A).
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "accountA",
                "name": "bucketA",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                    "write": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                },
            }
        ]
    }
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset the round robin counter and file system cache.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Monkey-patch s3fs.S3FileSystem with our DummyS3FileSystem.
    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    # Action: Call _get_fs() multiple times.
    fs1 = R2DatasetLoader._get_fs()
    fs2 = R2DatasetLoader._get_fs()
    fs3 = R2DatasetLoader._get_fs()

    # Expectations:
    # Every call should return an S3FileSystem instance configured for endpoint A.
    endpoint1 = fs1.kwargs["client_kwargs"]["endpoint_url"]
    endpoint2 = fs2.kwargs["client_kwargs"]["endpoint_url"]
    endpoint3 = fs3.kwargs["client_kwargs"]["endpoint_url"]

    assert endpoint1 == "https://accountA.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountA, got {endpoint1}"
    )
    assert endpoint2 == "https://accountA.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountA, got {endpoint2}"
    )
    assert endpoint3 == "https://accountA.r2.cloudflarestorage.com", (
        f"Expected endpoint for accountA, got {endpoint3}"
    )

    # The fs instances should be cached hence identical.
    assert fs1 is fs2 and fs1 is fs3, (
        "Expected the same cached S3FileSystem instance for accountA"
    )

    # The round robin counter should increment even if only one entry exists.
    assert R2DatasetLoader._round_robin_index == 3, (
        f"Expected round robin index to be 3, got {R2DatasetLoader._round_robin_index}"
    )


def test_configuration_without_multiple(monkeypatch):
    # Setup: Configure BUCKET_SECRETS["dataset"] without the "multiple" key (using a single, default configuration).
    test_dataset_config = {
        "account_id": "accountDefault",
        "name": "bucketDefault",
        "credentials": {
            "read": {
                "access_key_id": "READ_ACCESS",
                "secret_access_key": "READ_SECRET",
            },
            "write": {
                "access_key_id": "WRITE_ACCESS",
                "secret_access_key": "WRITE_SECRET",
            },
        },
    }
    from tplr.config import BUCKET_SECRETS
    from tplr.r2_dataset import R2DatasetLoader
    import s3fs

    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset the round robin counter and fs cache.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Monkey-patch s3fs.S3FileSystem with a dummy implementation.
    class DummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    # Action: Call _get_fs() and inspect the returned instance.
    fs = R2DatasetLoader._get_fs()

    # Expectations:
    # The returned S3FileSystem should be configured based on the provided single endpoint configuration.
    endpoint = fs.kwargs["client_kwargs"]["endpoint_url"]
    expected_endpoint = "https://accountDefault.r2.cloudflarestorage.com"
    assert endpoint == expected_endpoint, (
        f"Expected endpoint {expected_endpoint}, got {endpoint}"
    )


def test_round_robin_caching(monkeypatch):
    # Setup: With a given multiple-endpoint configuration, verify that caching behaves as expected.
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "accountA",
                "name": "bucketA",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                    "write": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                },
            },
            {
                "account_id": "accountB",
                "name": "bucketB",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                    "write": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                },
            },
        ]
    }
    from tplr.config import BUCKET_SECRETS
    from tplr.r2_dataset import R2DatasetLoader
    import s3fs

    # Override dataset configuration.
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset round robin counter and fs cache.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Monkey-patch s3fs.S3FileSystem with DummyS3FileSystem.
    class DummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    # Action: Call _get_fs() repeatedly (e.g., 6 times to cycle through endpoints).
    fs_instances = [R2DatasetLoader._get_fs() for _ in range(6)]

    # Determine expected endpoints for round robin:
    # For a 2-endpoint configuration, indices 0,2,4 should correspond to accountA,
    # and indices 1,3,5 should correspond to accountB.
    endpoint_A = "https://accountA.r2.cloudflarestorage.com"
    endpoint_B = "https://accountB.r2.cloudflarestorage.com"

    # Check that instances returning the same endpoint are identical (cached)
    assert fs_instances[0].kwargs["client_kwargs"]["endpoint_url"] == endpoint_A, (
        "Expected accountA endpoint at index 0"
    )
    assert fs_instances[1].kwargs["client_kwargs"]["endpoint_url"] == endpoint_B, (
        "Expected accountB endpoint at index 1"
    )
    assert fs_instances[2].kwargs["client_kwargs"]["endpoint_url"] == endpoint_A, (
        "Expected accountA endpoint at index 2"
    )
    assert fs_instances[3].kwargs["client_kwargs"]["endpoint_url"] == endpoint_B, (
        "Expected accountB endpoint at index 3"
    )
    assert fs_instances[4].kwargs["client_kwargs"]["endpoint_url"] == endpoint_A, (
        "Expected accountA endpoint at index 4"
    )
    assert fs_instances[5].kwargs["client_kwargs"]["endpoint_url"] == endpoint_B, (
        "Expected accountB endpoint at index 5"
    )

    # Verify caching: same instance for accountA calls at indices 0, 2, 4.
    assert fs_instances[0] is fs_instances[2] is fs_instances[4], (
        "Expected the same cached instance for accountA"
    )
    # Similarly, same instance for accountB calls at indices 1, 3, 5.
    assert fs_instances[1] is fs_instances[3] is fs_instances[5], (
        "Expected the same cached instance for accountB"
    )

    # Additionally, cache should have exactly 2 entries.
    assert len(R2DatasetLoader._fs_cache) == 2, (
        f"Expected fs cache to have 2 entries, got {len(R2DatasetLoader._fs_cache)}"
    )


def test_round_robin_thread_safety(monkeypatch):
    # Setup: Configure BUCKET_SECRETS["dataset"] with two endpoints.
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "accountA",
                "name": "bucketA",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                    "write": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                },
            },
            {
                "account_id": "accountB",
                "name": "bucketB",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                    "write": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                },
            },
        ]
    }
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset shared state.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Monkey-patch s3fs.S3FileSystem with DummyS3FileSystem that records instantiation params.
    class DummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    num_threads = 10
    calls_per_thread = 10
    total_calls = num_threads * calls_per_thread
    results = []  # Will store all fs instances.

    # Worker function: call _get_fs() calls_per_thread times.
    def worker():
        for _ in range(calls_per_thread):
            fs_instance = R2DatasetLoader._get_fs()
            results.append(fs_instance)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Expectation: The round_robin_index should equal total_calls.
    assert R2DatasetLoader._round_robin_index == total_calls, (
        f"Expected _round_robin_index to be {total_calls}, got {R2DatasetLoader._round_robin_index}"
    )

    # Valid endpoint URLs we expect.
    valid_endpoints = {
        "https://accountA.r2.cloudflarestorage.com",
        "https://accountB.r2.cloudflarestorage.com",
    }

    # All returned instances must have a valid endpoint.
    for instance in results:
        endpoint_url = instance.kwargs["client_kwargs"]["endpoint_url"]
        assert endpoint_url in valid_endpoints, (
            f"Invalid endpoint {endpoint_url} found in instance"
        )

    # Check caching: For each endpoint, repeated calls should return the same instance.
    # Build a mapping: endpoint_url -> instance (first encountered).
    endpoint_to_instance = {}
    for instance in results:
        endpoint_url = instance.kwargs["client_kwargs"]["endpoint_url"]
        if endpoint_url not in endpoint_to_instance:
            endpoint_to_instance[endpoint_url] = instance
        else:
            # Ensure the cached instance is always returned.
            assert instance is endpoint_to_instance[endpoint_url], (
                f"Different instances returned for endpoint {endpoint_url}"
            )


# Test Case 6: Concurrency under high load
# - Setup: Use a "multiple" endpoints configuration as above.
# - Action: Fire a high number of concurrent calls (e.g., 100 calls in parallel using threads or async tasks).
# - Expectations:
#   * The round robin mechanism should still cycle through the endpoints appropriately.
#   * The fs_cache should remain consistent, and no race conditions or exceptions should occur.
def test_round_robin_high_concurrency(monkeypatch):
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "accountA",
                "name": "bucketA",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                    "write": {
                        "access_key_id": "AKIAA",
                        "secret_access_key": "secretA",
                    },
                },
            },
            {
                "account_id": "accountB",
                "name": "bucketB",
                "credentials": {
                    "read": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                    "write": {
                        "access_key_id": "AKIAB",
                        "secret_access_key": "secretB",
                    },
                },
            },
        ]
    }
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset round robin counter and fs cache.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Use the DummyS3FileSystem to record instantiation parameters.
    class DummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    total_calls = 200
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(R2DatasetLoader._get_fs) for _ in range(total_calls)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Check that the round robin index equals the total number of calls.
    assert R2DatasetLoader._round_robin_index == total_calls, (
        f"Expected round robin index {total_calls}, got {R2DatasetLoader._round_robin_index}"
    )

    valid_endpoints = {
        "https://accountA.r2.cloudflarestorage.com",
        "https://accountB.r2.cloudflarestorage.com",
    }
    for instance in results:
        ep = instance.kwargs["client_kwargs"]["endpoint_url"]
        assert ep in valid_endpoints, f"Unexpected endpoint encountered: {ep}"

    # Verify caching: For instances of each endpoint, all calls should return the same object.
    cache = {}
    for instance in results:
        ep = instance.kwargs["client_kwargs"]["endpoint_url"]
        if ep not in cache:
            cache[ep] = instance
        else:
            assert instance is cache[ep], f"Different instances found for endpoint {ep}"
    # The cache size must match the number of endpoints.
    assert len(R2DatasetLoader._fs_cache) == 2, (
        f"Expected fs_cache size 2, got {len(R2DatasetLoader._fs_cache)}"
    )


# Test Case 7: Lock robustness with simulated delay
# - Setup: Temporarily simulate a delay inside S3FileSystem creation in _get_fs().
# - Action: While the creation is artificially delayed, trigger multiple concurrent calls.
# - Expectations:
#   * The lock should prevent race conditions.
#   * After the delay, the round_robin_index and fs_cache should reflect correct, sequential, and non-corrupted increments.
def test_lock_robustness_simulated_delay(monkeypatch):
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "delayA",
                "name": "bucketDelayA",
                "credentials": {
                    "read": {
                        "access_key_id": "DELAYA",
                        "secret_access_key": "secretDelayA",
                    },
                    "write": {
                        "access_key_id": "DELAYA",
                        "secret_access_key": "secretDelayA",
                    },
                },
            },
            {
                "account_id": "delayB",
                "name": "bucketDelayB",
                "credentials": {
                    "read": {
                        "access_key_id": "DELAYB",
                        "secret_access_key": "secretDelayB",
                    },
                    "write": {
                        "access_key_id": "DELAYB",
                        "secret_access_key": "secretDelayB",
                    },
                },
            },
        ]
    }
    BUCKET_SECRETS["dataset"] = test_dataset_config
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Create a DummyS3FileSystem that delays initialization.
    class DelayedDummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)  # Simulated delay
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DelayedDummyS3FileSystem)

    total_calls = 50
    results = []

    def worker():
        fs_inst = R2DatasetLoader._get_fs()
        results.append(fs_inst)

    threads = []
    for _ in range(total_calls):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify total round robin index.
    assert R2DatasetLoader._round_robin_index == total_calls, (
        f"Expected round robin index {total_calls}, got {R2DatasetLoader._round_robin_index}"
    )

    # Validate that each instance has the correct endpoint and caching works.
    valid_endpoints = {
        "https://delayA.r2.cloudflarestorage.com",
        "https://delayB.r2.cloudflarestorage.com",
    }
    cache = {}
    for inst in results:
        ep = inst.kwargs["client_kwargs"]["endpoint_url"]
        assert ep in valid_endpoints, f"Unexpected endpoint {ep}"
        if ep in cache:
            assert inst is cache[ep], (
                f"Caching failure: Different instances for endpoint {ep}"
            )
        else:
            cache[ep] = inst


# Test Case 8: Validate configuration correctness in returned instances
# - Setup: For each endpoint configuration in the "multiple" list, ensure that the expected endpoint_url (derived from the account_id)
#   is known.
# - Action: Call _get_fs() and inspect the 'endpoint_url' in the returned S3FileSystem instance.
# - Expectations:
#   * The instance's configuration should match the expected endpoint for the selected account_id.
def test_validate_configuration_correctness(monkeypatch):
    test_dataset_config = {
        "multiple": [
            {
                "account_id": "confA",
                "name": "bucketConfA",
                "credentials": {
                    "read": {
                        "access_key_id": "CONFA",
                        "secret_access_key": "secretConfA",
                    },
                    "write": {
                        "access_key_id": "CONFA",
                        "secret_access_key": "secretConfA",
                    },
                },
            },
            {
                "account_id": "confB",
                "name": "bucketConfB",
                "credentials": {
                    "read": {
                        "access_key_id": "CONFB",
                        "secret_access_key": "secretConfB",
                    },
                    "write": {
                        "access_key_id": "CONFB",
                        "secret_access_key": "secretConfB",
                    },
                },
            },
        ]
    }
    BUCKET_SECRETS["dataset"] = test_dataset_config

    # Reset state.
    R2DatasetLoader._round_robin_index = 0
    R2DatasetLoader._fs_cache = {}

    # Dummy S3FileSystem capturing instantiation parameters.
    class DummyS3FileSystem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(s3fs, "S3FileSystem", DummyS3FileSystem)

    # Call _get_fs() multiple times to get both endpoints.
    instances = [R2DatasetLoader._get_fs() for _ in range(10)]
    expected_endpoints = {
        "confA": "https://confA.r2.cloudflarestorage.com",
        "confB": "https://confB.r2.cloudflarestorage.com",
    }
    for inst in instances:
        ep = inst.kwargs["client_kwargs"]["endpoint_url"]
        # Determine which account id based on the endpoint.
        if ep == expected_endpoints["confA"]:
            account = "confA"
        elif ep == expected_endpoints["confB"]:
            account = "confB"
        else:
            account = None
        assert account is not None, (
            f"Endpoint {ep} does not match any expected configuration."
        )
        expected_region = R2DatasetLoader.CF_REGION_NAME
        region = inst.kwargs["client_kwargs"].get("region_name")
        assert region == expected_region, (
            f"Expected region {expected_region}, got {region}"
        )
