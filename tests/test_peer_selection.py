import numpy as np
import pytest

import tplr
from neurons.validator import Validator

hparams = tplr.load_hparams()


class TestPeerSelection:
    """Tests for peer selection functionality."""

    class TestSelectInitialGatherPeers:
        """Tests for select_initial_peers method."""

        def _assert_basic_requirements(self, mock_validator: Validator, success: bool):
            """Helper method to verify peer selection results.

            Args:
                mock_validator: The mock validator instance
                expected_peer_count: Expected number of peers
                top_incentive_indices: Expected peer indices (optional)
            """
            # Peers are successfully updated
            assert success is True

            # All peers should be active
            assert set(mock_validator.comms.peers).issubset(
                set(mock_validator.comms.active_peers)
            )

            # Number of peers is within range
            assert (
                mock_validator.hparams.minimum_peers
                <= len(mock_validator.comms.peers)
                <= mock_validator.hparams.max_topk_peers
            )

        def _assert_all_miners_with_non_zero_incentive_in_peers(self, mock_validator):
            """Verify that all miners with non-zero incentive are included in peers."""
            non_zero_incentive_uids = np.nonzero(mock_validator.metagraph.I)[0]
            return set(non_zero_incentive_uids).issubset(mock_validator.comms.peers)

        @pytest.mark.parametrize("num_non_zero_incentive", [100, 200])
        def test_with_only_incentive_peers(self, mock_validator, top_incentive_indices):
            # Call the actual method
            success = mock_validator.select_initial_peers()

            assert success is True

            self._assert_basic_requirements(
                mock_validator=mock_validator, success=success
            )

            # Peers should be the top incentive miners
            assert set(mock_validator.comms.peers) == set(top_incentive_indices)

            # Verify peer count
            assert (
                len(mock_validator.comms.peers) == mock_validator.hparams.max_topk_peers
            )

        @pytest.mark.parametrize(
            "num_non_zero_incentive", [0, hparams.minimum_peers - 1]
        )
        def test_with_some_zero_incentive_peers(self, mock_validator):
            # Call the actual method
            success = mock_validator.select_initial_peers()

            self._assert_basic_requirements(
                mock_validator=mock_validator, success=success
            )
            self._assert_all_miners_with_non_zero_incentive_in_peers(mock_validator)

            # Verify peer count
            assert (
                len(mock_validator.comms.peers) == mock_validator.hparams.max_topk_peers
            )

        @pytest.mark.parametrize(
            "num_non_zero_incentive",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        @pytest.mark.parametrize(
            "num_active_miners",
            [hparams.minimum_peers, hparams.max_topk_peers - 1],
            indirect=True,
        )
        def test_with_non_full_peer_list(self, mock_validator):
            # Call the actual method
            success = mock_validator.select_initial_peers()

            self._assert_basic_requirements(
                mock_validator=mock_validator, success=success
            )
            self._assert_all_miners_with_non_zero_incentive_in_peers(mock_validator)

            # Assert that all active peers are in peers and that it's fewer than
            # max allowed
            assert len(mock_validator.comms.peers) < hparams.max_topk_peers

        @pytest.mark.parametrize(
            "num_non_zero_incentive",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        @pytest.mark.parametrize(
            "num_active_miners",
            [0, hparams.minimum_peers - 1],
            indirect=True,
        )
        def test_with_too_few_active_peers(self, mock_validator):
            # Call the actual method
            success = mock_validator.select_initial_peers()

            assert success is False
            assert mock_validator.comms.peers is None

    @pytest.fixture
    def top_incentive_indices(self, mock_validator):
        incentives = mock_validator.metagraph.I
        return np.argsort(incentives)[::-1][: mock_validator.hparams.max_topk_peers]

    @pytest.fixture
    def num_non_zero_incentive(self):
        """Default fixture for number of non-zero incentive miners."""
        return 100  # Default value if not parameterized

    @pytest.fixture
    def num_active_miners(self, request):
        """Fixture for number of active miners.
        Returns parameterized value if available, otherwise returns default."""
        try:
            return request.param
        except AttributeError:
            return 250  # Default value if not parameterized

    @pytest.fixture
    def mock_metagraph(self, mocker, num_non_zero_incentive, num_miners=250):
        """Fixture that creates a mock metagraph with a specified number of miners and incentive distribution."""
        metagraph = mocker.Mock()

        metagraph.uids = np.arange(num_miners)

        # Create incentive distribution
        non_zero_incentives = np.random.rand(num_non_zero_incentive)
        non_zero_incentives /= non_zero_incentives.sum()  # Normalize to sum to 1
        zero_incentives = np.zeros(num_miners - num_non_zero_incentive)

        # Combine and shuffle incentives
        incentives = np.concatenate([non_zero_incentives, zero_incentives])
        np.random.shuffle(incentives)
        metagraph.I = incentives

        return metagraph

    @pytest.fixture
    def mock_validator(self, mocker, mock_metagraph, num_active_miners):
        # Initialize validator without calling the constructor
        validator = object.__new__(Validator)

        # Define necessary attributes
        validator.metagraph = mock_metagraph
        validator.hparams = hparams

        validator.comms = mocker.Mock(spec=["peers", "active_peers"])
        validator.comms.peers = None

        # Use the num_active_miners parameter from the fixture
        if num_active_miners > 0:
            validator.comms.active_peers = np.random.choice(
                a=validator.metagraph.uids,
                size=num_active_miners,
                replace=False,
            )
        else:
            validator.comms.active_peers = np.array([])
        tplr.logger.info(f"Created {len(validator.comms.active_peers)} active peers.")

        return validator
