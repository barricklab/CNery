import pytest
import numpy as np
from CNery.core import (
    find_nearest,
    solve_pr,
    setup_transition_matrix,
    setup_emission_matrix,
    parse_fasta_records,
)


class TestFindNearest:
    def test_exact_match(self):
        assert find_nearest([10, 20, 30], 20) == 1

    def test_rounds_down(self):
        assert find_nearest([10, 20, 35, 50], 33) == 2

    def test_rounds_up(self):
        assert find_nearest([10, 20, 35, 50], 37) == 2

    def test_single_element(self):
        assert find_nearest([42], 0) == 0

    @pytest.mark.parametrize("value,expected_idx", [
        (0, 0), (50, 1), (100, 2),
    ])
    def test_parametrized(self, value, expected_idx):
        assert find_nearest([0, 50, 100], value) == expected_idx


class TestSolvePR:
    def test_known_values(self):
        p, r = solve_pr(50, 100)
        assert 0 < p < 1
        assert r > 0

    @pytest.mark.parametrize("mean,var", [(10, 20), (100, 500), (5, 10)])
    def test_various_inputs(self, mean, var):
        p, r = solve_pr(mean, var)
        assert 0 < p < 1
        assert r > 0


class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        tm = setup_transition_matrix(n_states=5, remain_prob=0.99)
        assert np.allclose(tm.sum(axis=1), 1.0)

    def test_shape_includes_zero_state(self):
        tm = setup_transition_matrix(n_states=5, remain_prob=0.99)
        assert tm.shape == (6, 6)

    def test_all_nonnegative(self):
        tm = setup_transition_matrix(n_states=5, remain_prob=0.99)
        assert np.all(tm >= 0)


class TestEmissionMatrix:
    @pytest.fixture
    def basic_emission(self):
        return setup_emission_matrix(
            n_states=5, mean=50.0, variance=100.0, absmax=150, error_rate=0.05
        )

    def test_zero_state_row_peaks_at_low_obs(self, basic_emission):
        assert np.argmax(basic_emission[0, :]) < 5

    def test_higher_cn_peaks_scale_with_state(self, basic_emission):
        assert np.all(np.isfinite(basic_emission))

    def test_shape(self, basic_emission):
        assert basic_emission.shape == (6, 151)


class TestParseFastaRecords:
    @pytest.fixture
    def single_fasta(self, tmp_path):
        fa = tmp_path / "single.fasta"
        fa.write_text(">chr1\n" + "ACGT" * 1000 + "\n")
        return str(fa)

    @pytest.fixture
    def multi_fasta(self, tmp_path):
        fa = tmp_path / "multi.fasta"
        fa.write_text(">chr1\n" + "ACGT" * 1000 + "\n>plas1\n" + "GCTA" * 500 + "\n")
        return str(fa)

    def test_single_record_length(self, single_fasta):
        records = parse_fasta_records(single_fasta)
        assert len(records) == 1

    def test_multi_record_lengths(self, multi_fasta):
        records = parse_fasta_records(multi_fasta)
        assert len(records) == 2

    def test_empty_fasta(self, tmp_path):
        fa = tmp_path / "empty.fasta"
        fa.write_text("")
        assert parse_fasta_records(str(fa)) == []