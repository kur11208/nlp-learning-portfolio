"""Regression tests for a pandas + NumPy aggregation exercise."""

import numpy as np
import pytest

from examples.student_score_analysis import analyze_scores


def test_left_merge_cleaning_removes_missing_and_duplicates():
    cleaned, _, summary = analyze_scores()
    assert cleaned["student_id"].tolist() == [101, 102, 103, 104]
    assert summary["shape"] == (4, 2)


def test_department_means():
    _, department_means, _ = analyze_scores()
    assert department_means.loc["AI", "math"] == pytest.approx(90)
    assert department_means.loc["AI", "english"] == pytest.approx(95)
    assert department_means.loc["Systems", "math"] == pytest.approx(80)
    assert department_means.loc["Systems", "english"] == pytest.approx(75)


def test_numpy_axis_and_threshold():
    _, _, summary = analyze_scores()
    np.testing.assert_allclose(summary["subject_means"], [85, 85])
    np.testing.assert_allclose(summary["student_means"], [85, 65, 100, 90])
    assert summary["students_at_least_80"] == 3
