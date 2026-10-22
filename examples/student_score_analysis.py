"""Analyze exam scores using pandas merge and NumPy aggregation.

This is a small reproducible example based on a hands-on learning exercise.
"""

import numpy as np
import pandas as pd


STUDENTS = {
    "student_id": [101, 102, 103, 104, 105],
    "department": ["AI", "Systems", "AI", "Systems", "AI"],
}

EXAMS = {
    "student_id": [101, 102, 103, 104, 101, 103],
    "math": [80, 70, 100, 90, 80, None],
    "english": [90, 60, 100, 90, 90, 80],
}


def analyze_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join, clean, then summarize scores without modifying the source tables.

    The left join retains all students first. Missing exam scores and exact
    duplicate rows are removed only after joining.
    """
    student_df = pd.DataFrame(STUDENTS)
    exam_df = pd.DataFrame(EXAMS)
    joined = pd.merge(student_df, exam_df, on="student_id", how="left")
    cleaned = joined.dropna(subset=["math", "english"]).drop_duplicates()

    department_means = cleaned.groupby("department")[["math", "english"]].mean()
    scores = cleaned[["math", "english"]].to_numpy()
    student_means = np.mean(scores, axis=1)
    summary = {
        "shape": scores.shape,
        "subject_means": np.mean(scores, axis=0),
        "student_means": student_means,
        "students_at_least_80": int(len(scores[student_means >= 80])),
    }
    return cleaned, department_means, summary


if __name__ == "__main__":
    cleaned, department_means, summary = analyze_scores()
    print("Cleaned records:")
    print(cleaned.to_string(index=False))
    print("\nMean scores by department:")
    print(department_means.to_string())
    print("\nNumPy score shape:", summary["shape"])
    print("Mean by subject:", summary["subject_means"])
    print("Mean by student:", summary["student_means"])
    print("Students with mean score >= 80:", summary["students_at_least_80"])
