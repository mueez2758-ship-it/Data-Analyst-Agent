
import pandas as pd

from app.insights import generate_insights


def test_generate_insights_returns_expected_structure():
    """The insights report should contain its three main fields."""

    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Chair"],
        "Sales": [1200, 800, 250],
        "Profit": [200, 150, 40],
    })

    result = generate_insights(df)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "findings" in result
    assert "recommendations" in result


def test_generate_insights_reports_dataset_dimensions():
    """The report should describe the uploaded dataset dimensions."""

    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Chair"],
        "Sales": [1200, 800, 250],
    })

    result = generate_insights(df)

    all_findings = " ".join(result["findings"])

    assert "3 rows" in all_findings
    assert "2 columns" in all_findings


def test_generate_insights_handles_missing_values():
    """Missing values should be identified in the findings."""

    df = pd.DataFrame({
        "Product": ["Laptop", None, "Chair"],
        "Sales": [1200, 800, 250],
    })

    result = generate_insights(df)

    all_findings = " ".join(result["findings"]).lower()

    assert "missing" in all_findings


def test_generate_insights_handles_duplicate_rows():
    """Duplicate rows should be identified in the findings."""

    df = pd.DataFrame({
        "Product": ["Laptop", "Laptop", "Chair"],
        "Sales": [1200, 1200, 250],
    })

    result = generate_insights(df)

    all_findings = " ".join(result["findings"]).lower()

    assert "duplicate" in all_findings


def test_generate_insights_returns_list_fields():
    """Findings and recommendations should be lists."""

    df = pd.DataFrame({
        "Category": ["Electronics", "Furniture", "Electronics"],
        "Sales": [100, 200, 300],
    })

    result = generate_insights(df)

    assert isinstance(result["findings"], list)
    assert isinstance(result["recommendations"], list)
