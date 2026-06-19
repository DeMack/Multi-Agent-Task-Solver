import pytest

from src.tools.executor import execute_python


@pytest.mark.integration
def test_matplotlib_produces_png(tmp_path):
    code = """
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("Integration Test Chart")
plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "chart.png"))
plt.close()
"""
    result = execute_python(code, tmp_path)
    assert result.exit_code == 0
    assert result.timed_out is False
    chart = tmp_path / "chart.png"
    assert chart.exists()
    assert chart.stat().st_size > 0


@pytest.mark.integration
def test_pandas_dataframe_runs(tmp_path):
    code = """
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(df.describe().to_string())
"""
    result = execute_python(code, tmp_path)
    assert result.exit_code == 0
    assert "mean" in result.stdout


@pytest.mark.integration
def test_matplotlib_with_pandas_produces_png(tmp_path):
    code = """
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3"], "revenue": [100, 150, 130]})
fig, ax = plt.subplots()
ax.bar(df["quarter"], df["revenue"])
ax.set_title("Revenue by Quarter")
plt.savefig(os.path.join(os.environ["OUTPUT_DIR"], "revenue.png"))
plt.close()
print("Chart saved.")
"""
    result = execute_python(code, tmp_path)
    assert result.exit_code == 0
    assert any("revenue.png" in p for p in result.artifact_paths)
    assert (tmp_path / "revenue.png").stat().st_size > 0
