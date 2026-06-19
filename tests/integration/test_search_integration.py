import pytest

from src.tools.search import search


@pytest.mark.integration
def test_search_returns_real_results():
    results = search("Python programming language", max_results=3)
    assert len(results) > 0


@pytest.mark.integration
def test_search_result_structure():
    results = search("Python programming language", max_results=3)
    for r in results:
        assert "title" in r and r["title"]
        assert "url" in r and r["url"].startswith("http")
        assert "snippet" in r and r["snippet"]


@pytest.mark.integration
def test_search_respects_max_results_against_real_api():
    results = search("Python programming language", max_results=2)
    assert len(results) <= 2
