from typing import Any
from unittest.mock import patch

from src.tools.search import SEARCH_TOOL_DEFINITION, search

# --- tool definition ---


def test_tool_definition_has_name():
    assert SEARCH_TOOL_DEFINITION["name"] == "search"


def test_tool_definition_has_description():
    assert "description" in SEARCH_TOOL_DEFINITION
    assert len(SEARCH_TOOL_DEFINITION["description"]) > 0


def test_tool_definition_has_input_schema():
    schema: dict[str, Any] = SEARCH_TOOL_DEFINITION["input_schema"]  # type: ignore[assignment]
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "query" in schema["required"]


# --- search results ---


@patch("src.tools.search.DDGS")
def test_search_returns_list(mock_ddgs_class):
    mock_ddgs_class.return_value.text.return_value = []
    results = search("test query")
    assert isinstance(results, list)


@patch("src.tools.search.DDGS")
def test_search_normalizes_result_keys(mock_ddgs_class):
    mock_ddgs_class.return_value.text.return_value = [
        {"title": "Test Title", "href": "https://example.com", "body": "Test snippet"},
    ]
    results = search("test query")
    assert results[0]["title"] == "Test Title"
    assert results[0]["url"] == "https://example.com"
    assert results[0]["snippet"] == "Test snippet"


@patch("src.tools.search.DDGS")
def test_search_respects_max_results(mock_ddgs_class):
    mock_instance = mock_ddgs_class.return_value
    mock_instance.text.return_value = []
    search("test query", max_results=3)
    mock_instance.text.assert_called_once_with("test query", max_results=3)


@patch("src.tools.search.DDGS")
def test_search_default_max_results(mock_ddgs_class):
    mock_instance = mock_ddgs_class.return_value
    mock_instance.text.return_value = []
    search("test query")
    mock_instance.text.assert_called_once_with("test query", max_results=5)


@patch("src.tools.search.DDGS")
def test_search_returns_empty_list_on_no_results(mock_ddgs_class):
    mock_ddgs_class.return_value.text.return_value = []
    results = search("obscure query with no results")
    assert results == []


@patch("src.tools.search.DDGS")
def test_search_returns_multiple_results(mock_ddgs_class):
    mock_ddgs_class.return_value.text.return_value = [
        {"title": "Result 1", "href": "https://one.com", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://two.com", "body": "Snippet 2"},
    ]
    results = search("test")
    assert len(results) == 2
    assert results[1]["url"] == "https://two.com"
