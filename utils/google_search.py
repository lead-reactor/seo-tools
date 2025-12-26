"""
Google Custom Search JSON API integration.

Retrieves Top 10 URLs for a given query (bucket) using Google's Programmable Search Engine.

Endpoint: https://customsearch.googleapis.com/customsearch/v1
Documentation: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
"""

import requests
from dataclasses import dataclass
from typing import Optional
from .google_config import GoogleAPIConfig, ENDPOINTS


@dataclass
class SearchResult:
    """Single search result from Custom Search API."""
    position: int
    url: str
    title: str
    snippet: str
    display_link: str

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "display_link": self.display_link,
        }


@dataclass
class SearchResponse:
    """Response from Custom Search API containing Top N results."""
    query: str
    total_results: int
    results: list[SearchResult]
    search_time: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "total_results": self.total_results,
            "search_time": self.search_time,
            "results": [r.to_dict() for r in self.results],
        }

    def get_urls(self) -> list[str]:
        """Extract just the URLs from results."""
        return [r.url for r in self.results]


class GoogleSearchClient:
    """
    Client for Google Custom Search JSON API.

    Usage:
        config = GoogleAPIConfig.from_env()
        client = GoogleSearchClient(config)
        results = client.search("assurance auto pas cher", num_results=10)
        urls = results.get_urls()
    """

    def __init__(self, config: GoogleAPIConfig):
        self.config = config
        self.endpoint = ENDPOINTS["custom_search"]

        if not config.validate_custom_search():
            raise ValueError(
                "Custom Search API not configured. "
                "Set GOOGLE_CUSTOM_SEARCH_API_KEY and GOOGLE_CUSTOM_SEARCH_ENGINE_ID"
            )

    def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = "fr",
        country: str = "fr",
        safe_search: str = "off",
    ) -> SearchResponse:
        """
        Execute a search query and return Top N results.

        Args:
            query: Search query string (the "bucket" query)
            num_results: Number of results to retrieve (1-10, API limit)
            language: Language code for results (fr, en, etc.)
            country: Country code for localization (fr, us, etc.)
            safe_search: Safe search setting (off, medium, high)

        Returns:
            SearchResponse with ranked results

        Raises:
            requests.RequestException: On API errors
        """
        # API limit is 10 results per request
        num_results = min(num_results, 10)

        params = {
            "key": self.config.custom_search_api_key,
            "cx": self.config.custom_search_engine_id,
            "q": query,
            "num": num_results,
            "lr": f"lang_{language}",  # Language restrict
            "gl": country,  # Geolocation
            "safe": safe_search,
        }

        response = requests.get(self.endpoint, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        return self._parse_response(query, data)

    def search_with_site_filter(
        self,
        query: str,
        include_sites: Optional[list[str]] = None,
        exclude_sites: Optional[list[str]] = None,
        num_results: int = 10,
    ) -> SearchResponse:
        """
        Search with site inclusion/exclusion filters.

        Args:
            query: Base search query
            include_sites: Only search these domains
            exclude_sites: Exclude these domains from results
            num_results: Number of results

        Returns:
            SearchResponse with filtered results
        """
        modified_query = query

        if include_sites:
            site_filter = " OR ".join(f"site:{site}" for site in include_sites)
            modified_query = f"{query} ({site_filter})"

        if exclude_sites:
            for site in exclude_sites:
                modified_query = f"{modified_query} -site:{site}"

        return self.search(modified_query, num_results=num_results)

    def _parse_response(self, query: str, data: dict) -> SearchResponse:
        """Parse API JSON response into structured objects."""
        results = []

        items = data.get("items", [])
        for idx, item in enumerate(items, start=1):
            results.append(SearchResult(
                position=idx,
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                display_link=item.get("displayLink", ""),
            ))

        search_info = data.get("searchInformation", {})

        return SearchResponse(
            query=query,
            total_results=int(search_info.get("totalResults", 0)),
            search_time=float(search_info.get("searchTime", 0)),
            results=results,
        )


def get_top_10_urls(
    query: str,
    config: Optional[GoogleAPIConfig] = None,
    language: str = "fr",
) -> list[str]:
    """
    Convenience function to get Top 10 URLs for a bucket query.

    Args:
        query: The bucket/keyword query
        config: Optional config (uses env if not provided)
        language: Result language

    Returns:
        List of 10 URLs
    """
    if config is None:
        config = GoogleAPIConfig.from_env()

    client = GoogleSearchClient(config)
    response = client.search(query, num_results=10, language=language)

    return response.get_urls()
