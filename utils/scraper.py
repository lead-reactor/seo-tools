"""
Module for web scraping with SEO-focused content extraction.
User-Agent: LeadFactorBot
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from urllib.parse import urlparse


class WebScraper:
    """Web scraper optimized for SEO semantic analysis."""

    USER_AGENT = "LeadFactorBot"
    TIMEOUT = 30

    # Elements to exclude from content extraction
    EXCLUDED_TAGS = ['nav', 'header', 'footer', 'script', 'style', 'noscript', 'iframe']

    # SEO-relevant elements to extract
    SEO_ELEMENTS = {
        'headings': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        'content': ['p', 'li', 'td', 'th', 'span', 'div'],
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def fetch_url(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse URL content.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None if error
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")

            response = self.session.get(url, timeout=self.TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'

            return BeautifulSoup(response.text, 'html.parser')

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for {url}: {e}")
            return None

    def _remove_excluded_elements(self, soup: BeautifulSoup) -> None:
        """Remove excluded HTML elements in-place."""
        for tag in self.EXCLUDED_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

    def extract_seo_content(self, url: str) -> Dict[str, any]:
        """
        Extract SEO-relevant content from URL.

        Args:
            url: URL to analyze

        Returns:
            Dictionary with extracted content:
            - url: The URL
            - title: Page title
            - meta_description: Meta description
            - h1: List of H1 texts
            - h2: List of H2 texts
            - h3-h6: Lists of H3-H6 texts
            - body_text: Main body text (cleaned)
            - images_alt: List of image alt texts
            - all_text: Combined text for analysis
        """
        soup = self.fetch_url(url)

        if not soup:
            return {
                'url': url,
                'error': 'Failed to fetch URL',
                'all_text': ''
            }

        # Remove excluded elements
        self._remove_excluded_elements(soup)

        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ''

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc.get('content', '') if meta_desc else ''

        # Extract headings
        headings = {}
        for i in range(1, 7):
            tag = f'h{i}'
            headings[tag] = [h.get_text(strip=True) for h in soup.find_all(tag)]

        # Extract image alt texts
        images_alt = [
            img.get('alt', '').strip()
            for img in soup.find_all('img')
            if img.get('alt', '').strip()
        ]

        # Extract main body text
        body = soup.find('body')
        if body:
            # Get text from paragraphs and list items primarily
            body_text = ' '.join([
                elem.get_text(strip=True)
                for elem in body.find_all(['p', 'li'])
            ])
        else:
            body_text = ''

        # Combine all text for semantic analysis
        all_text_parts = [
            title_text,
            meta_description,
            ' '.join(headings.get('h1', [])),
            ' '.join(headings.get('h2', [])),
            ' '.join(headings.get('h3', [])),
            body_text,
            ' '.join(images_alt)
        ]

        all_text = ' '.join(filter(None, all_text_parts))

        return {
            'url': url,
            'title': title_text,
            'meta_description': meta_description,
            'h1': headings.get('h1', []),
            'h2': headings.get('h2', []),
            'h3': headings.get('h3', []),
            'h4': headings.get('h4', []),
            'h5': headings.get('h5', []),
            'h6': headings.get('h6', []),
            'body_text': body_text,
            'images_alt': images_alt,
            'all_text': all_text,
            'text_length': len(all_text)
        }

    def extract_multiple_urls(self, urls: list) -> Dict[str, Dict]:
        """
        Extract content from multiple URLs.

        Args:
            urls: List of URLs to analyze

        Returns:
            Dictionary mapping URL to extracted content
        """
        results = {}
        for url in urls:
            results[url] = self.extract_seo_content(url)
        return results
