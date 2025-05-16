from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "extract_news_article_content"
    description = "This tool extracts the content of a news article from its URL."
    inputs = {"url":{"type":"string","description":"The URL of the news article."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        """
        This tool extracts the content of a news article from its URL.

        Args:
            url (str): The URL of the news article.

        Returns:
            str: The content of the news article.
        """
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        content = ""
        for paragraph in soup.find_all('p'):
            content += paragraph.get_text().strip() + " "
        return content