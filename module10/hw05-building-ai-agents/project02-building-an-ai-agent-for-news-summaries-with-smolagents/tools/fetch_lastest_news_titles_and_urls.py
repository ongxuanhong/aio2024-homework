from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "fetch_lastest_news_titles_and_urls"
    description = "This tool extracts the titles and URLs of the latest news articles from a news website's homepage."
    inputs = {'url': {'type': 'string', 'description': "The URL of the news website's homepage."}}
    output_type = "array"

    def forward(self, url: str) -> list[tuple[str, str]]:
        """
        This tool extracts the titles and URLs of the latest news articles from a news website's homepage.

        Args:
            url: The URL of the news website's homepage.

        Returns:
            list[tuple[str, str]]: A list of titles and URLs of the latest news articles.
        """
        import requests
        from bs4 import BeautifulSoup

        article_urls = []
        article_titles = []
        navigation_urls = []

        # Send a GET request to the homepage
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the navigation bar containing news sections
        navigation_bar = soup.find('nav', class_='main-nav')
        if navigation_bar:
            # Extract URLs of news sections
            for header in navigation_bar.ul.find_all('li')[2:7]:
                navigation_urls.append(url + header.a['href'])
            # Iterate over each section URL to extract articles
            for section_url in navigation_urls:
                response = requests.get(section_url)
                section_soup = BeautifulSoup(response.text, 'html.parser')
                # Extract article titles and URLs
                for article in section_soup.find_all('article')[:10]:
                    title_tag = article.find('h3', class_='title-news')
                    if title_tag:
                        title = title_tag.text.strip()
                        article_url = article.find('a')['href']
                        article_titles.append(title)
                        article_urls.append(article_url)

        return list(zip(article_titles, article_urls))