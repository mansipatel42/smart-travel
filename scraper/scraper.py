import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    handlers=[
        logging.FileHandler("scraper.log"),  # Save logs to file
        logging.StreamHandler()  # Print logs to console
    ],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class TravelScraper:
    def __init__(self, config):
        self.base_url = config.get("base_url")
        self.start_urls = config.get("start_urls")
        self.content_selectors = config['selectors'].get("content")
        self.title_selector = config['selectors'].get("title", "h1")
        self.link_selectors = config['selectors'].get("link", "a")
        self.data_file = "./data/scraped_data"
        self.visited_urls = set()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        logging.debug(f"Initialized scraper for {self.base_url}")

    def http_get(self, url):
        try:
            res = self.session.get(url, timeout=10)
            res.raise_for_status()
            return res
        except Exception as e:
            logging.error(f"Error while GET request for {url} : {e}")
            return None

    def get_soup(self, url):
        """Fetch and parse a webpage."""

        response = self.http_get(url)
        if response is None:
            return None

        try:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup
        except Exception as e:
            logging.error(f"Error while getting soup for {url} : {e}")

    def get_internal_links(self, urls):
        """Finds all internal links on a given page."""
        links = set()
        for url in urls:
            try:
                if url.startswith("/") and not url.startswith("//"):
                    url = f"{self.base_url}{url}"

                res = self.http_get(url)

                if not res or res.status_code != 200:
                    logging.error(f"Failed to fetch links from {url}")
                    continue

                links.add(url)

                soup = BeautifulSoup(res.text, "html.parser")
                if not soup:
                    return []

                for a_tag in soup.select(self.link_selectors):
                    link = a_tag["href"]
                    if link.startswith("/") and not link.startswith("//"):
                        link = self.base_url + link.split("?")[0]  # Remove tracking params
                    elif link.startswith("#"):
                        continue

                    if link not in self.visited_urls:
                        links.add(link)
            except Exception as e:
                logging.error(f"Error while fetching internal link : {e}")

        return list(links)

    def scrape_content(self, page_url):
        """Extracts main content from a page."""
        soup = self.get_soup(page_url)
        if not soup:
            return None

        # Extract title
        title = "N/A"
        if soup.select_one(self.title_selector):
            title = soup.select_one(self.title_selector).get_text(strip=True)

        if len(title.split()) < 2:
            logging.info(f"Ignoring data with title of less than 2 words {title}")
            return None

        content = []
        for selector in self.content_selectors:
            for element in soup.select(selector):
                text = element.get_text(strip=True)
                if text:
                    content.append(text)

        content_txt = " ".join(content)
        if len(content_txt.split()) < 50:
            logging.info(f"Ignoring content less that 50 words")
            return None

        return {"input": title, "instruction": title, "output": " ".join(content)}

    def append_to_file(self, data, format):
        if format == "all":
            self.append_to_csv(data)
            self.append_to_jsonl(data)
        elif format == "jsonl":
            self.append_to_jsonl(data)
        elif format == "csv":
            self.append_to_csv(data)
        elif format == "txt":
            self.append_to_txt(data)
        else:
            logging.error(f"Unknown format for appending data to file")

    def append_to_csv(self, data):
        """Appends data to a single CSV file."""
        file_name = self.data_file + ".csv"
        df = pd.DataFrame([data])
        df.to_csv(file_name, mode='a', index=False, header=not pd.io.common.file_exists(file_name), encoding="utf-8")
        logging.debug(f"Appended CSV data")

    def append_to_jsonl(self, data):
        """Appends data to a JSONL file (line-by-line JSON format)."""
        file_name = self.data_file + ".jsonl"
        with open(file_name, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")  # Newline after each JSON object
        logging.debug(f"Appended JSONL data")

    def append_to_txt(self, data):
        file_name = self.data_file + ".txt"
        with open(file_name, "a", encoding="utf-8") as f:
            f.write("<s>[INST] " + data["input"] + " [/INST] " + data["output"] + " </s>\n")

    def run(self):
        """Main loop to scrape the website."""
        # to_visit = set(self.start_urls)
        logging.info(f"Running scraper for {self.base_url}")
        to_visit = set(self.get_internal_links(self.start_urls))

        count = 0

        while to_visit:
            page_url = to_visit.pop()
            self.visited_urls.add(page_url)

            logging.info(f"Scraping: {page_url}")
            scraped_data = self.scrape_content(page_url)
            if scraped_data:
                self.append_to_file(scraped_data, "jsonl")

            """
            TODO:
            Enable below line of code for nested link exploration.
            However, be careful before enabling. This can keep on going because of
            nested link scraping. This needs to be wrapped under max nesting level
            configuration

            count = count + 1
            if count < 5:
                # Discover new links from the page
                new_links = self.get_internal_links([page_url])
                to_visit.update(set(new_links) - self.visited_urls)
            else:
                logging.info(f"Stopping nested link discovery for {self.base_url}")
            """
