import json
from scraper import TravelScraper

def load_config(filename="config.json"):
    """Load configuration from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)

def main():
    """Load configurations and run scrapers."""
    config = load_config()

    for site_name, site_config in config.items():
        scraper = TravelScraper(site_config)
        scraper.run()

if __name__ == "__main__":
    main()