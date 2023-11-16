import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def fetch_all_urls(base_url, max_depth):
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)

        # Normalize URL by removing query parameters and fragments
        parsed = urlparse(current_url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        if normalized_url not in visited_urls:
            print(f"Visiting {normalized_url} at depth {current_depth}")
            visited_urls.add(normalized_url)
            try:
                response = requests.get(normalized_url)
                if response.status_code == 200:
                    if current_depth < max_depth:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            new_url = urljoin(normalized_url, link['href'])
                            parsed_new_url = urlparse(new_url)
                            normalized_new_url = f"{parsed_new_url.scheme}://{parsed_new_url.netloc}{parsed_new_url.path}"
                            if normalized_new_url.startswith(base_url) and normalized_new_url not in visited_urls:
                                urls_to_visit.append((normalized_new_url, current_depth + 1))
            except Exception as e:
                print(f"An error occurred: {e}")

            # write the url to 'url_list.txt'
            with open('url_list.txt', 'a') as f:
                f.write(f"{normalized_url}\n")

    return visited_urls

if __name__ == "__main__":
    base_url = "https://www.giallozafferano.it/"
    max_depth = 3  # Define the maximum depth to which you want to scrape URLs
    urls = fetch_all_urls(base_url, max_depth)
