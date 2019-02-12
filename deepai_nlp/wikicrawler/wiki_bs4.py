from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlencode


class WikiTextCrawler():
    # Crawl text from wiki articles
    def __init__(self,
                 url_search='https://vi.wikipedia.org/w/index.php?',
                 url_root="https://vi.wikipedia.org",
                 ):
        """
        :param url_search:
        :param url_root:
        """
        self.url_search = url_search
        self.url_root = url_root

    def search(self, keyword=None):
        # access top 20 wiki search urls by keyword and return all search result links

        url_list = []
        search = {"search": keyword}
        encoded = self.url_search + urlencode(search)
        html = urlopen(encoded).read()
        soup = BeautifulSoup(html, 'html.parser')
        search_result = soup.find('ul', attrs={'class': 'mw-search-results'})
        if search_result is None:
            url_list.append(encoded)
        else:
            for link in search_result.find_all('a'):
                url_list.append(self.url_root + link.get('href'))
        return url_list

    @staticmethod
    def crawl_text(url):
        # access a wiki url and return text contents
        html = urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        all_p_tag = soup.find_all('p')
        text_data = [tag.get_text() for tag in all_p_tag]
        print('done crawling url: %s' %url)

        return text_data

    def write_text(self, output_file, url, mode='w'):
        # write contents from a url into a .txt file
        with open(output_file, mode) as f:
            text_data = self.crawl_text(url)
            for line in text_data:
                f.write(line)


if __name__ == "__main__":

    wiki_crawler = WikiTextCrawler()
    keywords = ['Học máy']
    for word in keywords:
        results = wiki_crawler.search(word)
    sample_url = results[0]
    wiki_crawler.write_text(output_file='wiki.txt', url=sample_url, mode='w')




