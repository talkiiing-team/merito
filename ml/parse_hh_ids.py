from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm

import time

driver_path = "chromedriver.exe"

options = webdriver.ChromeOptions()
options.add_argument("user-agent=Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0")

driver = webdriver.Chrome()

NUM_PAGES = 3

def get_ids_by_request(search_link_base, num_pages=5):
    """
    Достаем id-шники резюме для дальнейшего парсинга

    :param search_link_base: ссылка для поиска заканчивающаяся на page=
    :param num_pages: кол-во страниц для парсинга
    :return: список id-шников резюме
    """

    all_finded_links = []

    for page_num in tqdm(range(num_pages)):
        link_for_search = search_link_base + str(page_num)

        driver.get(link_for_search)
        # Даем странице время загрузиться
        time.sleep(0.5)

        # Получаем HTML-код страницы
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        elements = soup.find_all('a', rel='nofollow')
        all_finded_links.extend([link['href'] for link in elements])


    # выдергиваем id-шники из ссылок
    all_finded_ids = [i.split("/")[2] for i in all_finded_links]
    all_finded_ids = [i[:i.index("?")] for i in all_finded_ids if "?" in i]
    return all_finded_ids

if __name__ == "__main__":
    all_finded_ids = get_ids_by_request(f"https://hh.ru/resumes/programmist?page=", num_pages=10)
    print(all_finded_ids)

