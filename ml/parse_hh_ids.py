from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm

import pickle
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
        time.sleep(1)

        for i in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.2)

        # Получаем HTML-код страницы
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        elements = soup.find_all('a', rel='nofollow')
        all_finded_links.extend([link['href'] for link in elements])
        print(len(all_finded_links))

    # выдергиваем id-шники из ссылок
    all_finded_ids = [i.split("/")[2] for i in all_finded_links]
    all_finded_ids = [i[:i.index("?")] for i in all_finded_ids if "?" in i]
    return all_finded_ids

if __name__ == "__main__":
    BASE_LINK = "https://hh.ru/search/resume?text=%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B8%D1%81%D1%82&logic=normal&pos=full_text&exp_period=all_time&exp_company_size=any&filter_exp_period=all_time&area=113&relocation=living_or_relocation&age_from=&age_to=&gender=unknown&salary_from=&salary_to=&currency_code=RUR&order_by=relevance&search_period=0&items_on_page=100&hhtmFrom=resume_search_form&page="
    all_finded_ids = get_ids_by_request(BASE_LINK, num_pages=100)
    print("Len finded links list: ", len(all_finded_ids))

    with open("parsed_links.pkl", "wb") as f:
        pickle.dump(all_finded_ids, f)



