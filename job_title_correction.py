import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import os
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from tqdm import tqdm
from gensim.models import Word2Vec
import pickle
import pymorphy2
from autocorrect import Speller
from pathlib import Path
import language_tool_python

my_path = Path(__file__).parent
MODEL_DISK_LINK = "https://disk.yandex.ru/d/x4X4lYMdc-ZWqA"


def download_ya_disk(public_key):
    """
    Функция для скачивания файлов с Yandex Disk по публичной ссылке.

    Args:
        public_key (str): Публичный ключ (ссылка на Yandex Disk).

    Returns:
        None. Скачанный файл сохраняется на диск.
    """
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    final_url = base_url + urlencode(dict(public_key=public_key))

    # Получение ссылки для скачивания
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Скачивание файла
    response = requests.get(download_url)
    dist_path = os.path.join(my_path, 'models/sphere_clf.pickle')

    # Сохранение файла
    with open(dist_path, 'wb') as f:
        f.write(response.content)


# Проверяем, существуют ли модели локально, если нет, скачиваем их
if not os.path.exists(os.path.join(my_path, "models/sphere_clf.pickle")):
    print("Downloading models from disk...")
    download_ya_disk("https://disk.yandex.ru/d/efUr02dykHDV8g")
    print("Downloading ok!")
else:
    print("Models are already downloaded!")


def download_ya_disk(public_key):
    """
    Функция для скачивания файлов с Yandex Disk по публичной ссылке.

    Args:
        public_key (str): Публичный ключ (ссылка на Yandex Disk).

    Returns:
        None. Скачанный файл сохраняется на диск.
    """
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    final_url = base_url + urlencode(dict(public_key=public_key))

    # Получение ссылки для скачивания
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Скачивание файла
    response = requests.get(download_url)
    dist_path = os.path.join(my_path, 'models.zip')

    # Сохранение файла
    with open(dist_path, 'wb') as f:
        f.write(response.content)

    fname = os.path.join(my_path, 'models.zip')
    path = os.path.join(my_path, 'models/')

    with zipfile.ZipFile(fname, 'r') as zf:
        for entry in tqdm(zf.infolist(), desc='Extracting '):
            try:
                zf.extract(entry, path)
            except zipfile.error as e:
                pass
    os.remove(os.path.join(my_path, "models.zip"))



# Проверяем, существуют ли модели локально, если нет, скачиваем их
if not os.path.exists(os.path.join(my_path, "models/demand_word2vec_russian.model")):
    print("Downloading models from disk...")
    download_ya_disk(MODEL_DISK_LINK)
    print("Downloading ok!")
else:
    print("Models are already downloaded!")



russian_stopwords = pickle.load(open(os.path.join(my_path, "utils/stopwords.pkl"), "rb"))
morph = pymorphy2.MorphAnalyzer()
tool = language_tool_python.LanguageTool('ru-RU')
spheres_clf = pickle.load(open(os.path.join(my_path, "models/sphere_clf.pickle"), "rb"))


def get_emb_by_modele(model, data):
	all_tokens = set(model.wv.index_to_key)
	word_vectors_dict = {word: model.wv[word] for word in model.wv.index_to_key}

	all_embds = []

	for sent in tqdm(data):
		all_emb = [word_vectors_dict[word] for word in sent.split() if word in all_tokens]

		if len(all_emb) == 0:
			emb = np.zeros(100)
		else:
			emb = np.mean(all_emb, axis=0)

		all_embds.append(emb)
	emb_comp_name_df = pd.DataFrame(np.array(all_embds), columns=[f"emb_{i}" for i in range(100)])
	return emb_comp_name_df


jobs_dict = pickle.load(open(os.path.join(my_path, "utils/correctors_database.pickle"), "rb"))
jobs_dict_keys = set(jobs_dict.keys())
model_demand = Word2Vec.load(os.path.join(my_path, "models/demand_word2vec_russian.model"))


def text_correction(text, predict_spheres=True):
	global model_demand, spheres_clf
	text = tool.correct(text)
	if not predict_spheres:
		return text, [sphere, subsphere]
	
	text = text.replace(",", " ").strip().lower()
	text = re.sub(' +', ' ', text)
	text = "".join([x for x in text if not x.isdigit()])
	
	text = " ".join([morph.parse(word)[0].normal_form for word in text.split() if word not in russian_stopwords])
	embed = get_emb_by_modele(model_demand, [text])
	sphere = spheres_clf.predict(embed)[0]
	for word in text.split():
		if word in jobs_dict_keys:
			subsphere = jobs_dict[word]
			break
	else:
		subsphere = None
	
	
	return text, [sphere, subsphere]