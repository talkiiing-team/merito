import os
import requests
from urllib.parse import urlencode
from gensim.models import Word2Vec
import pickle
import zipfile
from tqdm import tqdm
from util_functions import *

# Ссылка на Yandex Disk с моделями
MODEL_DISK_LINK = "https://disk.yandex.ru/d/efUr02dykHDV8g"

# Загружаем обученную модель классификатора
model_clf = pickle.load(open("models/clf.pkl", "rb"))


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
    dist_path = 'models'

    # Сохранение файла
    with open(dist_path, 'wb') as f:
        f.write(response.content)


# Проверяем, существуют ли модели локально, если нет, скачиваем их
if not os.path.exists("models/company_word2vec_russian.model"):
    print("Downloading models from disk...")
    download_ya_disk(MODEL_DISK_LINK)
    print("Downloading ok!")
else:
    print("Models are already downloaded!")


def get_predict_for_resume(df_resume):
    """
    Функция для предсказания класса вакансии по резюме.

    Args:
        df_resume (DataFrame): Датафрейм с резюме. Формат должен быть таким же, как у vprod_train/TRAIN_RES_1.csv.

    Returns:
        job_predict (array): Массив с предсказаниями классов вакансий.
    """

    # Удаляем ненужную колонку
    df_resume.drop("achievements_modified", axis=1, inplace=True)

    # Заполняем пропущенные значения в текстовых признаках
    for text_feature in ["achievements", "company_name", "demands"]:
        df_resume[text_feature].fillna("", inplace=True)

    # Добавляем колонку с длиной текста в "achievements"
    df_resume["achievements_len"] = [len(i) for i in df_resume["achievements"]]

    # Удаляем дубликаты строк
    df_resume.drop_duplicates(inplace=True)

    DROP_PART_DUPLICATES = False
    if DROP_PART_DUPLICATES:
        df_resume["comp_name_demands"] = df_resume["demands"] + df_resume["company_name"]
        df_resume = df_resume.drop_duplicates(subset="comp_name_demands")
        df_resume.drop("comp_name_demands", axis=1, inplace=True)

    # Добавляем количество строк для каждого резюме по id_cv
    df_resume["len_group"] = df_resume.groupby("id_cv").agg({"company_name": "count"})

    # Работа с названием компании с помощью Word2Vec
    company_names = [i["company_name"] for _, i in df_resume.iterrows()]
    sentences = preprocess_names(company_names)
    RETRAIN_MODEL = False
    if RETRAIN_MODEL:
        # Обучаем новую модель Word2Vec
        model_comp_name = Word2Vec(sentences, vector_size=64, window=4, min_count=1, sg=1)
        model_comp_name.save("models/company_word2vec_russian.model")
    else:
        # Загружаем готовую модель
        model_comp_name = Word2Vec.load("models/company_word2vec_russian.model")

    # Преобразуем названия компаний в векторы
    comp_names_without_job = preprocess_names(df_resume["company_name"])
    emb_comp_name_df = get_emb_by_modele(model_comp_name, comp_names_without_job, column_prefix="comp_name_emb")
    df_resume = pd.concat([df_resume.reset_index(drop=True), emb_comp_name_df.reset_index(drop=True)], axis=1)

    # Работа с "demands" с помощью Word2Vec
    company_names = [i["demands"] for _, i in df_resume.iterrows()]
    sentences = preprocess_names(company_names)
    RETRAIN_MODEL = False
    if RETRAIN_MODEL:
        model_demand = Word2Vec(sentences, vector_size=100, window=8, min_count=1, sg=1)
        model_demand.save("models/demand_word2vec_russian.model")
    else:
        model_demand = Word2Vec.load("models/demand_word2vec_russian.model")

    # Преобразуем требования к вакансиям в векторы
    demand_without_job = preprocess_names(df_resume["demands"])
    emb_demand_df = get_emb_by_modele(model_demand, demand_without_job, column_prefix="demand_emb")
    df_resume = pd.concat([df_resume.reset_index(drop=True), emb_demand_df.reset_index(drop=True)], axis=1)

    # Работа с названиями компаний с помощью TF-IDF
    sentences = preprocess_names(df_resume["company_name"])
    RETRAIN_MODEL = False
    column_prefix = "comp_name_tfidf"
    if RETRAIN_MODEL:
        comp_name_vectorizer, comp_name_tfidf = get_emb_by_tfidf([" ".join(sent) for sent in sentences],
                                                                 column_prefix=column_prefix)
    else:
        comp_name_vectorizer = pickle.load(open("models/comp_name_vectorizer.pkl", "rb"))
        comp_name_tfidf = comp_name_vectorizer.transform([" ".join(sent) for sent in sentences])
        comp_name_tfidf = pd.DataFrame(comp_name_tfidf.toarray(), columns=[f"{column_prefix}_{word}" for word in
                                                                           comp_name_vectorizer.get_feature_names_out()])

    df_resume = pd.concat([df_resume.reset_index(drop=True), comp_name_tfidf.reset_index(drop=True)], axis=1)

    # Работа с требованиями вакансий с помощью TF-IDF
    sentences = preprocess_names(df_resume["demands"])
    RETRAIN_MODEL = False
    column_prefix = "demnds_tfidf"
    if RETRAIN_MODEL:
        demands_vectorizer, demands_tfidf = get_emb_by_tfidf([" ".join(sent) for sent in sentences],
                                                             column_prefix=column_prefix)
    else:
        demands_vectorizer = pickle.load(open("models/demands_vectorizer.pkl", "rb"))
        demands_tfidf = demands_vectorizer.transform([" ".join(sent) for sent in sentences])
        demands_tfidf = pd.DataFrame(demands_tfidf.toarray(), columns=[f"{column_prefix}_{word}" for word in
                                                                       demands_vectorizer.get_feature_names_out()])

    df_resume = pd.concat([df_resume.reset_index(drop=True), demands_tfidf.reset_index(drop=True)], axis=1)

    # Работа с достижениями (achievements) с помощью TF-IDF
    sentences = preprocess_names(df_resume["achievements"])
    RETRAIN_MODEL = False
    column_prefix = "achiv_tfidf"
    if RETRAIN_MODEL:
        achiv_vectorizer, achiv_tfidf = get_emb_by_tfidf([" ".join(sent) for sent in sentences],
                                                         column_prefix=column_prefix)
    else:
        achiv_vectorizer = pickle.load(open("models/achiv_vectorizer.pkl", "rb"))
        achiv_tfidf = achiv_vectorizer.transform([" ".join(sent) for sent in sentences])
        achiv_tfidf = pd.DataFrame(achiv_tfidf.toarray(), columns=[f"{column_prefix}_{word}" for word in
                                                                   demands_vectorizer.get_feature_names_out()])

    df_resume = pd.concat([df_resume.reset_index(drop=True), achiv_tfidf.reset_index(drop=True)], axis=1)

    columns_to_drop = ["achievements", "company_name", "demands", "id_cv", ]
    df_resume = df_resume.drop(columns_to_drop, axis=1)

    # Предсказание класса вакансии
    job_predict = model_clf.predict(df_resume)

    return job_predict

