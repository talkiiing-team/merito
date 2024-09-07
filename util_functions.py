from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from tqdm import tqdm
import numpy as np
import re
import pandas as pd

# Инициализация анализатора pymorphy2 для лемматизации
morph = pymorphy2.MorphAnalyzer()


# Функция для лемматизации слова
def lemmatize_word(word):
    return morph.parse(word)[0].normal_form


# Функция для предобработки данных (разбиение на слова, лемматизация, удаление пунктуации)
def preprocess_names(company_names):
    processed_names = []
    for name in tqdm(company_names):
        # Приводим название к нижнему регистру
        name = name.lower()
        # Удаляем всю пунктуацию, оставляем только слова
        name = re.sub(r'[^\w\s]', '', name)
        # Разбиваем на отдельные слова
        words = name.split()

        # Лемматизируем каждое слово
        # lemmatized_words = [lemmatize_word(word) for word in words]

        processed_names.append(words)
    return processed_names


def get_emb_by_modele(model, comp_names_without_job, column_prefix, vector_size=100):
    all_tokens = set(model.wv.index_to_key)
    word_vectors_dict = {word: model.wv[word] for word in model.wv.index_to_key}

    all_embds = []

    for sent in tqdm(comp_names_without_job):
        all_emb = [word_vectors_dict[word] for word in sent if word in all_tokens]

        if len(all_emb) == 0:
            emb = np.zeros(vector_size)
        else:
            emb = np.mean(all_emb, axis=0)

        all_embds.append(emb)

    emb_comp_name_df = pd.DataFrame(np.array(all_embds), columns=[f"{column_prefix}_{i}" for i in range(vector_size)])
    return emb_comp_name_df


def get_emb_by_tfidf(processed_sentences, column_prefix, max_features=100, ):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_sentences)

    # Преобразуем разреженную матрицу TF-IDF в плотный формат и создаем DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{column_prefix}_{word}" for word in tfidf_vectorizer.get_feature_names_out()])
    return tfidf_vectorizer, tfidf_df
