import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
from collections import Counter
import pickle
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import os
from pathlib import Path
my_path = Path(__file__).parent

import pymorphy2
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = pickle.load(open(os.path.join(my_path, "utils/stopwords.pkl"), "rb"))
PREPROC_CONFIG = pickle.load(open(os.path.join(my_path, "utils/SALARY_PREPROC_CONFIG.pickle"), "rb"))
reg = pickle.load(open(os.path.join(my_path, "models/salary_regressor.pickle"), "rb"))


def list_col_preproc(text):
	if pd.isna(text):
		return []
	if isinstance(text, list):
		return text
	if "[" in text:
		text = eval(text)
	else:
		text = text.split(",")
	lst = [x.strip() for x in text]
	return lst


def base_preproc(text):
	text = text.replace(",", " ").strip().lower()
	text = re.sub(' +', ' ', text)
	text = "".join([x for x in text if not x.isdigit()])
	
	text = " ".join(
		[morph.parse(word)[0].normal_form for word in text.split() if word not in russian_stopwords])
	return text



COLS_FOR_DROP = ['foreign_workers_capability', 'oknpo_code', 'regionNameTerm',
	   'retraining_condition', 'contactList', 'company_name', "id", "change_time", "code_external_system",
		"company_code", "contact_person", "data_ids", "date_create", "date_modify", "deleted",
		"original_source_type", "publication_period", "published_date", 
		"vacancy_address_code", "vacancy_address", "visibility", "company", "company_inn", 
		"industryBranchName", "state_region_code", "vacancy_address_house", "metro_ids",
		"vacancy_address_additional_info", "is_moderated", "languageKnowledge"]

cat_features = ["academic_degree", "accommodation_capability", "busy_type", "career_perspective",
				"code_professional_sphere", "contact_source", "education", "is_mobility_program",
				"is_uzbekistan_recruitment", "is_quoted", "need_medcard", "okso_code", 
				"accommodation_type", "additional_requirements", "bonus_type", "measure_type",
				"regionName", "retraining_capability", "retraining_grant", "schedule_type",
				"status", "source_type", "transport_compensation", "professionalSphereName",
				"federalDistrictCode", "company_business_size"
			   ]

text_features=  ["additional_requirements", "education_speciality", "other_vacancy_benefit",
				 "position_requirements", "position_responsibilities", "required_certificates",
				"vacancy_name", "full_company_name", "hardSkills", "softSkills"]

cols_for_aggregation = ["code_professional_sphere", "regionName", "busy_type", "education"]


def remove_html_tags(text):
	if pd.isna(text):
		return None
	soup = BeautifulSoup(text, "html.parser")
	return soup.get_text()


def predict_salary_by_vacancy(df):
	global PREPROC_CONFIG, COLS_FOR_DROP, cat_features, cols_for_aggregation
	df = deepcopy(df)

	df.drop(COLS_FOR_DROP, axis=1, inplace=True)
	if "salary_min" in df.columns:
		df.drop("salary_min", axis=1, inplace=True)
	if "salary_max" in df.columns:
		df.drop("salary_max", axis=1, inplace=True)
	if "salary" in df.columns:
		df.drop("salary", axis=1, inplace=True)

	html_tag_cols = ["additional_requirements", "other_vacancy_benefit", "position_requirements", 
				 "position_responsibilities"]
	for html_col in tqdm(html_tag_cols):
		df[html_col] = df[html_col].map(remove_html_tags)

	df.loc[~df["academic_degree"].isna(), "academic_degree"] = 1
	df["academic_degree"] = df["academic_degree"].fillna(0)


	df.loc[df["accommodation_type"].isin(["FLAT", "ROOM", "HOUSE"]), "accommodation_type"] = 2
	df.loc[df["accommodation_type"] == "DORMITORY", "accommodation_type"] = 1
	df.loc[df["accommodation_type"].isna(), "accommodation_type"] = 0

	df.loc[df["additional_premium"].isna(), "additional_premium"] = -1

	df.loc[df["bonus_type"].isna(), "bonus_type"] = 0
	df.loc[df["measure_type"].isna(), "measure_type"] = 0
	df.loc[df["code_professional_sphere"].isna(), "code_professional_sphere"] = "unknown"
	df.loc[df["contact_source"].isna(), "contact_source"] = "unknown"
	df.loc[df["is_mobility_program"].isna(), "is_mobility_program"] = False
	df.loc[df["need_medcard"].isna(), "need_medcard"] = "unknown"
	df.loc[df["regionName"].isna(), "regionName"] = "unknown"
	df.loc[df["required_experience"].isna(), "required_experience"] = -1
	df.loc[df["retraining_capability"].isna(), "retraining_capability"] = "unknown"
	df.loc[df["retraining_grant_value"].isna(), "retraining_grant_value"] = 0
	df.loc[df["transport_compensation"].isna(), "transport_compensation"] = -1
	df.loc[df["federalDistrictCode"].isna(), "federalDistrictCode"] = 4

	rare_okso = PREPROC_CONFIG["rare_values_replace"]["okso_code"]
	df.loc[df["okso_code"].isin(rare_okso), "okso_code"] = -1
	df.loc[df["okso_code"].isna(), "okso_code"] = -1


	rare_profs = PREPROC_CONFIG["rare_values_replace"]["code_profession"]

	df.loc[df["code_profession"].isin(rare_profs), "code_profession"] = -1
	df.loc[df["code_profession"].isna(), "code_profession"] = -1

	for col in tqdm(list(PREPROC_CONFIG["list_OHE_preproc"].keys())):
		df[col] = df[col].map(list_col_preproc)
		for l in PREPROC_CONFIG["list_OHE_preproc"][col]:
			df[f"{col}_type_{l}"] = df[col].map(lambda x: l in x)
		df.drop(col, axis=1, inplace=True)

	potential_none_texts = ["additional_requirements", "other_vacancy_benefit", "position_requirements",
						"position_responsibilities", "required_certificates", "education_speciality"]
	for col in potential_none_texts:
		df[f"have_{col}"] = df[col].isna()

	df["hardSkills"] = df["hardSkills"].map(lambda x: [k["hard_skill_name"] for k in eval(x)])
	df["hardSkills"] = df["hardSkills"].map(lambda x: " ".join(x))

	df["softSkills"] = df["softSkills"].map(lambda x: [k["soft_skill_name"] for k in eval(x)])
	df["softSkills"] = df["softSkills"].map(lambda x: " ".join(x))

	df["full_text"] = ""
	for c in text_features:
		df["full_text"] += " " + df[c].fillna("")

	preproc_text = []
	for txt in tqdm(df["full_text"]):
		preproc_text.append(base_preproc(txt))
	df["full_text"] = preproc_text

	tfidf = PREPROC_CONFIG["text_vectorizer"]
	vector = tfidf.transform(preproc_text)
	vector = pd.DataFrame(vector.toarray(), columns=[f"text_embed_{i}" for i in range(650)])
	df = pd.concat([df, vector], axis=1, ignore_index=False).dropna(subset=["academic_degree"])

	for col in cols_for_aggregation:
		col_dct = PREPROC_CONFIG["salary_aggregators"][col]
		df[f"min_{col}_salary"] = df[col].map(lambda x: col_dct[x][0])
		df[f"max_{col}_salary"] = df[col].map(lambda x: col_dct[x][1])
		df[f"med_{col}_salary"] = df[col].map(lambda x: col_dct[x][2])
		df[f"mean_{col}_salary"] = df[col].map(lambda x: col_dct[x][3])


	t = ["mean" in x for x in df.columns.tolist()]
	df["nearest_mean_sal"] = df[df.columns[t]].mean(axis=1)

	df["vacancy_address_latitude"] = df["vacancy_address_latitude"].fillna(df["vacancy_address_latitude"].mean())
	df["vacancy_address_longitude"] = df["vacancy_address_longitude"].fillna(df["vacancy_address_longitude"].mean())


	float_f = [x for x in list(set(df.columns[df.dtypes == float]) - set(cat_features)) if "text" not in x]
	for_rescale = ["work_places", "additional_premium", 
	 "vacancy_address_longitude",
	 "required_experience", "vacancy_address_latitude"] + [x for x in float_f if "sal" in x]
	for f in for_rescale:
		df[f] = np.log(df[f])
		
	df = df.fillna(0)
	cat_features = list(set(cat_features) - set(text_features))
	df[cat_features] = df[cat_features].map(lambda x: str(x))

	df.drop(text_features, axis=1, inplace=True)
	df.drop("full_text", axis=1, inplace=True)

	pool = Pool(df, cat_features=cat_features)
	preds = reg.predict(pool)
	preds = np.exp(preds)
	return preds



