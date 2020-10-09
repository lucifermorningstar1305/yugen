import numpy as np
import pandas as pd
from joblib import load


def preprocess(df):
	
	# ---------------------------------------------------------------------------
	# 						Applying the Encodings
	# ---------------------------------------------------------------------------

	jobs = {"management":0, "technician" : 1, "entrepreneur": 2, "blue-collar": 3, 
	       "unknown" : 4, "retired" : 5, "admin" : 6, "services" : 7, "self-employed" : 8,
	       "unemployed" : 9, "housemaid" : 10, "student": 11} # Nominal field

	marital = {"single" : 0, "married": 1, "divorced": 2} # Nominal field

	educational = {"unknown":0, "primary" : 1, "secondary" : 2, "tertiary": 3} # Ordinal field

	yes_no_type_fields = {"no" : 0, "yes" : 1}

	contact = {"unknown" : 0, "cellular" : 1, "telephone" : 2} # Nominal field

	month = {"jan":0, "feb":1, "mar":2, "apr":3, "may":4, "jun":5, "jul":6,
	        "aug":7, "sep":8, "oct":9, "nov":10, "dec":11} # Ordinal field

	poutcome = {"unknown":0, "failure":1, "other":2, "success": 3} # Nominal field

	df["job"] = df["job"].apply(lambda x: jobs[x] if x in jobs else -1)
	df["marital"] = df["marital"].apply(lambda x: marital[x] if x in marital else -1)
	df["education"] = df["education"].apply(lambda x: educational[x] if x in educational else -1)
	df["default"] = df["default"].apply(lambda x: yes_no_type_fields[x] if x in yes_no_type_fields else -1)
	df["housing"] = df["housing"].apply(lambda x: yes_no_type_fields[x] if x in yes_no_type_fields else -1)
	df["loan"] = df["loan"].apply(lambda x: yes_no_type_fields[x] if x in yes_no_type_fields else -1)
	df["month"] = df["month"].apply(lambda x: month[x] if x in month else -1)
	df["contact"] = df["contact"].apply(lambda x: contact[x] if x in contact else -1)
	df["poutcome"] = df["poutcome"].apply(lambda x: poutcome[x] if x in poutcome else -1)

	#  ---------------------------------------------------------------------------
	#					Applying Standarization
	# ---------------------------------------------------------------------------

	X = df.values

	std_scaler = load('./MODEL_DATA/std.bin')

	X_std = std_scaler.transform(X)

	return X_std


def predict(data):

	data = pd.DataFrame(data, index=[0])
	preprocessed_data = preprocess(data)

	xgboost_model = load("./MODEL_DATA/model.dat")

	prediction = xgboost_model.predict(preprocessed_data)

	return prediction


def main(full_data):

	predictions = []
	for k, v in full_data.items():
		predictions.extend(list(predict(v)))

	return predictions






