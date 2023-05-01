import streamlit as st
import pickle
import joblib

model = joblib.load('models/svc_tuned.pkl')
file = open("models/tfidf.pickle", "rb")
vectorizer = pickle.load(file)

def predict(model, sentence):
	output = model.predict(vectorizer.transform([text]))[0]
	result = output.item()

	categories = {
		0: "NORMAL",
		1: "SPAM",
	}

	return st.success("THIS MESSAGE IS: " + categories.get(result))

st.title("TURKISH SPAM DETECTION")
text = st.text_input('...')
res = st.button('PREDICT')

if res:
	predict(model, text)
