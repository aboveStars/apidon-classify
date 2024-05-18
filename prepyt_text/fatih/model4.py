from transformers import pipeline
classifier = pipeline("text-classification", model="vietdata/vietnamese-content-cls", return_all_scores=True)
classifier("The United Nations issued a report today highlighting the urgent need for global action to address climate change.")

