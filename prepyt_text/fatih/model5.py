from transformers import pipeline
classifier = pipeline("text-classification", model="Softechlb/articles_classification", return_all_scores=True)
classifier("The United Nations issued a report today highlighting the urgent need for global action to address climate change.")

