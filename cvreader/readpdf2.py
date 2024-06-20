import pdfplumber
import pandas as pd
import numpy as np
from transformers import pipeline
import re

# Funkcja do ekstrakcji tekstu z pliku PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Funkcja do analizy i przetwarzania tekstu za pomocą modelu językowego
def interpret_cv_text_with_model(text):
    # Użycie modelu językowego do ekstrakcji informacji
    nlp = pipeline('question-answering', model="deepset/roberta-base-squad2")
    
    questions = {
        'experience': "How many years of experience does Tasiana Ukura have as a software engineer?",
        'education': "How many years of education in computer science does Tasiana Ukura have?",
        'coding_skills': "How would you rate Tasiana Ukura's coding skills on a scale from 0 to 10?",
        'teamwork': "How would you rate Tasiana Ukura's teamwork skills on a scale from 0 to 10?"
    }
    
    context = text
    answers = {}
    
    for key, question in questions.items():
        result = nlp(question=question, context=context)
        answers[key] = result['answer']
    
    # Konwersja odpowiedzi do odpowiednich typów danych
    def extract_number(answer):
        match = re.search(r'\d+', answer)
        if match:
            return int(match.group(0))
        return 0
    
    experience = extract_number(answers.get('experience', '0'))
    education = extract_number(answers.get('education', '0'))
    coding_skills = extract_number(answers.get('coding_skills', '0'))
    teamwork = extract_number(answers.get('teamwork', '0'))
    
    return experience, education, coding_skills, teamwork

# Przykład użycia funkcji
pdf_path = 'Python-Dev-1.pdf'
text = extract_text_from_pdf(pdf_path)
experience, education, coding_skills, teamwork = interpret_cv_text_with_model(text)

# Tworzenie DataFrame z wyodrębnionymi danymi
data = {
    'Experience': [experience],
    'Education': [education],
    'Coding Skills': [coding_skills],
    'Teamwork': [teamwork]
}
df = pd.DataFrame(data)

# Wyświetlanie wyników
print(df)
