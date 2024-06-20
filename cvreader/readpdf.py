import pdfplumber
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
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
        'experience': "How many years of experience does the candidate have?",
        'education': "How many years of education does the candidate have?",
        'coding_skills': "How does the candidate rate their coding skills on a scale of 0 to 10?",
        'teamwork': "How does the candidate rate their teamwork skills on a scale of 0 to 10?"
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
pdf_path = 'ben-hoyt-cv-resume.pdf'
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

# Ustawienie wartości w symulacji rekrutacyjnej
experience_ant = ctrl.Antecedent(np.arange(0, 16, 1), 'experience')
education_ant = ctrl.Antecedent(np.arange(0, 9, 1), 'education')
coding_skills_ant = ctrl.Antecedent(np.arange(0, 11, 1), 'coding_skills')
teamwork_ant = ctrl.Antecedent(np.arange(0, 11, 1), 'teamwork')
decision = ctrl.Consequent(np.arange(0, 101, 1), 'decision')

# Definiowanie funkcji przynależności
experience_ant['novice'] = fuzz.trapmf(experience_ant.universe, [-5, 0, 2, 5])
experience_ant['experienced'] = fuzz.trapmf(experience_ant.universe, [2, 5, 9, 12])
experience_ant['expert'] = fuzz.trimf(experience_ant.universe, [8, 15, 20])

education_ant['moderate'] = fuzz.trapmf(education_ant.universe, [-3, 0, 2, 5])
education_ant['good'] = fuzz.trapmf(education_ant.universe, [4, 5, 6, 7])
education_ant['excellent'] = fuzz.trapmf(education_ant.universe, [6, 7, 8, 10])

for antecedent in [coding_skills_ant, teamwork_ant]:
    antecedent['poor'] = fuzz.trimf(antecedent.universe, [-5, 1, 5])
    antecedent['moderate'] = fuzz.trimf(antecedent.universe, [4, 6, 8])
    antecedent['excellent'] = fuzz.trimf(antecedent.universe, [7, 10, 12])

decision['decline_weak'] = fuzz.trimf(decision.universe, [0, 0, 25])
decision['decline_moderate'] = fuzz.trimf(decision.universe, [0, 25, 50])
decision['consider_moderate'] = fuzz.trimf(decision.universe, [25, 50, 75])
decision['consider_strong'] = fuzz.trimf(decision.universe, [50, 75, 100])
decision['accept_strong'] = fuzz.trimf(decision.universe, [75, 100, 100])
decision['accept_exceptional'] = fuzz.trimf(decision.universe, [75, 100, 100])

# Definiowanie reguł inferencji
rules = [
    ctrl.Rule(experience_ant['expert'] & education_ant['excellent'] & coding_skills_ant['excellent'] & teamwork_ant['excellent'],
              decision['accept_exceptional']),
    ctrl.Rule(experience_ant['expert'] & education_ant['excellent'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['expert'] & education_ant['good'] & coding_skills_ant['excellent'] & teamwork_ant['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['expert'] & education_ant['good'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['excellent'] & coding_skills_ant['excellent'] & teamwork_ant['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['excellent'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['good'] & coding_skills_ant['excellent'] & teamwork_ant['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['good'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience_ant['novice'] & education_ant['moderate'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience_ant['novice'] & education_ant['moderate'] & coding_skills_ant['poor'] & teamwork_ant['poor'],
              decision['decline_weak']),
    ctrl.Rule(coding_skills_ant['poor'] | teamwork_ant['poor'], decision['decline_weak']),
    ctrl.Rule(coding_skills_ant['excellent'] | teamwork_ant['excellent'], decision['accept_strong']),
    ctrl.Rule(coding_skills_ant['moderate'] | teamwork_ant['moderate'], decision['consider_strong'])
]

# Dodajemy reguły na pokrycie wszystkich możliwych przypadków
rules += [
    ctrl.Rule(experience_ant['novice'] & education_ant['moderate'] & coding_skills_ant['excellent'] & teamwork_ant['excellent'],
              decision['consider_strong']),
    ctrl.Rule(experience_ant['novice'] & education_ant['moderate'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience_ant['novice'] & education_ant['good'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience_ant['novice'] & education_ant['excellent'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['moderate'] & coding_skills_ant['excellent'] & teamwork_ant['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience_ant['experienced'] & education_ant['good'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience_ant['expert'] & education_ant['moderate'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience_ant['expert'] & education_ant['good'] & coding_skills_ant['moderate'] & teamwork_ant['moderate'],
              decision['consider_strong'])
]

# Tworzenie systemu kontroli
recruitment_system = ctrl.ControlSystem(rules)
recruitment_simulation = ctrl.ControlSystemSimulation(recruitment_system)

# Ustawianie wartości wejściowych na podstawie wyekstrahowanych danych
recruitment_simulation.input['experience'] = experience
recruitment_simulation.input['education'] = education
recruitment_simulation.input['coding_skills'] = coding_skills
recruitment_simulation.input['teamwork'] = teamwork

try:
    # Obliczanie wyniku decyzji
    recruitment_simulation.compute()
    decision_value = recruitment_simulation.output['decision']
except ValueError as e:
    print("Błąd w obliczaniu wyniku decyzji:", e)
    decision_value = 0

# Dodawanie wyniku do DataFrame
df['Decision'] = decision_value

# Wyświetlanie wyników
print(df)

# Wizualizacja wyników decyzji
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df.index, df['Decision'], color='skyblue')
ax.set_yticks(df.index)
ax.set_yticklabels([f'Candidate {i}' for i in df.index])
ax.set_xlabel('Decision Score')
ax.set_title('Decision Scores for Sample Candidates')

for index, value in enumerate(df['Decision']):
    ax.text(value, index, f'{value:.1f}', va='center')

plt.show()