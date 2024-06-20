import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

# Definiowanie Antecedentów (Wejść) i Konsekwencji (Wyjścia)
experience = ctrl.Antecedent(np.arange(0, 16, 1), 'experience')
education = ctrl.Antecedent(np.arange(0, 9, 1), 'education')
coding_skills = ctrl.Antecedent(np.arange(0, 11, 1), 'coding_skills')
teamwork = ctrl.Antecedent(np.arange(0, 11, 1), 'teamwork')
decision = ctrl.Consequent(np.arange(0, 101, 1), 'decision')

# Definiowanie funkcji przynależności
experience['novice'] = fuzz.trapmf(experience.universe, [-5, 0, 2, 5])
experience['experienced'] = fuzz.trapmf(experience.universe, [2, 5, 9, 12])
experience['expert'] = fuzz.trimf(experience.universe, [8, 15, 20])

education['moderate'] = fuzz.trapmf(education.universe, [-3, 0, 2, 5])
education['good'] = fuzz.trapmf(education.universe, [4, 5, 6, 7])
education['excellent'] = fuzz.trapmf(education.universe, [6, 7, 8, 10])

for antecedent in [coding_skills, teamwork]:
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
    ctrl.Rule(experience['expert'] & education['excellent'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_exceptional']),
    ctrl.Rule(experience['expert'] & education['excellent'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['expert'] & education['good'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience['expert'] & education['good'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['expert'] & education['moderate'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience['expert'] & education['moderate'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['excellent'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['excellent'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['good'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['good'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['accept_strong']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['moderate'] & teamwork['excellent'],
              decision['consider_strong']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['moderate'] & teamwork['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['consider_strong']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['consider_strong']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['excellent'] & teamwork['excellent'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['excellent'] & teamwork['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['moderate'] & teamwork['excellent'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['moderate'] & teamwork['moderate'],
              decision['consider_moderate']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['moderate'] & teamwork['excellent'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['moderate'] & teamwork['moderate'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['moderate'] & teamwork['excellent'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['moderate'] & teamwork['moderate'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_moderate']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['novice'] & education['good'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['novice'] & education['moderate'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['novice'] & education['excellent'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['excellent'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['excellent'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['good'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['good'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['experienced'] & education['moderate'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['excellent'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['excellent'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['good'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['good'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['moderate'] & coding_skills['poor'] & teamwork['excellent'],
              decision['decline_weak']),
    ctrl.Rule(experience['expert'] & education['moderate'] & coding_skills['poor'] & teamwork['moderate'],
              decision['decline_weak']),
    ctrl.Rule(coding_skills['poor'] | teamwork['poor'], decision['decline_weak']),
    ctrl.Rule(coding_skills['excellent'] | teamwork['excellent'], decision['accept_strong']),
    ctrl.Rule(coding_skills['moderate'] | teamwork['moderate'], decision['consider_strong'])
]

# Tworzenie systemu kontroli
recruitment_system = ctrl.ControlSystem(rules)
recruitment_simulation = ctrl.ControlSystemSimulation(recruitment_system)

# Definiowanie przykładów kandydatów
sample_candidates = pd.DataFrame({
    'Experience': [3, 10, 7, 15, 1],
    'Education': [4, 8, 5, 6, 2],
    'Coding Skills': [6, 9, 7, 10, 4],
    'Teamwork': [5, 7, 6, 8, 3]
})

# Obliczanie wyników decyzji dla przykładowych kandydatów
sample_candidates['Decision'] = 0
sample_candidates['Linguistic Decision'] = ''

def linguistic_decision(crisp_output):
    if crisp_output >= 75:
        if crisp_output > 90:
            return 'Accept Exceptional'
        return 'Accept Strong'
    elif crisp_output >= 50:
        if crisp_output > 62.5:
            return 'Consider Strong'
        return 'Consider Moderate'
    else:
        if crisp_output > 25:
            return 'Decline Moderate'
        return 'Decline Weak'

for index, candidate in sample_candidates.iterrows():
    recruitment_simulation.input['experience'] = candidate['Experience']
    recruitment_simulation.input['education'] = candidate['Education']
    recruitment_simulation.input['coding_skills'] = candidate['Coding Skills']
    recruitment_simulation.input['teamwork'] = candidate['Teamwork']
    
    recruitment_simulation.compute()
    
    decision_score = recruitment_simulation.output['decision']
    sample_candidates.at[index, 'Decision'] = decision_score
    sample_candidates.at[index, 'Linguistic Decision'] = linguistic_decision(decision_score)

# Wyświetlanie wyników
print(sample_candidates)

# Wizualizacja wyników decyzji dla przykładowych kandydatów
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(sample_candidates.index, sample_candidates['Decision'], color='skyblue')
ax.set_yticks(sample_candidates.index)
ax.set_yticklabels([f'Candidate {i}' for i in sample_candidates.index])
ax.set_xlabel('Decision Score')
ax.set_title('Decision Scores for Sample Candidates')

for index, value in enumerate(sample_candidates['Decision']):
    ax.text(value, index, f'{value:.1f}', va='center')

plt.show()
