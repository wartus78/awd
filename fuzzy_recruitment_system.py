
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

def generate_pseudorandom_cvs(num_cvs=100):
    """
    Generates a DataFrame of simulated CVs with pseudorandom values, following sensible distributions for each attribute.
    Experience and Coding Skills are considered crucial and are modeled with a higher mean, whereas Education and Teamwork
    are distributed across a broader range to reflect varied importance and characteristics in candidates.
    
    Parameters:
    - num_cvs (int): Number of CVs to generate.
    
    Returns:
    - DataFrame with columns for each attribute and pseudorandom values within specified ranges.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generating data for each attribute with chosen distributions
    # Experience: Normal distribution around a mean indicating candidates have some years of experience, but with variability.
    experience = np.random.normal(loc=8, scale=3, size=num_cvs).clip(0, 15).astype(int)
    
    # Education: Uniform distribution to represent a wide range of educational backgrounds.
    education = np.random.randint(0, 9, num_cvs)
    
    # Coding Skills: Normal distribution with a higher mean as it's crucial for the role, showing most candidates have good to high skills.
    coding_skills = np.random.normal(loc=7, scale=2, size=num_cvs).clip(0, 10).astype(int)
    
    # Teamwork: Uniform distribution reflecting that candidates can vary widely in this soft skill.
    teamwork = np.random.randint(0, 11, num_cvs)
    
    # Creating DataFrame
    cvs_df = pd.DataFrame({
        'Experience': experience,
        'Education': education,
        'Coding Skills': coding_skills,
        'Teamwork': teamwork
    })
    
    return cvs_df

# Generating pseudorandom CVs with sensible distributions
pseudorandom_cvs = generate_pseudorandom_cvs(100)
pseudorandom_cvs.head()  # Display the first few rows to verify the output


# Redefining the fuzzy control system with updated rules and setup
# Antecedents and Consequent
experience = ctrl.Antecedent(np.arange(0, 16, 1), 'experience')
education = ctrl.Antecedent(np.arange(0, 9, 1), 'education')
coding_skills = ctrl.Antecedent(np.arange(0, 11, 1), 'coding_skills')
teamwork = ctrl.Antecedent(np.arange(0, 11, 1), 'teamwork')
decision = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'decision')

# Experience Membership Functions
experience.automf(3)
# Education Membership Functions
education.automf(3)
# Coding Skills Membership Functions
coding_skills.automf(3)
# Teamwork Membership Functions
teamwork.automf(3)

# Decision Membership Functions
decision['decline'] = fuzz.gaussmf(decision.universe, 0, 0.15)
decision['consider'] = fuzz.gaussmf(decision.universe, 0.5, 0.15)
decision['accept'] = fuzz.gaussmf(decision.universe, 1, 0.15)

# Rules
rules = [
    ctrl.Rule(experience['average'] & coding_skills['good'] | teamwork['good'], decision['consider']),
    ctrl.Rule(coding_skills['good'] & teamwork['good'], decision['accept']),
    ctrl.Rule(coding_skills['poor'] | teamwork['poor'], decision['decline'])
]

# Control System Creation and Simulation Setup
recruitment_system = ctrl.ControlSystem(rules)
recruitment_simulation = ctrl.ControlSystemSimulation(recruitment_system)

# Redefine membership functions with direct specifications based on the recruitment system's design

# Experience Membership Functions
experience['novice'] = fuzz.trapmf(experience.universe, [0, 0, 3, 7])
experience['experienced'] = fuzz.trimf(experience.universe, [3, 7, 10])
experience['expert'] = fuzz.trapmf(experience.universe, [7, 10, 15, 15])

# Education Membership Functions
education['low'] = fuzz.trimf(education.universe, [0, 0, 4])
education['medium'] = fuzz.trimf(education.universe, [2, 4, 6])
education['high'] = fuzz.trimf(education.universe, [4, 6, 8])

# Coding Skills Membership Functions
coding_skills['poor'] = fuzz.trimf(coding_skills.universe, [0, 0, 5])
coding_skills['average'] = fuzz.trimf(coding_skills.universe, [3, 5, 7])
coding_skills['good'] = fuzz.trimf(coding_skills.universe, [5, 7, 10])

# Teamwork Membership Functions
teamwork['poor'] = fuzz.trimf(teamwork.universe, [0, 0, 5])
teamwork['average'] = fuzz.trimf(teamwork.universe, [3, 5, 7])
teamwork['good'] = fuzz.trimf(teamwork.universe, [5, 7, 10])

# Define rules based on the provided setup and the importance of coding skills and teamwork
rules = [
    ctrl.Rule(experience['expert'] | coding_skills['good'] & teamwork['good'], decision['accept']),
    ctrl.Rule(coding_skills['poor'] & teamwork['poor'], decision['decline']),
    ctrl.Rule(experience['novice'] | education['low'], decision['decline']),
    ctrl.Rule(experience['experienced'] & coding_skills['average'] | teamwork['average'], decision['consider'])
]

# Control System Creation and Simulation
recruitment_system = ctrl.ControlSystem(rules)
recruitment_simulation = ctrl.ControlSystemSimulation(recruitment_system)

# Use the modified function to simulate the recruitment system for the generated CVs
# decision_scores = simulate_recruitment(pseudorandom_cvs)

# # Update the DataFrame with decision scores
# pseudorandom_cvs['Decision Score'] = decision_scores

# pseudorandom_cvs.head()


# Define a function to simulate the fuzzy recruitment system for multiple CVs
def simulate_recruitment(cvs_df):
    """
    Simulates the fuzzy recruitment system for a set of CVs.
    
    Parameters:
    - cvs_df (DataFrame): DataFrame containing CVs with columns for Experience, Education, Coding Skills, and Teamwork.
    
    Returns:
    - A list containing the decision scores for each CV.
    """
    # List to store decision scores
    decision_scores = []
    
    # Iterate over each CV in the DataFrame
    for _, cv in cvs_df.iterrows():
        recruitment_simulation.input['experience'] = cv['Experience']
        recruitment_simulation.input['education'] = cv['Education']
        recruitment_simulation.input['coding_skills'] = cv['Coding Skills']
        recruitment_simulation.input['teamwork'] = cv['Teamwork']
        
        # Compute the result for the current CV
        recruitment_simulation.compute()
        
        # Append the decision score to the list
        decision_scores.append(recruitment_simulation.output['decision'])
    
    return decision_scores

# Generating pseudorandom CVs
pseudorandom_cvs = generate_pseudorandom_cvs(100)

# Simulate the recruitment system for the generated CVs
decision_scores = simulate_recruitment(pseudorandom_cvs)

# Add decision scores to the DataFrame
pseudorandom_cvs['Decision Score'] = decision_scores

# Display the updated DataFrame with decision scores
pseudorandom_cvs.head()


 