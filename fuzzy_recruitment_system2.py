import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

# Redefining the Antecedents (Inputs) and Consequent (Output)
experience = ctrl.Antecedent(np.arange(0, 16, 1), 'experience')
education = ctrl.Antecedent(np.arange(0, 9, 1), 'education')
coding_skills = ctrl.Antecedent(np.arange(0, 11, 1), 'coding_skills')
teamwork = ctrl.Antecedent(np.arange(0, 11, 1), 'teamwork')
#decision = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'decision')

# Experience Membership Functions
experience['novice'] = fuzz.trapmf(experience.universe, [-5, 0, 2, 5])
experience['experienced'] = fuzz.trapmf(experience.universe, [2, 5, 9, 12])
experience['expert'] = fuzz.trimf(experience.universe, [8, 15, 20])

# Education Membership Functions
education['moderate'] = fuzz.trapmf(education.universe, [-3, 0, 2, 5])
education['good'] = fuzz.trapmf(education.universe, [4, 5, 6, 7])
education['excellent'] = fuzz.trapmf(education.universe, [6, 7, 8, 10])

# Coding Skills and Teamwork Membership Functions (Same for both)
for antecedent in [coding_skills, teamwork]:
    antecedent['poor'] = fuzz.trimf(antecedent.universe, [-5, 0, 5])
    antecedent['moderate'] = fuzz.trimf(antecedent.universe, [4, 6, 8])
    antecedent['excellent'] = fuzz.trimf(antecedent.universe, [7, 10, 12])


decision = ctrl.Consequent(np.arange(0, 101, 1), 'decision')

# Decision Membership Functions
# decision['decline'] = fuzz.gaussmf(decision.universe, 0, 50)
# decision['consider'] = fuzz.gaussmf(decision.universe, 1, 80)
# decision['accept'] = fuzz.gaussmf(decision.universe, 50, 100)


decision['decline_weak'] = fuzz.trimf(decision.universe, [0, 0, 25])
decision['decline_moderate'] = fuzz.trimf(decision.universe, [0, 25, 50])
decision['consider_moderate'] = fuzz.trimf(decision.universe, [25, 50, 75])
decision['consider_strong'] = fuzz.trimf(decision.universe, [50, 75, 100])
decision['accept_strong'] = fuzz.trimf(decision.universe, [75, 100, 100])
decision['accept_exceptional'] = fuzz.trimf(decision.universe, [75, 100, 100])

#show the membership functions
# experience.view()
# education.view()
# coding_skills.view()
# teamwork.view()

# decision.view()
# plt.show()

# Define the rules based on the provided methodology and system development details
# rules = [
#     ctrl.Rule(experience['expert'] & coding_skills['excellent'] & teamwork['excellent'], decision['accept']),
#     ctrl.Rule(coding_skills['poor'] | teamwork['poor'], decision['decline']),
#     ctrl.Rule(coding_skills['moderate'] | teamwork['moderate'], decision['consider']),
#     ctrl.Rule(experience['novice'] & education['moderate'], decision['consider']),
#     # Add more rules as needed based on the detailed system development description
#     # Continue adding all other rules here following the initial pattern
# ]
rules = [
    # Expertise level: expert
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
    # Expertise level: experienced
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
    # Expertise level: novice
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




#graph rules



# Control System Creation and Simulation
recruitment_system = ctrl.ControlSystem(rules)
recruitment_simulation = ctrl.ControlSystemSimulation(recruitment_system)

# Example Simulation
recruitment_simulation.input['experience'] = 15
recruitment_simulation.input['education'] = 9
recruitment_simulation.input['coding_skills'] = 8
recruitment_simulation.input['teamwork'] = 10

# Compute the result
recruitment_simulation.compute()

# Output the decision score
print("Decision Score:", recruitment_simulation.output['decision'])
#print the decision
# print("Decision ", recruitment_simulation.output['decision'])

#linguistic interpretation of decision score
#decision.view(sim=recruitment_simulation)
#hold the plot
#input("Press Enter to continue...")


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
    teamwork = np.random.randint(0, 10, num_cvs)
    #print histogram of each attribute
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, (attr, name) in enumerate(zip([experience, education, coding_skills, teamwork], ['Experience', 'Education', 'Coding Skills', 'Teamwork'])):
        axs[i].hist(attr, bins=10, edgecolor='black')
        axs[i].set_title(f'{name} Distribution')
        axs[i].set_xlabel(name)
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Creating DataFrame
    cvs_df = pd.DataFrame({
        'Experience': experience,
        'Education': education,
        'Coding Skills': coding_skills,
        'Teamwork': teamwork
    })
    
    return cvs_df




# Generating pseudorandom CVs with sensible distributions
pseudorandom_cvs = generate_pseudorandom_cvs(1000)
print(pseudorandom_cvs)  # Display the first few rows to verify the output



 # add the decision to the dataframe
pseudorandom_cvs['Decision'] = 0

def linguistic_decision(crisp_output):
    """
    Converts a crisp decision score into a linguistic interpretation based on defined levels.
    
    Parameters:
    - crisp_output (float): The crisp decision score.
    
    Returns:
    - str: The linguistic interpretation of the decision.
    """
    if crisp_output >= 75:
        if crisp_output > 90:  # Adjust this threshold as needed for "exceptional"
            return 'Accept Exceptional'
        return 'Accept Strong'
    elif crisp_output >= 50:
        if crisp_output > 62.5:  # Midpoint between 50 and 75 for "strong" consideration
            return 'Consider Strong'
        return 'Consider Moderate'
    else:  # Below 50 falls into the decline categories
        if crisp_output > 25:
            return 'Decline Moderate'
        return 'Decline Weak'

# Loop through each CV and calculate the decision score
for index, cv in pseudorandom_cvs.iterrows():
    recruitment_simulation.input['experience'] = cv['Experience']
    recruitment_simulation.input['education'] = cv['Education']
    recruitment_simulation.input['coding_skills'] = cv['Coding Skills']
    recruitment_simulation.input['teamwork'] = cv['Teamwork']
    
    recruitment_simulation.compute()
    
    decision_score = recruitment_simulation.output['decision']
    pseudorandom_cvs.at[index, 'Decision'] = decision_score
    #pseudorandom_cvs['Linguistic Decision'] = pseudorandom_cvs['Decision'].apply(linguistic_decision)
    print("Decision ", decision_score, " ", linguistic_decision(decision_score))
    
    # Display the CV details and decision
    print("\nCV Details:")
    print(cv)
    print("\nDecision Score:", decision_score)
    print("Linguistic Decision:", linguistic_decision(decision_score))

    # Display the decision plot
    # decision.view(sim=recruitment_simulation)
    #wait
    # input("Press Enter to continue...")

    
# Display the updated DataFrame with decision scores
print(pseudorandom_cvs)

#sort the dataframe by decision score
pseudorandom_cvs.sort_values(by='Decision', ascending=False, inplace=True)
print(pseudorandom_cvs)





# # Apply the linguistic decision function to the 'Decision' column
# pseudorandom_cvs['Linguistic Decision'] = pseudorandom_cvs['Decision'].apply(linguistic_decision)

# # Display the updated DataFrame with linguistic decisions
# print(pseudorandom_cvs)


# # print(decision.view(sim=recruitment_simulation))

# #display ranking of CVs based on decision scores
# # Sort the DataFrame by the 'Decision' column in descending order
# pseudorandom_cvs.sort_values(by='Decision', ascending=False, inplace=True)

# # Display the updated DataFrame with decision scores and rankings
# print(pseudorandom_cvs)


#HISTORGRAM OF DECISION SCORES
# Plot a histogram of the decision scores to visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(pseudorandom_cvs['Decision'], bins=10, edgecolor='black')
plt.title('Distribution of Decision Scores')
plt.xlabel('Decision Score')
plt.ylabel('Frequency')
plt.show()



#



def visualize_example_decisions(cvs_df, num_examples=1):
    sample_cvs = cvs_df.sample(num_examples)
    for index, cv in sample_cvs.iterrows():
        recruitment_simulation.input['experience'] = cv['Experience']
        recruitment_simulation.input['education'] = cv['Education']
        recruitment_simulation.input['coding_skills'] = cv['Coding Skills']
        recruitment_simulation.input['teamwork'] = cv['Teamwork']
        recruitment_simulation.compute()
        decision_score = recruitment_simulation.output['decision']

        print(f"\nCV {index}:")
        print(cv)
        print(f"Decision Score: {decision_score}")
        print(f"Linguistic Decision: {linguistic_decision(decision_score)}")

        # Visualize the decision
        experience.view(sim=recruitment_simulation)
        # save figure
        plt.savefig(f'awd\dokumentacja\cv_{index}_experience.png')
        education.view(sim=recruitment_simulation)
        #save figure
        plt.savefig(f'awd\dokumentacja\cv_{index}_education.png')
        coding_skills.view(sim=recruitment_simulation)
        #save figure
        plt.savefig(f'awd\dokumentacja\cv_{index}_coding_skills.png')
        teamwork.view(sim=recruitment_simulation)
        #save figure
        plt.savefig(f'awd\dokumentacja\cv_{index}_teamwork.png')
        decision.view(sim=recruitment_simulation)
        #save figures
        plt.savefig(f'awd\dokumentacja\cv_{index}_decision.png')
        #hold the plot

        plt.show()


# Visualize example decisions
visualize_example_decisions(pseudorandom_cvs)

