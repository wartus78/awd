import matplotlib.pyplot as plt
import networkx as nx

# Definicja reguł w formie listy krotek (antecedent, consequent)
rules = [
    ("expert & excellent_education & excellent_coding & excellent_teamwork", "accept_exceptional"),
    ("expert & excellent_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("expert & good_education & excellent_coding & excellent_teamwork", "accept_strong"),
    ("expert & good_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("expert & moderate_education & excellent_coding & excellent_teamwork", "accept_strong"),
    ("expert & moderate_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("experienced & excellent_education & excellent_coding & excellent_teamwork", "accept_strong"),
    ("experienced & excellent_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("experienced & good_education & excellent_coding & excellent_teamwork", "accept_strong"),
    ("experienced & good_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("experienced & moderate_education & excellent_coding & excellent_teamwork", "accept_strong"),
    ("experienced & moderate_education & excellent_coding & moderate_teamwork", "accept_strong"),
    ("experienced & moderate_education & moderate_coding & excellent_teamwork", "consider_strong"),
    ("experienced & moderate_education & moderate_coding & moderate_teamwork", "consider_strong"),
    ("novice & excellent_education & excellent_coding & excellent_teamwork", "consider_strong"),
    ("novice & excellent_education & excellent_coding & moderate_teamwork", "consider_strong"),
    ("novice & good_education & excellent_coding & excellent_teamwork", "consider_moderate"),
    ("novice & good_education & excellent_coding & moderate_teamwork", "consider_moderate"),
    ("novice & moderate_education & excellent_coding & excellent_teamwork", "consider_moderate"),
    ("novice & moderate_education & excellent_coding & moderate_teamwork", "consider_moderate"),
    ("novice & moderate_education & moderate_coding & excellent_teamwork", "consider_moderate"),
    ("novice & moderate_education & moderate_coding & moderate_teamwork", "consider_moderate"),
    ("novice & excellent_education & moderate_coding & excellent_teamwork", "decline_moderate"),
    ("novice & excellent_education & moderate_coding & moderate_teamwork", "decline_moderate"),
    ("novice & good_education & moderate_coding & excellent_teamwork", "decline_moderate"),
    ("novice & good_education & moderate_coding & moderate_teamwork", "decline_moderate"),
    ("novice & moderate_education & poor_coding & excellent_teamwork", "decline_moderate"),
    ("novice & moderate_education & poor_coding & moderate_teamwork", "decline_moderate"),
    ("novice & good_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("novice & good_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("novice & moderate_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("novice & moderate_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("novice & excellent_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("novice & excellent_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("experienced & excellent_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("experienced & excellent_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("experienced & good_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("experienced & good_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("experienced & moderate_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("experienced & moderate_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("expert & excellent_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("expert & excellent_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("expert & good_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("expert & good_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("expert & moderate_education & poor_coding & excellent_teamwork", "decline_weak"),
    ("expert & moderate_education & poor_coding & moderate_teamwork", "decline_weak"),
    ("coding_skills['poor'] | teamwork['poor']", "decline_weak"),
    ("coding_skills['excellent'] | teamwork['excellent']", "accept_strong"),
    ("coding_skills['moderate'] | teamwork['moderate']", "consider_strong")
]

# Tworzenie grafu
G = nx.DiGraph()

for antecedent, consequent in rules:
    G.add_edge(antecedent, consequent)

# Rysowanie grafu
plt.figure(figsize=(20, 15))
pos = nx.spring_layout(G, k=0.5)  # Adjust k for better spacing
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', font_size=9, font_weight='bold', arrows=True)
plt.title("Graf zależności reguł inferencji")
# plt.savefig("/mnt/data/rules_graph.png", bbox_inches='tight')  # Zapisz obrazek z odpowiednim przycięciem
plt.show()


