import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Define the rankings
data = {
    "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "ESPN": ["Christian McCaffrey", "Saquon Barkley", "Ezekiel Elliott", "Dalvin Cook", "Alvin Kamara", 
             "Derrick Henry", "Clyde Edwards-Helaire", "Miles Sanders", "Kenyan Drake", "Josh Jacobs",
             "Nick Chubb", "Austin Ekeler", "Joe Mixon", "Aaron Jones", "Chris Carson", 
             "David Johnson", "Todd Gurley", "James Conner", "Melvin Gordon", "Jonathan Taylor"],
    "Your Prediction": ["Christian McCaffrey", "Derrick Henry", "Aaron Jones", "Ezekiel Elliott", 
                        "Chris Carson", "Dalvin Cook", "Nick Chubb", "Austin Ekeler", "Todd Gurley", 
                        "Mark Ingram", "Alvin Kamara", "David Montgomery", "Melvin Gordon", 
                        "Leonard Fournette", "Saquon Barkley", "JoshJacobs", "RaheemMostert", 
                        "JoeMixon", "MilesSanders", "KenyanDrake"],
    "Actual": ["Derrick Henry", "Alvin Kamara", "Dalvin Cook", "Jonathan Taylor", "Aaron Jones", 
               "David Montgomery", "James Robinson", "Josh Jacobs", "Nick Chubb", "Kareem Hunt", 
               "Ezekiel Elliott", "Melvin Gordon", "Antonio Gibson", "Kenyan Drake", "Ronald Jones", 
               "Chris Carson", "Mike Davis", "David Johnson", "D'Andre Swift", "JK Dobbins"]
}

df = pd.DataFrame(data)

# Calculate the rankings based on the Actual list
actual_ranking = {player: rank for rank, player in enumerate(df["Actual"], 1)}
espn_ranking = [actual_ranking.get(player, 21) for player in df["ESPN"]]
your_ranking = [actual_ranking.get(player, 21) for player in df["Your Prediction"]]
actual_rank = list(range(1, len(df) + 1))

# Metrics
kendall_espn, _ = kendalltau(espn_ranking, actual_rank)
kendall_yours, _ = kendalltau(your_ranking, actual_rank)
spearman_espn, _ = spearmanr(espn_ranking, actual_rank)
spearman_yours, _ = spearmanr(your_ranking, actual_rank)
mare_espn = sum(abs(e - a) for e, a in zip(espn_ranking, actual_rank)) / len(df)
mare_yours = sum(abs(y - a) for y, a in zip(your_ranking, actual_rank)) / len(df)

# Results
print(f"Kendall's Tau (ESPN): {kendall_espn:.3f}")
print(f"Kendall's Tau (Your Prediction): {kendall_yours:.3f}")
print(f"Spearman's Rank Correlation (ESPN): {spearman_espn:.3f}")
print(f"Spearman's Rank Correlation (Your Prediction): {spearman_yours:.3f}")
print(f"Mean Absolute Ranking Error (ESPN): {mare_espn:.3f}")
print(f"Mean Absolute Ranking Error (Your Prediction): {mare_yours:.3f}")
