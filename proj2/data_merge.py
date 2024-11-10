import pandas as pd

# Load both CSV files
df1 = pd.read_csv('2020.csv')  # The file that has the new FantasyPoints
df2 = pd.read_csv('2019.csv')  # The file that you want to update
df2['Player'] = df2['Player'].str.replace(r'[^a-zA-Z]', '', regex=True)
df1['Player'] = df1['Player'].str.replace(r'[^a-zA-Z]', '', regex=True)
# Merge df1's FantasyPoints column into df2 based on the 'Player' column
df2 = pd.merge(df2, df1[['Player', 'FantasyPoints']], on='Player', how='left')

# Optionally, save the updated DataFrame to a new CSV file
df2.to_csv('updated_file.csv', index=False)

