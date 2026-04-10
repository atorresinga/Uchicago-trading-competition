import pandas as pd

# 1. Load the data
# Replace 'your_file.csv' with your actual filename
optuna = 'optuna_momentum_tilt'
df = pd.read_csv(f'{optuna}.csv')

# 2. Calculate a 'Stability Score' (Signal-to-Noise Ratio)
# Higher is better: it means more mean return per unit of standard deviation
df['Stability_Score'] = df['Mean_Sharpe'] / df['Std_Sharpe']

# 3. Define "High Average" and "Low Std" thresholds
# Here we filter for trials where Mean is above the 75th percentile 
# and Std is below the 25th percentile.
mean_threshold = df['Mean_Sharpe'].quantile(0.75)
std_threshold = df['Std_Sharpe'].quantile(0.25)

top_performers = df[
    (df['Mean_Sharpe'] >= mean_threshold) & 
    (df['Std_Sharpe'] <= std_threshold)
].copy()

# 4. Sort by our Stability Score to find the absolute winners
top_performers = top_performers.sort_values(by='Stability_Score', ascending=False)

# 5. Display the best combinations
print(f"Top combinations (Mean > {mean_threshold:.2f} and Std < {std_threshold:.2f}):")
cols_to_show = ['trial', 'blend_rate', 'borrow_penalty', 'Mean_Sharpe', 'Std_Sharpe', 'Stability_Score']
print(top_performers[cols_to_show].head(10))

# 6. Optional: Save the best results to a new CSV
# top_performers.to_csv('best_combinations.csv', index=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['Std_Sharpe'], df['Mean_Sharpe'], alpha=0.5, c=df['Stability_Score'], cmap='viridis')
plt.colorbar(label='Stability Score')
plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Mean Sharpe (Return)')
plt.title('Mean vs. Std Dev: Finding the Efficient Combinations')
plt.show()
plt.savefig(f'{optuna}.png') 