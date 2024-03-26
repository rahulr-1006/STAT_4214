import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/Users/rahulramakrishnan/Documents/spring_2024/STAT_4214/code/Breast_Cancer.csv'
data = pd.read_csv(file_path)

corr_matrix = data[['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
