# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.______("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt
# Import seaborn as a alias 'sns'
import ______ as sns
sns.set()
# Creatr marathon_2015_2017_under60 dataframe under age 60
marathon_2015_2017_under60 = marathon_2015_2017[marathon_2015_2017.Age.______(______(0,60))]
# Counting by age, Male and Female 
marathon = marathon_2015_2017_under60.groupby(_____)[_____]._________._______.______
# Draw a heatmap with the numeric values in each cell
f, ax = plt.______(figsize=(10, 20))
sns.______(marathon, annot=True, fmt="d", linewidths=.5, ax=ax)


