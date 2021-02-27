# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.________("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt
# Import seaborn as a alias 'sns'
import seaborn as sns

# Select runners from USA by conditional expression
USA_runner = marathon_2015_2017[marathon_2015_2017.________ == _____]
USA_MALE_runner = USA_runner[USA_runner[_____] == ___]
USA_FEMALE_runner = USA_runner[USA_runner[_____] == ___]
# Configure figure size
plt.figure(figsize=(20,10))
sns.set(style="ticks", palette="pastel")
# Draw a nested boxplot to show Pace by Gender
sns._____(x=_____, y=_____,
            palette=["m", "g"],
            data=USA_runner)

# Generate USA_MALE_runner_statistics, USA_FEMALE_runner_statistics 
USA_MALE_runner_statistics = USA_MALE_runner[_____]._______
USA_FEMALE_runner_statistics = USA_FEMALE_runner[______].______
