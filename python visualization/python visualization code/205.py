# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt


# Select male and female runners by conditional expression
MALE_runner = marathon_2015_2017[marathon_2015_2017[____] == ___]
FEMALE_runner = marathon_2015_2017[marathon_2015_2017[____] == ___]
# Configure figure size
plt.figure(figsize=(20,20))
# Set MALE_runner.Age as x_male
x_male = MALE_runner.___
# Set MALE_runner Official Time as y_male
y_male = MALE_runner['Official Time']
# Set FEMALE_runner.Age as x_male
x_female = FEMALE_runner.___
# Set FEMALE_runner Official Time as y_male
y_female = FEMALE_runner['Official Time']
# Creae Charts
plt.plot(______, _____, '.', color='b', alpha=0.5)
plt.plot(______, _____ '.', color='r', alpha=0.5)
# Generate labels and title
plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (Second)",fontsize=16)
plt.title("Distribution by Running Time and Age",fontsize=20)
# Show plot
plt.show()

