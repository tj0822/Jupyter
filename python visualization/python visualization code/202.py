# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._______("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt

# Select runners from Age 18 to 60 by conditional expression
runner_1860 = marathon_2015_2017[marathon_2015_2017.Age.____(______(20,60))]
# Create runner_1860_counting Dataframe with counting by Age
runner_1860_counting = runner_1860['Age'].__________
# Store index of runner_1860_counting into x
x = runner_1860_counting.______
# Conver x values to String in order to avoid int sorting
x = [str(i) for i in x]
# Store values of runner_1860_counting into y
y = runner_1860_counting.______
# Calculate ratio and accumulated ratio
ratio = y / y._____
ratio_sum = ratio._______

# Configure figure size
fig, barChart = plt.________(figsize=(20,10))
# Creae bar Chart
barChart.____(x, y)
# Creae line Chart
lineChart = barChart.______
lineChart.plot(x, _______, _____, _____=0.5)
# Creae right side labels
ranges = lineChart.get_yticks()
lineChart.___________(['{:,.1%}'.format(x) for x in ______])
# Creae annotations on line chart
ratio_sum_percentages = ['{0:.0%}'.format(x) for x in ______]
for i, txt in ________(ratio_sum_percentages):
    lineChart.________(txt, (x[i], _______[i]), fontsize=14)    
# Generate labels and title
barChart.set_xlabel('Age', fontdict= {'size':16})
barChart.set_ylabel('Number of runner', fontdict= {'size':16})
plt.title('Pareto Chart - Number of runner by Age', fontsize=18)
# Show plot
plt.____()
