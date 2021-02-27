# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd._________("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.______ as plt

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd._______(marathon_2015_2017,columns=['5K',  '10K',  '15K',  '20K', 'Half',  '25K',  '30K',  '35K',  '40K',  'Official Time']).sort_values(by=['Official Time'])

# Insert Rank column
record._______(0, 'Rank', range(1, 1 + len(record)))
# Select Top 100
top100 = record[______]
# Set Rank as x
xData = top100.Rank
# Set yData_full, yData_10K, yData_20K, yData_30K
yData_full = top100['Official Time']
yData_10K = top100['10K']
yData_20K = top100['20K']
yData_30K = top100['30K']

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt
# Configure figure size
plt.figure(figsize=(20,10))
# plot the data yData_full, yData_10K, yData_20K, yData_30K
plt.____(xData, yData_full, color='blue', linewidth=4, linestyle='-', marker='x', label='Full time')
plt.____(xData, yData_10K, color='red', linewidth=3, linestyle='--', marker='s', label='10K time')
plt.____(xData, yData_20K, color='orange', linewidth=2, linestyle='-.', marker='o', label='20K time')
plt.____(xData, yData_30K, color='green', linewidth=1, linestyle=':', marker='d', label='30K time')

# Add a title
plt.title("Top 100 ranker's records")

# Add x-axis label
plt.xlabel("Rank")

# Add y-axis label
plt.ylabel("Time(Second)")

# Legend
plt.______

# Style : ggplot, fivethirtyeight, seaborn, default
plt.style.use('seaborn')

# display the plot
plt.show()
