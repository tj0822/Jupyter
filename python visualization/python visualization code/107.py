# Import pandas as a alias 'pd'
import ______ as __

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015 = pd.________("./data/marathon_results_2015.csv")
marathon_2016 = pd.________("./data/marathon_results_2016.csv")
marathon_2017 = pd.________("./data/marathon_results_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
marathon_2015_2017 = pd.________([marathon_2015, marathon_2016, marathon_2017], ignore_index=True, sort=False).set_index(___________)


# Display data frame statistics using the .describe() method
print(marathon_2015_2017._________)

# Sort data by Age 
marathon_2015_2017.__________(by=[___________])

# Display the first five initial rows using the .head() method
print(marathon_2015_2017._______)

# Sort data Descending order by Age 
marathon_2015_2017.__________(by=__________, ascending=______)

# Display the first five initial rows using the .head() method
print(marathon_2015_2017._______)

# Save to CSV file "marathon_2015_2017.csv" with header
marathon_2015_2017._________("./data/marathon_2015_2017.csv", index = None, header=_____)
