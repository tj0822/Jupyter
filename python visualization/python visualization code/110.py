"""
Step 1. Read Files 
"""
# Import pandas as a alias 'pd'
import ______ as __

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015 = pd.________("./data/marathon_results_2015.csv")
marathon_2016 = pd.________("./data/marathon_results_2016.csv")
marathon_2017 = pd.________("./data/marathon_results_2017.csv")

"""
Step 2. Add Year column 
"""
# Add Year column to the CSV files "marathon_results_2015 ~ 2017.csv"
marathon_2015[______] = ______
marathon_2016[______] = ______
marathon_2017[______] = ______

"""
Step 3. Merge Files
"""
# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Age
marathon_2015_2017 = pd.______([marathon_2015, marathon_2016, marathon_2017], ignore_index=True, sort=False)

"""
Step 4. Drop columns
"""
#Drop unnecessary columns 
marathon_2015_2017 = marathon_2015_2017.______(['Unnamed: 0','Bib', 'Citizen', 'Unnamed: 9', 'Proj Time', 'Unnamed: 8'], axis='columns')

"""
Step 5. Convert columns value
"""
# Import Numpy Library and call it as np
import ______ as __

# Convert using pandas to_timedelta method
marathon_2015_2017['5K'] = pd.___________(marathon_2015_2017['5K'])
marathon_2015_2017['10K'] = pd.___________(marathon_2015_2017['10K'])
marathon_2015_2017['15K'] = pd.___________(marathon_2015_2017['15K'])
marathon_2015_2017['20K'] = pd.___________(marathon_2015_2017['20K'])
marathon_2015_2017['Half'] = pd.___________(marathon_2015_2017['Half'])
marathon_2015_2017['25K'] = pd.___________(marathon_2015_2017['25K'])
marathon_2015_2017['30K'] = pd.___________(marathon_2015_2017['30K'])
marathon_2015_2017['35K'] = pd.___________(marathon_2015_2017['35K'])
marathon_2015_2017['40K'] = pd.___________(marathon_2015_2017['40K'])
marathon_2015_2017['Pace'] = pd.___________(marathon_2015_2017['Pace'])
marathon_2015_2017['Official Time'] = pd.___________(marathon_2015_2017['Official Time'])

# Convert time to seconds value using astype method
marathon_2015_2017['5K'] = marathon_2015_2017['5K'].______('m8[s]').______(np.int64)
marathon_2015_2017['10K'] = marathon_2015_2017['10K'].______('m8[s]').______(np.int64)
marathon_2015_2017['15K'] = marathon_2015_2017['15K'].______('m8[s]').______(np.int64)
marathon_2015_2017['20K'] = marathon_2015_2017['20K'].______('m8[s]').______(np.int64)
marathon_2015_2017['Half'] = marathon_2015_2017['Half'].______('m8[s]').______(np.int64)
marathon_2015_2017['25K'] = marathon_2015_2017['25K'].______('m8[s]').______(np.int64)
marathon_2015_2017['30K'] = marathon_2015_2017['30K'].______('m8[s]').______(np.int64)
marathon_2015_2017['35K'] = marathon_2015_2017['35K'].______('m8[s]').______(np.int64)
marathon_2015_2017['40K'] = marathon_2015_2017['40K'].______('m8[s]').______(np.int64)
marathon_2015_2017['Pace'] = marathon_2015_2017['Pace'].______('m8[s]').______(np.int64)
marathon_2015_2017['Official Time'] = marathon_2015_2017['Official Time'].______('m8[s]').______(np.int64)

"""
Step 6. Save file
"""
# Save to CSV file "marathon_2015_2017.csv"
marathon_2015_2017.______("./data/marathon_2015_2017.csv", index = None, header=True)
