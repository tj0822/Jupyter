# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.________("./data/marathon_2015_2017.csv")

# Import pyplot as a alias 'plt'
import matplotlib.pyplot as plt
# Import seaborn as a alias 'sns'
import ______ as sns

# Configure figure size
plt.figure(figsize=(20,10))
# Create histogram - Distribution Rate
age_count = sns.______(marathon_2015_2017.____)
age_count.set_xlabel('Ages',fontdict= {'size':16})
age_count.set_ylabel('Distribution Rate',fontdict= {'size':16})
age_count.set_title('Distribution Rate by Ages',fontsize=18)
plt.show()

# Configure figure size
plt.figure(figsize=(20,10))
# Create histogram - Distribution by Ages
age_count = sns._______('Age',data=_______________)
age_count.set_title('Distribution by Ages', fontsize=18)
age_count.set_xlabel('Ages', fontdict= {'size':16})
age_count.set_ylabel('Number of Runner', fontdict= {'size':16})
plt.show()

# Configure figure size
plt.figure(figsize=(20,10))
# Create histogram - Distribution by Ages Sorted 
age_count = sns._________('Age',data=______________,order = __________['Age']._________.____)
age_count.set_title('Distribution by Ages Sorted', fontsize=18)
age_count.set_xlabel('Ages', fontdict= {'size':16})
age_count.set_ylabel('Number of Runner', fontdict= {'size':16})
plt.show()
