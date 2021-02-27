# 1. User defined Function
# Define function name to_seconds
def ________(record):
 hms = record.___._____(':', n = 2, expand = True)
 return hms[0]._____(___) * 3600  + hms[1]._____(___) * __ + hms[2].astype(int)

# Call user defined function to_seconds
marathon_2017['Official Time Sec'] = to_seconds(marathon_2017['Official Time'])

# Display updated data frame with .head() method
print(marathon_2017._____)


# 2. Pre defined Function
# Import Numpy Library and call it as np
import ______ as __

# Convert using pandas to_timedelta method
marathon_2017['Official Time Sec'] = pd.__________(marathon_2017['Official Time'])

# Convert time to seconds value using astype method
marathon_2017['Official Time Sec'] = marathon_2017['Official Time Sec'].______('m8[s]')._______(np.int64)

# Display updated data frame with .head() method
print(marathon_2017.______)
