# upper confidence bound 

# importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
n = 10000 #no of rounds
d = 10 #no of ads
ad_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0, n):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            avg_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound= upper_bound
            ad = i 
    ad_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward
    
    # the first 10 values of ad_selected will be 0, 1, 3, ....9. now after i = d-1 (9)
    # we reach end of inner loop, the two values get updated and n increments to one
    # number_of_selections for each ad is now 1 so we go into the first if condition where the ucb algo does its job.
        
# Visalising Results
plt.hist(ad_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected ')
plt.show()
        