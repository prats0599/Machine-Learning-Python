# apriori
# optimising sales of a grocery store

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# importing dataset. apriori takes input as a list of lists having a string datatype.
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range (0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
    
# Training Apriori on the dataset.
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) #output
# min_support = 3*7/7500 -> a product which is purchased atleast three times a day. makes it 7*3 times a week.
# since there are 7500 transactions min_support will be 21/7500 = 0.0028(rounding up)
# min_confidence =20% (20 is optimum for this dataset) minimum 20% confidence that the association will happen

# Visualising the results
results = list(rules)
the_rules = []
for result in results:
    the_rules.append({'rule':','.join(result.items),
                      'support':result.support,
                      'confidence':result.ordered_statistics[0].confidence,
                      'lift':result.ordered_statistics[0].lift})
df = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])
