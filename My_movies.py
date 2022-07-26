##    MY_MOVIES
# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt

X= pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\my_movies.csv")

X.drop(['V1','V2','V3','V4','V5'], axis=1, inplace=True)
X.dtypes



frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

k = ('red','blue','green','violet','yellow')
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color =k)
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
L=rules_no_redudancy.sort_values('lift', ascending = False).head(10)

 