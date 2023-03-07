import csv
import pandas as pd
max=1000

data=[]

the_file = pd.read_csv("RecipeNLG_dataset.csv")

the_file.drop('ingredients', inplace=True, axis=1)
the_file.drop('link', inplace=True, axis=1)
the_file.drop('source', inplace=True, axis=1)

# row_count = sum(1 for row in the_file)

for row in range(max):
    data.append(the_file.loc[row])
    

# the_file.close()  

with open('new_recipes.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)  