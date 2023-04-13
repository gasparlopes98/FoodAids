import csv
import pandas as pd
max=10000

data=[]

the_file = pd.read_csv("csv_file/Food_Recipes.csv",delimiter=',')

the_file.drop('Image_Name', inplace=True, axis=1)
the_file.drop('Cleaned_Ingredients', inplace=True, axis=1)
# the_file.drop('source', inplace=True, axis=1)

# row_count = sum(1 for row in the_file)

data = the_file[:max]
data.drop('Unnamed: 0', inplace=True, axis=1)
    
print(data)
data.to_csv('csv_file/new_recipes.csv', index=False)
# with open('new_recipes.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data)