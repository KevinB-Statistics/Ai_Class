'''
Uses candy_data.csv
'''
import pandas as pd
import random
import copy
import math

def preprocess_data(demand, df):
    # Create a profit column for each candy
    df["profit"] = ((1+1*(df["winpercent"]/100)) - (1*df["pricepercent"]))*demand*(df["winpercent"]/100)
    overall_dict = {}
    list_of_records = df.to_dict('records')
    for record in list_of_records:
        overall_dict[record['competitorname']] = record
    overall_dict.pop('One dime')
    overall_dict.pop('One quarter')
    return overall_dict

class Line:
    def __init__(self, candy_options, candy_dict, already_candies=[]):
        self.candy_dict = candy_dict.copy()
        # Candies that will be present on the line
        self.candy_list = []
        self.candy_units = 0
        self.candy_limit = 8 #Magic number - good candidate to paramaterize later
        self.candy_options = candy_options.copy()
        self.random_init_candies(already_candies=already_candies)
        self.prev_candy = None
        self.new_candy = None
    
    def get_candy_units(self, candy_name):
        if self.candy_dict[candy_name]["pluribus"] == 1:
            units = 2
        else:
            units = 1
        return units
    
    def random_init_candies(self, already_candies=[]):
        tries = 0
        candy_options = self.candy_options.copy()
        # if the candy is already being made somewhere on our line
        for candy in already_candies:
            candy_options.remove(candy)
        while self.candy_units < self.candy_limit and tries <20:
            #pick a candy
            tries += 1
            candy_choice = random.choice(candy_options)
            units = self.get_candy_units(candy_choice)
            total_units = self.candy_units + units
            if total_units <= self.candy_limit:
                # Add the units
                self.candy_units += units
                # Add the candy to the list
                self.candy_list.append(candy_choice)
                #Pop this candy off our options list
                self.candy_options.remove(candy_choice)
                candy_options.remove(candy_choice)
    
    def create_new_candy_options(self, other_candies=[]):
        new_candy_options = self.candy_options.copy()
        if other_candies != None:
            for candy in other_candies:
                new_candy_options.remove(candy)
        return new_candy_options
    
    def replace_candy(self, other_candies=None):
        #randomly pick a candy on our line
        old_candy_choice = random.choice(self.candy_list)
        old_units = self.get_candy_units(old_candy_choice)
        new_candy_options = self.create_new_candy_options(other_candies=other_candies)
        new_candy_choice = random.choice(new_candy_options)
        new_units = self.get_candy_units(new_candy_choice)
        return old_candy_choice, old_units, new_candy_choice, new_units
    
    def mutate_candy(self, other_candies = None):
        tries = 0
        old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
        while((self.candy_units - old_units + new_units > self.candy_limit)) and tries < 20:
            tries += 1
            old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
        if (self.candy_units - old_units + new_units) <= self.candy_limit:
            # Release our old candy back into options
            self.candy_options.append(old_candy_choice)
            self.candy_units = self.candy_units - old_units 
            self.candy_list.remove(old_candy_choice)
            # add new candy
            self.candy_list.append(new_candy_choice)
            self.prev_candy = old_candy_choice
            self.candy_units = self.candy_units + new_units
            self.new_candy = new_candy_choice
            self.candy_options.remove(new_candy_choice)
    
    
    def return_candy_list(self):
        return self.candy_list.copy()
    
    def calc_total_candy_profit(self):
        total_profit = 0
        for candy_name in self.candy_list:
            total_profit += self.candy_dict[candy_name]['profit']
        return total_profit

    def __lt__(self,other):
        return self.calc_total_candy_profit() < other.calc_total_candy_profit()



demand = 796142
df = pd.read_csv("candy-data.csv")
candy_dict = preprocess_data(demand,df)
# print(json.dumps(candy_dict,indent=4))
candy_options = list(candy_dict.keys())
candy_line = Line(candy_options, candy_dict)
print(candy_line.return_candy_list())
print(candy_line.calc_total_candy_profit())
candy_line.mutate_candy()
print(candy_line.return_candy_list())
print(candy_line.calc_total_candy_profit())