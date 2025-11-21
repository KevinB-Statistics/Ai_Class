#Kevin Bui
######################################
# Load packages
#######################################
import pandas as pd 
import numpy as np
import random 
import math 
import matplotlib.pyplot as plt
import seaborn as sns
# • Add a second line to the population. You are not allowed to make the same candies on both lines (no repeat candies).​
# • Add recombination to next generation selection. Repeat candies still not allowed. ​

######################################
# Preprocess Code
#######################################

def preprocess_data(demand, df):
    #Create a profit column for each candy 
    df["profit"] = ((1+1*(df["winpercent"]/100)) - (1*df["pricepercent"]))*demand*(df["winpercent"]/100)
    overall_dict = {}
    list_of_records = df.to_dict('records')
    for record in list_of_records:
        overall_dict[record['competitorname']] = record
    overall_dict.pop('One dime')
    overall_dict.pop('One quarter')
    return overall_dict

# candy_dict: from preprocessing; maps name -> attributes, including profit and pluribus.
# candy_list: the current set of candies on this line (the "genes" of the chromosome).
# candy_units: a capacity measure, kept less than or equal to candy_limit = 8. Each candy uses 1 or 2 units.
# candy_options: candies that are not currently on this line and can be added later.
# already_candies: intended for multi-line setups to avoid duplicates across lines; here usually empty.
# random_init_candies randomly fills the line subject to the capacity constraint.
# prev_candy, new_candy: bookkeeping for last mutation.
class Line:
    def __init__(self, candy_options, candy_dict, already_candies=[]):
        self.candy_dict = candy_dict.copy()
        #Candies that will be present on the line 
        self.candy_list = []
        self.candy_units = 0 
        self.candy_limit = 8 #Magic Number - good candidate to parameterize later 
        self.candy_options = candy_options.copy()
        self.random_init_candies(already_candies=already_candies)
        self.prev_candy = None
        self.new_candy = None 

# Uses the pluribus attribute from the dataset.
# If pluribus == 1, the candy counts as 2 units (e.g., multi-piece bags).
# Otherwise, 1 unit.
# This enforces a constraint: total candy_units must not exceed 8.
    def get_candy_units(self, candy_name):
        if self.candy_dict[candy_name]["pluribus"] == 1:
            units = 2
        else:
            units = 1
        return units 
# Copy available candy_options.
# Remove candies in already_candies (so this line won’t reuse them).
# While the line has capacity and we have not tried too many times (max 20):
# Randomly choose a candidate candy.
# Compute units for that candy.
# If adding it keeps candy_units less than equal to candy_limit:
# Add candy to candy_list.
# Increase candy_units.
# Remove candy from both self.candy_options and the local candy_options.
# This yields a random feasible configuration (subject to the unit constraint).
    def random_init_candies(self, already_candies=[]):
        tries = 0 
        candy_options = self.candy_options.copy()
        #if the candy is already being made somewhere on our line 
        for candy in already_candies:
            candy_options.remove(candy)
        while self.candy_units < self.candy_limit and tries < 20:
            #Pick a candy
            tries += 1 
            candy_choice = random.choice(candy_options)
            units = self.get_candy_units(candy_choice)
            total_units = self.candy_units + units 
            if total_units <= self.candy_limit:
                #Add the units
                self.candy_units += units 
                #Add the candy to the list
                self.candy_list.append(candy_choice)
                #Pop this candy off our options list 
                self.candy_options.remove(candy_choice)
                candy_options.remove(candy_choice)
# This method prepares a set of candies that:
    # Are not currently used on this line (self.candy_options), and
    # Are not used in other_candies (e.g., another line, to ensure no duplicates across lines).
    def create_new_candy_options(self, other_candies = None):
    # Start from all candies currently in this line's options
        new_candy_options = self.candy_options.copy()
    # If no other_candies are given, just return the copy
        if other_candies is None:
            return new_candy_options
    # Remove only candies that are actually present
        for candy in other_candies:
            if candy in new_candy_options:
                new_candy_options.remove(candy)
        return new_candy_options
# This is a proposed mutation step:
# Select a random old_candy_choice already on the line.
# Compute its old_units.
# Build new_candy_options (respecting exclusions).
# Randomly choose a new_candy_choice from that set.
# Compute its new_units.
# Return the pair (old_candy, new_candy) with their units.
    def replace_candy(self, other_candies=None):
        #Randomly pick a candy on our line 
        old_candy_choice = random.choice(self.candy_list)
        old_units = self.get_candy_units(old_candy_choice)
        new_candy_options = self.create_new_candy_options(other_candies=other_candies)
        new_candy_choice = random.choice(new_candy_options)
        new_units = self.get_candy_units(new_candy_choice)
        return old_candy_choice, old_units, new_candy_choice, new_units

# This is the genetic mutation on one individual:
# 1. Propose a replacement (old, new) candy.
# 2. While the replacement would violate the unit constraint and we have tried less than 20 times:
        # Propose a new (old, new) pair.
# 3. If a feasible replacement is found:
        # Return the old candy to self.candy_options.
        # Remove it from candy_list and subtract its units.
        # Add the new candy to candy_list, increase units.
        # Remove the new candy from options.
        # Record prev_candy and new_candy for possible debugging.
        # So mutation is a swap: replace one candy on the line with another, respecting capacity and (potentially) cross-line uniqueness.

    def mutate_candy(self, other_candies=None):
        tries = 0 
        old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
        while ((self.candy_units - old_units + new_units > self.candy_limit)) and tries < 20:
            tries += 1
            old_candy_choice, old_units, new_candy_choice, new_units = self.replace_candy(other_candies=other_candies)
        if (self.candy_units - old_units + new_units) <= self.candy_limit:
            #Release our old candy back into options 
            self.candy_options.append(old_candy_choice)
            self.candy_units = self.candy_units - old_units 
            self.candy_list.remove(old_candy_choice)
            #Add the new candy
            self.candy_list.append(new_candy_choice)
            self.prev_candy = old_candy_choice
            self.candy_units = self.candy_units + new_units
            self.new_candy = new_candy_choice
            self.candy_options.remove(new_candy_choice)

    def return_candy_list(self):
        return self.candy_list.copy()

#This is the fitness function for a line:
# fitness(Line) = sum(profit(c))
    def calc_total_candy_profit(self):
        total_profit = 0
        for candy_name in self.candy_list:
            total_profit += self.candy_dict[candy_name]['profit']
        return total_profit
#Helper methods for inspecting line
    def print_self(self):
        print(f"Candies: {self.candy_list}")
        print(f"Candy Units: {self.candy_units}")
        print(f"Line Profit {round(self.calc_total_candy_profit(), 2)}")
    
    def print_profit(self):
        print(f"Line Profit {round(self.calc_total_candy_profit(), 2)}")
# Comparison operator:
# Defines "less than" so that:
# Line objects are sortable,
# Ordering is by total profit.
# list.sort(reverse=True) will then put higher-profit lines first.
    def __lt__(self, other):
        return self.calc_total_candy_profit() < other.calc_total_candy_profit()
######################################
# A6 Extension Code Two Lines + Crossover
#######################################
class TwoLineIndividual:
    def __init__(self, candy_options, candy_dict):
        #First line A
        self.lineA = Line(candy_options, candy_dict, already_candies=[])

        #Second line B (mutually exclusive from A)
        self.lineB = Line(candy_options, candy_dict, already_candies = self.lineA.return_candy_list())

        self.check_constraints()
    # Calc fitness (profit from sum of two disjoint lines)
    def calc_total_profit(self):
        return(self.lineA.calc_total_candy_profit() + self.lineB.calc_total_candy_profit())
    
    def print_self(self):
        print("Line A")
        self.lineA.print_self()
        print("Line B")
        self.lineB.print_self()
    
    def print_profit(self):
        print(f"Total Two Line Profit: {self.calc_total_profit():.2f}")

    def __lt__(self,other):
        return self.calc_total_profit() < other.calc_total_profit()
    
    def mutate(self):
        # Mutate either line A or line B ensuring disjoint sets
        if random.random() < 0.5:
            #mutate line A, excluding candies currently in line B
            self.lineA.mutate_candy(other_candies=self.lineB.return_candy_list())
        else:
            #mutate line B, excluding candies currently in line A
            self.lineB.mutate_candy(other_candies=self.lineA.return_candy_list())

        #Checking constratins
        self.check_constraints()
    @classmethod
    def crossover(cls, parent1, parent2, candy_options, candy_dict):
        #Creates child from two parents while respecting the capacity limits and no repeated candies
        def build_child_line(parent_line1, parent_line2, forbidden_list):
            #Get union of candies from the two parents
            union = []
            for candy in parent_line1.candy_list:
                if candy not in union:
                    union.append(candy)
            for candy in parent_line2.candy_list:
                if candy not in union:
                    union.append(candy)
            #Remove forbidden candies 
            cleaned_union = []
            for candy in union:
                if candy not in forbidden_list:
                    cleaned_union.append(candy)
            #Sort candies by profit
            cleaned_union.sort(key=lambda c: candy_dict[c]["profit"], reverse=True)
            #Build the child line by adding candies until full
            child_candies = []
            units_used = 0
            for candy in cleaned_union:
                pluribus_value = candy_dict[candy]["pluribus"]
                if pluribus_value == 1:
                    units = 2
                else:
                    units = 1

                fits_capacity = (units_used + units <= 8)
                not_duplicate = (candy not in child_candies)
                if fits_capacity and not_duplicate:
                    child_candies.append(candy)
                    units_used += units
            return child_candies
        #Build line A (no forbidden candies)
        childA_list = build_child_line(parent1.lineA, parent2.lineA, forbidden_list=[])

        #Build line B (forbidden candies)
        childB_list = build_child_line(parent1.lineB, parent2.lineB, forbidden_list=childA_list)

        #Create brand new individual
        child = cls(candy_options, candy_dict)

        #Overwrite Line A in the child
        child.lineA.candy_list = childA_list

        total_units_A = 0
        for candy in childA_list:
            if candy_dict[candy]["pluribus"] == 1:
                total_units_A += 2
            else:
                total_units_A += 1
        child.lineA.candy_units = total_units_A

        #Overwrite Line B in the child
        child.lineB.candy_list = childB_list

        total_units_B = 0
        for candy in childB_list:
            if candy_dict[candy]["pluribus"] == 1:
                total_units_B += 2
            else:
                total_units_B += 1
        child.lineB.candy_units = total_units_B

        #Build valid candy options for line A
        A_options = []
        for c in candy_options:
            if (c not in childA_list) and (c not in childB_list):
                A_options.append(c)
        child.lineA.candy_options = A_options


        #Build valid candy options for line B
        B_options = []
        for c in candy_options:
            if (c not in childB_list) and (c not in childA_list):
                B_options.append(c)
        child.lineB.candy_options = B_options
        
        #Checking constraints
        child.check_constraints()
        return child
    
    def check_constraints(self):
        # No duplicates in line A
        seen_A = set()
        for candy in self.lineA.candy_list:
            if candy in seen_A:
                raise ValueError(f"Duplicate candy '{candy}' found in line A")
            seen_A.add(candy)

        # No duplicates in line B
        seen_B = set()
        for candy in self.lineB.candy_list:
            if candy in seen_B:
                raise ValueError(f"Duplicate candy '{candy}' found in line B")
            seen_B.add(candy)

        # No candy appears in both lines
        intersection = seen_A.intersection(seen_B)
        if len(intersection) > 0:
            raise ValueError(f"Candies present in BOTH lines: {intersection}")

class Population:
    def __init__(self, candy_options, candy_dict, members, top_members, use_crossover=False, mutation_rate = 0.1):
        self.candy_options = candy_options.copy()
        self.candy_dict = candy_dict.copy()
        self.member_num = members 
        self.top_members_num = top_members
        self.tournament_size = 4 
        self.mutation_rate = mutation_rate
        self.use_crossover = use_crossover

        self.members = []
        self.top_members = [] 
        #Initialize our population 
        for i in range(0, self.member_num):
            new_individuals = TwoLineIndividual(self.candy_options, self.candy_dict)
            self.members.append(new_individuals)
        # Sort by fitness (highest first)
        self.members.sort(reverse=True)
        #Copy best members to top members list 
        for i in range(0, self.top_members_num):
            self.top_members.append(self.copy_member(self.members[i]))

    #Checking constraints
    def check_population_constraints(self):
        for i, individual in enumerate(self.members):
            try:
                individual.check_constraints()
            except ValueError as e:
                raise ValueError(f"Constraint violation in individual {i}: {e}")
    
    #Update top_members after evolution
    def update_top_rules(self):
        self.members.sort(reverse=True)
        self.top_members = self.members[:self.top_members_num]


    def copy_member(self, og):
        #Intialize a new individual with two lines
        new_ind = TwoLineIndividual(self.candy_options, self.candy_dict)

        #Copy line A
        new_ind.lineA.candy_list = og.lineA.candy_list.copy()
        new_ind.lineA.candy_units = og.lineA.candy_units
        new_ind.lineA.candy_options = og.lineA.candy_options.copy()

        #Copy line B
        new_ind.lineB.candy_list = og.lineB.candy_list.copy()
        new_ind.lineB.candy_units = og.lineB.candy_units
        new_ind.lineB.candy_options = og.lineB.candy_options.copy()
        
        return new_ind
################################################
# Mutation at population level
################################################
    def mutate(self):
        #Number of population members to mutate based on our mutation rate 
        mutation_number = math.floor(self.member_num*self.mutation_rate)
        #Sample of our mutants 
        to_mutate = random.sample(self.members, mutation_number)
        #Perform the mutation 
        for member in to_mutate:
            member.mutate() #mutates either line A or line B

################################################
# Tournament selection helpers
################################################
    def _select_parent(self):
        #Pick one parent using tournament selection (no copying)
        #Randomly sample tournament_size individuals
        selection_list = random.sample(self.members, self.tournament_size)
        selection_list.sort(reverse=True)
        winner = selection_list[0]
        return winner

    def _make_child(self):
        #Creates a single child individual. Use crossover if enable; otherwise, clones a tournament winner like in class code
        if self.use_crossover:
            #Selects two parents with replacement
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            child = TwoLineIndividual.crossover(parent1, parent2, self.candy_options, self.candy_dict)
        else:
            winner = self._select_parent()
            child = self.copy_member(winner)
        return child
    
################################################
# Make new generation
################################################
    def new_generation(self):
        new_generation = []
        for i in range(0, self.member_num):
            child = self._make_child()
            new_generation.append(child)
        self.members = new_generation
        self.members.sort(reverse=True)
################################################
# Run a single generation
################################################
    def run_generation(self):
        #In one generation:
        #1. Mutate a subset of the current members.
        #2. Use selection and optionally crossover to create a new generation
        #3. Update the list of top members.
        self.mutate()
        self.new_generation()
        self.update_top_rules()
        #Checks for constraints
        self.check_population_constraints()
################################################
#Print detailed information (candies, units, profit) for the best num_members individuals.
################################################
    def print_top_members(self, num_members=None):
        if num_members == None:
            num_members = self.top_members_num
        self.top_members.sort(reverse=True)
        for i in range(0, num_members):
            #Each member is a TwoLineIndividual with a print_self function
            self.top_members[i].print_self()

    # Print just the profits of the best num_members individuals.
    def print_top_members_profit(self, num_members=None):
        if num_members == None:
            num_members = self.top_members_num
        self.top_members.sort(reverse=True)
        for i in range(0, num_members):
            self.top_members[i].print_profit()

def run_experiment(pop_size, generations, use_crossover, mutation_rate, seed):
    random.seed(seed)
    np.random.seed(seed)

    pop = Population(candy_options = candy_options, candy_dict=candy_dict, members = pop_size, top_members=10, use_crossover=use_crossover, mutation_rate=mutation_rate)

    for i in range(generations):
        pop.run_generation()
    best = pop.top_members[0]
    best_profit = best.calc_total_profit()
    best_lineA = best.lineA.candy_list.copy()
    best_lineB = best.lineB.candy_list.copy()

    return{"pop_size": pop_size,
           "generations": generations,
           "use_crossover": use_crossover,
           "mutation_rate": mutation_rate,
           "seed": seed,
           "best_profit": best_profit,
           "lineA": best_lineA,
           "lineB": best_lineB}

######################################
# Load in data and preprocess
#######################################
demand = 796142
df = pd.read_csv("candy-data.csv")
candy_dict = preprocess_data(demand, df)
candy_options = list(candy_dict.keys())

results = []
for use_crossover in [False, True]:
    for pop_size in [100, 200]:
        for mutation_rate in [0.05, 0.1, 0.2]:
            gens = 100
            for seed in [1,2,3]:
                res = run_experiment(pop_size=pop_size,
                                     generations=gens,
                                     use_crossover=use_crossover,
                                     mutation_rate=mutation_rate,
                                     seed=seed)
                results.append(res)
#######################################
# Save results for assignment extension
#######################################
# df_results = pd.DataFrame(results)
# print(df_results.to_string(index=False))
# df_results.to_csv("ga_two_lines_results.csv", index=False)

########################################
# One line code (same as class) used as baseline
#######################################
class BaselinePopulation:
    def __init__(self, candy_options, candy_dict, members, top_members, mutation_rate=0.1):
        self.candy_options = candy_options.copy()
        self.candy_dict = candy_dict.copy()
        self.member_num = members 
        self.top_members_num = top_members
        self.tournament_size = 4 
        self.mutation_rate = mutation_rate 
        self.members = []
        self.top_members = [] 
        #Initialize our population 
        for i in range(0, self.member_num):
            new_line = Line(candy_options, candy_dict)
            self.members.append(new_line)
        self.members.sort(reverse=True)
        #Copy best members to top members list 
        for i in range(0, self.top_members_num):
            self.top_members.append(self.copy_member(self.members[i]))

    def update_top_rules(self):
        self.members.sort(reverse=True)
        self.top_members = self.members[:self.top_members_num]

    def copy_member(self, og):
        #Intialize a new individual 
        new_line = Line(self.candy_options, self.candy_dict)
        #Copy the candy list 
        new_line.candy_list = og.candy_list.copy()
        #Copy the candy options 
        new_line.candy_options = og.candy_options.copy()
        #Copy the candy units 
        new_line.candy_units = og.candy_units
        return new_line

    def mutate(self):
        #Number of population members to mutate based on our mutation rate 
        mutation_number = math.floor(self.member_num*self.mutation_rate)
        #Sample of our mutants 
        to_mutate = random.sample(self.members, mutation_number)
        #Perform the mutation 
        for member in to_mutate:
            member.mutate_candy()

    def tournament_selection(self):
        selection_list = random.sample(self.members, self.tournament_size)
        selection_list.sort(reverse=True)
        winner = selection_list[0]
        return self.copy_member(winner)

    def new_generation(self):
        new_generation = []
        for i in range(0, self.member_num):
            new_generation.append(self.tournament_selection())
        self.members = new_generation
        self.members.sort(reverse=True)

    def run_generation(self):
        #self.update_top_rules()
        self.mutate()
        self.new_generation()
        self.update_top_rules()

    def print_top_members(self, num_members=None):
        if num_members == None:
            num_members = self.top_members_num
        self.top_members.sort(reverse=True)
        for i in range(0, num_members):
            #Each member is a Line with a print_self function
            self.top_members[i].print_self()

    def print_top_members_profit(self, num_members=None):
        if num_members == None:
            num_members = self.top_members_num
        self.top_members.sort(reverse=True)
        for i in range(0, num_members):
            #Each member is a Line with a print_self function
            self.top_members[i].print_profit()

######################################
# Loop through class notes
#######################################
def run_experiment_baseline(pop_size, generations, mutation_rate, seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Create baseline population
    pop = BaselinePopulation(
        candy_options=candy_options,
        candy_dict=candy_dict,
        members=pop_size,
        top_members=10,
        mutation_rate=mutation_rate
    )

    # Run GA loop
    for g in range(generations):
        pop.run_generation()

    # Extract best line
    best = pop.top_members[0]
    best_profit = best.calc_total_candy_profit()
    best_line = best.candy_list.copy()

    # Return experiment record
    return {
        "pop_size": pop_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "best_profit": best_profit,
        "best_line": best_line
    }


baseline_results = []

for pop_size in [100, 200]:
    for mutation_rate in [0.05, 0.1, 0.2]:
        gens = 100
        for seed in [1, 2, 3]:
            res = run_experiment_baseline(
                pop_size=pop_size,
                generations=gens,
                mutation_rate=mutation_rate,
                seed=seed
            )
            baseline_results.append(res)
######################################
# Save csv for baseline comparison
#######################################
# df_baseline = pd.DataFrame(baseline_results)
# print(df_baseline.to_string(index=False))
# df_baseline.to_csv("ga_baseline_results.csv", index=False)

######################################
# Load both CSV to compare results
#######################################
df_base = pd.read_csv("ga_baseline_results.csv")
df_two = pd.read_csv("ga_two_lines_results.csv")

#Add setup label
df_base["setup"] = "baseline (1 line)"
df_two_no_cross = df_two[df_two["use_crossover"] == False].copy()
df_two_cross = df_two[df_two["use_crossover"] == True].copy()

df_two_no_cross["setup"] = "2 lines (no crossover)"
df_two_cross["setup"] = "2 lines (with crossover)"

cols_common = ["pop_size", "generations", "mutation_rate", "seed", "best_profit", "setup"]

df_base_small = df_base[["pop_size", "generations", "mutation_rate", "seed", "best_profit", "setup"]]

df_two_no_cross_small = df_two_no_cross[cols_common]
df_two_cross_small = df_two_cross[cols_common]

#Combined dataframe
df_all = pd.concat([df_base_small, df_two_no_cross_small, df_two_cross_small], ignore_index=True)
df_all.to_csv("df_all.csv", index=False)

######################################
# Plotting
######################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import string


# Types, ordering, aggregation
setup_order = ["baseline (1 line)", "2 lines (no crossover)", "2 lines (with crossover)"]

df_all["setup"] = pd.Categorical(df_all["setup"],
                                 categories=setup_order,
                                 ordered=True)

df_all["seed"] = df_all["seed"].astype(int).astype(str)
df_all["pop_size"] = df_all["pop_size"].astype(int).astype(str)
df_all["mutation_rate"]= df_all["mutation_rate"].astype(str)

agg_cols = ["pop_size", "setup", "seed", "mutation_rate"]

df_plot = (df_all.groupby(agg_cols, observed=False, as_index=False)["best_profit"].mean())


# Color scheme
sns.set_theme(style="whitegrid", context="talk")

seed_levels = sorted(df_plot["seed"].unique(), key=lambda s: int(s))
n_seeds = len(seed_levels)

base_palette = sns.color_palette("bright", n_colors=3)
base_color_by_setup = {
    "baseline (1 line)": base_palette[0], # blue
    "2 lines (no crossover)": base_palette[1], # orange
    "2 lines (with crossover)": base_palette[2], # green
}

# Map (setup, seed) -> specific shade of the base color
color_map = {}
for setup in setup_order:
    base = base_color_by_setup[setup]
    # light-to-dark shades of the base color
    shades = sns.light_palette(base, n_colors=n_seeds + 2, reverse=True)
    shades = shades[1:n_seeds + 1]  # drop extreme ends
    for k, seed in enumerate(seed_levels):
        color_map[(setup, seed)] = shades[k]


# catplot: grouped by setup, hue=seed
g = sns.catplot(data=df_plot, kind="bar", x="setup", y="best_profit",hue="seed", col="pop_size", row="mutation_rate",dodge=True, errorbar=None, height=4.5,aspect=1.3, legend=False)

g.set_axis_labels("", "Best profit")
g.set_titles("pop_size = {col_name} | mutation_rate = {row_name}")

for ax_row in g.axes:
    for ax in ax_row:
        if ax is None:
            continue
        ax.set_xticklabels([])

# Recolor bars and add {seed} labels
n_x = len(setup_order)
n_h = n_seeds

for ax_row in g.axes:
    for ax in ax_row:
        if ax is None:
            continue
        patches = ax.patches
        if len(patches) == 0:
            continue
        for i, setup in enumerate(setup_order):
            for h, seed in enumerate(seed_levels):
                idx = h * n_x + i                        
                if idx >= len(patches):
                    continue
                patch = patches[idx]
                patch.set_facecolor(color_map[(setup, seed)])
                patch.set_edgecolor("black")
                x = patch.get_x() + patch.get_width() / 2.0
                y = patch.get_height()
                if not (y > 0):                         
                    continue
                ax.text(x, y * 1.01, f"{{{seed}}}",ha="center", va="bottom", fontsize=10)

# Format y ticks
for ax in g.axes.flat:
    if ax is None:
        continue
    sf = mtick.ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

# Panel Labels
panel_labels = [f"{c})" for c in string.ascii_uppercase]
for idx, ax in enumerate(g.axes.flat):
    if ax is None:
        continue
    ax.text(0.02, 0.96, panel_labels[idx], transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
    
#Figure title
g.figure.suptitle("Comparison of GA extensions: best profit across seeds and settings", fontsize=24, x=0.55, y=0.98)

# legend
legend_handles = []
legend_labels = []

for setup in setup_order:
    base = base_color_by_setup[setup]
    legend_handles.append(Patch(facecolor=base, edgecolor="black"))
    legend_labels.append(setup)

g.figure.legend(
    handles=legend_handles,
    labels=legend_labels,
    title="Extension (setup)",
    loc="upper center",
    ncol=len(setup_order),
    bbox_to_anchor=(0.575, 0.02)
)

g.figure.subplots_adjust()
plt.tight_layout()
plt.show()