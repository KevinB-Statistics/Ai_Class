import numpy as np
import matplotlib.pyplot as plt
#business monte carlo simulation

#modeling function - maps our inuts to an output
def calc_profit(demand, cost_to_produce, sale_price):
    margin = sale_price - cost_to_produce
    profit = demand*margin
    return profit

def get_price(price_type):
    if price_type == "low":
        price = 10
    if price_type =="medium":
        price = 12
    if price_type == "high":
        price = 15
    return price

def get_demand_distribution(price_type):
    #Percent of market size
    if price_type == "low":
        mean = 700
        stdv = 50
    if price_type == "medium":
        mean = 600
        stdv = 10
    if price_type == "high":
        mean = 400
        stdv = 100
    return mean, stdv

#make choice - factors
mean_cost = 6
stdv_cost = 0.5
market_size = 1000
num_samples = 1000
price_type = "low"

price = get_price(price_type)
mean, stdv = get_demand_distribution(price_type)
#Run the monte carlo sampling
demand_samples = np.random.normal(mean,stdv, num_samples)
cost_samples = np.random.normal(mean_cost, stdv_cost, num_samples)
profit_list = []
for demand_num, cost_num in zip(demand_samples, cost_samples):
    profit_num = calc_profit(demand_num, cost_num, price)
    profit_list.append(profit_num)


print(f"Worst case {min(profit_list)}")
print(f"Best Case {max(profit_list)}")
print(f"Average Case {sum(profit_list)/len(profit_list)}")

#Plot it
plt.hist(profit_list)
plt.title(f"{price_type} Output Profit Distribution")
plt.show()
