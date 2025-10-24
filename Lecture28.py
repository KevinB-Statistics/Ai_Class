'''
Look for solutions in just about any sapce
Optimization is not the only thing it can do
"Smarter" way to search spaces, typically for optimal solutions
Subset of evolutionary computation

Example:
Trick or treating
*Goal: Maximize candy intake
*Several ways to rank this
** Overall candy amounts
** Quality of candy
** #of full size candy bars

How to setup genetic algorithm
Things to consider:
1. Total candy intake
2. total effort / distance covered
3. quality of candy

Genetic Algorithm Setup:
Individuals
* Are actual solutions you want
* Parameters you want to optimize
* Design you want to use
* Path you want to take
* Model you want to design
* etc
* Constraints: Can't visit the same house twice
* Would like minimum amount of back tracking possible
* Like to visit houses in continuous manner

Population
* Made of individuals
* However many you want - they are different indidivduals, typically
* for trick or treat example, I usually only have one indidivdual: the route I take
* But if I have more contacts with other trick or treaters, maybe I can have a larger population size

Generations
* Different periods of algorithm runs with different population members
* In this example = 1 year
* Different years will ahve different trick or treaters with different routes

Mutation
* This is changing the individuals in your population
* Changing some parameters of the solution, etc
* Lots of different ways of doing this
* Lots of different rates to do this
* In our example, changing the houses visited on the route would be the ideal mutation
* Subject ot the constraints that we still want a mostly continuous paths with minimal revisits

Next generation:
* Getting new generation from existing generation
* Getting next year's route pool from previous years
* Could copy all individuals from the last generation
* Could select a subset based on fitness (tournament - pick some random number of routes and pick the best route of that number)
* could copy parts of individual routes to others (find a route crossover point - take half of one route and half of another. Would probably want to make sure routes halves don't overlap much)

Fitness:
* Likely most important part of genetic algorithm, often hardest to implement or longest to implement
* This is how you score the individual in your population - how good they are, how well they fit the problem
* Single objective - only score on one thing
* Multi-objective - score based on several things. Usually harder to calculate, often more "real-world"
* Performming the fitness calculation is often the hardest or longest part of the genetic algorithm
* Can use a function
* Sometimes a neural network run
* Often some kind of simulation
    Several things we might care about in this example:
    Overall quantityt of candy
    Quality of candy
    How many king size
    how long you have to walk
    how far you have to walk

    Ex: Couple of ways we could calculate this example
    Run experiments (Halloween night routes) and use directly
    Calculate "rules of thumb" (potentailly from experiments) and plug into an equation
    We need to decide what matters most to us here

Exploitation vs. Exploratoin
*Exploitation finding best solution in the subspace
*Exploration finding the best subspace
*Could easily get stuck/converge in local optimization and miss the global

Scoring Fitness
*Could build a high fidelity simulation of this neighborhood
*Could build a heuristic function based on our domain expertise
*Could build a probability distribution and do random sampling from those distributions - repeated sampling to build a confidence interval of route outcomes (could be time or computationally expensive)

Parameterization
*Different aspects of your fitness score
*If you build your genetic algorithm to parameterize things like your hyper parameters and fitness function (so you can define/adjust them later or in different experiments), its often very useful
*Adjusting these will probability have different impacts on the optimal solution
'''