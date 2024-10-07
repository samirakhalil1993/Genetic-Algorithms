# Imports
import numpy as np          # Importerar numpy-biblioteket, ett populärt paket för vetenskapliga beräkningar och arbete med stora flerdimensionella matriser.
import random               # Importerar random-modulen, som tillhandahåller funktioner för att generera slumpmässiga tal och val.
from datetime import datetime # Importerar datetime-klassen från datetime-modulen, används för att arbeta med datum och tid.

# Parameters
population_sizes = [10, 20, 50, 100]  # Olika populationsstorlekar att testa
mutation_rates = [0.9, 0.6, 0.3, 0.1]  # Olika mutationsfrekvenser att testa
n_cities = 20            # Definierar antalet städer i problemet, troligen för ett resande säljare-problem (TSP) eller liknande optimeringsproblem.

# Generating a list of coordinates representing each city
coordinates_list = [[x,y] for x,y in zip(np.random.randint(0,100,n_cities), np.random.randint(0,100,n_cities))]
# Skapar en lista med koordinater för varje stad. Koordinaterna är slumpmässigt genererade med heltal mellan 0 och 100, både för x- och y-värden. 
# Listförståelse används här för att skapa en lista med par [x, y] för varje stad.

names_list = np.array(['Berlin', 'London', 'Moscow', 'Barcelona', 'Rome', 'Paris', 'Vienna', 'Munich', 'Istanbul', 'Kyiv', 
                       'Bucharest', 'Minsk', 'Warsaw', 'Budapest', 'Milan', 'Prague', 'Sofia', 'Birmingham', 'Brussels', 'Amsterdam'])
# Skapar en NumPy-array som innehåller namn på 20 städer. Varje namn motsvarar en stad.

cities_dict = { x:y for x,y in zip(names_list, coordinates_list)}
# Skapar en dictionary (ordbok) där varje stadsnamn (från names_list) mappas till en uppsättning koordinater (från coordinates_list). 
# Detta gör det enkelt att slå upp en stads koordinater baserat på namnet.

# Function to compute the distance between two points
def compute_city_distance_coordinates(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
# Denna funktion beräknar avståndet mellan två punkter (a och b) med hjälp av Euklidiskt avstånd. 
# a och b är listor som innehåller två värden (x, y), och formeln används för att beräkna distansen mellan dessa punkter.

def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])
# Denna funktion tar två städers namn som argument (city_a och city_b) och använder cities_dict för att slå upp deras koordinater.
# Sedan används funktionen compute_city_distance_coordinates för att beräkna avståndet mellan städerna baserat på deras koordinater.

cities_dict
# Returnerar ordboken (cities_dict) så att du kan se eller använda den i senare kod.

# First step: Create the first population set
def genesis(city_list, population_sizes):
    # Skapar en funktion som genererar den initiala populationen av lösningar.
    # city_list är en lista över alla städer, och population_sizes är antalet lösningar som ska genereras.

    population_set = []  # Skapar en tom lista för att lagra varje genererad lösning.
    for i in range(population_sizes):  # Loopar population_sizes gånger för att generera varje lösning.
        # Randomly generating a new solution
        sol_i = city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
        # Genererar en slumpmässig lösning genom att slumpmässigt välja n_cities städer från city_list utan återplacering.
        # np.random.choice används för att slumpmässigt välja index och sedan hämta motsvarande städer.

        population_set.append(sol_i)  # Lägger till den genererade lösningen till population_set-listan.

    return np.array(population_set)  # Returnerar populationen som en NumPy-array för mer effektiv hantering.



def fitness_eval(city_list, cities_dict):
    # Denna funktion beräknar fitness-värdet (eller kostnaden) för en viss lösning (city_list) genom att summera avstånden mellan alla städer.
    # city_list är en lista över städer i den ordning de besöks, och cities_dict är en ordbok med stadens namn som nyckel och koordinater som värde.

    total = 0  # Initialiserar totalavståndet till noll.
    for i in range(n_cities - 1):  # Loopar genom varje stad i lösningen fram till den näst sista staden.
        a = city_list[i]  # Hämtar den aktuella staden.
        b = city_list[i + 1]  # Hämtar nästa stad i listan.
        total += compute_city_distance_names(a, b, cities_dict)  
        # Beräknar avståndet mellan den aktuella staden (a) och nästa stad (b), 
        # och lägger till detta avstånd till totalvärdet.

    return total  # Returnerar den totala distansen för staden i den aktuella ordningen.

def get_all_fitnes(population_set, cities_dict):
    # Denna funktion beräknar fitness-värdet för varje lösning i hela populationen.
    # population_set är en uppsättning av alla lösningar, och cities_dict är en ordbok med städers namn och koordinater.

    fitnes_list = np.zeros(n_population)
    # Skapar en NumPy-array fylld med nollor som kommer att lagra fitness-värdet för varje lösning. Arrayen har storleken population_sizes.

    # Looping over all solutions computing the fitness for each solution
    for i in range(n_population):  # Loopar genom varje lösning i populationen.
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict)
        # Beräknar fitness-värdet för den i:te lösningen genom att använda fitness_eval-funktionen och lagrar resultatet i fitnes_list.

    return fitnes_list  # Returnerar listan med fitness-värden.


def progenitor_selection(population_set, fitnes_list):
    # Denna funktion väljer progenitorer (föräldrar) för nästa generation genom att använda ett sannolikhetsbaserat urval.
    # population_set är hela populationen, och fitnes_list är en lista med fitness-värden för varje lösning i populationen.

    total_fit = fitnes_list.sum()
    # Summerar alla fitness-värden i populationen. Detta används för att normalisera sannolikheterna.

    prob_list = (total_fit / fitnes_list)
    # Beräknar sannolikheten för varje lösning att bli vald som progenitor.
    # De lösningar med lägre fitness får en högre sannolikhet, eftersom fitness-värden är lägre för bättre lösningar (omvänd proportion).

    prob_list = prob_list / prob_list.sum()
    # Normaliserar sannolikheterna så att de summerar till 1, vilket krävs för att kunna använda dem i np.random.choice.

    # Notice there is the chance that a progenitor mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list, replace=True)
    # Väljer en uppsättning progenitorer (föräldrar) från populationen med sannolikheterna angivna av prob_list.
    # replace=True betyder att en progenitor kan väljas flera gånger, vilket innebär att en individ kan para sig med sig själv.

    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set), p=prob_list, replace=True)
    # Samma som ovan, men för den andra uppsättningen progenitorer (andra föräldern i varje par).

    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]
    # Använder de valda indexen för att hämta motsvarande lösningar från population_set och skapa listor över progenitorer.

    return np.array([progenitor_list_a, progenitor_list_b])
    # Returnerar en array där progenitor_list_a och progenitor_list_b representerar föräldraparen.


def mate_progenitors(prog_a, prog_b):
    # Denna funktion parar två progenitorer (prog_a och prog_b) och genererar en avkomma (offspring).
    
    offspring = prog_a[0:5]
    # Tar de första 5 städerna från progenitor A (prog_a) som en grund för avkomman.

    for city in prog_b:
        # Itererar över alla städer i progenitor B (prog_b).
        
        if not city in offspring:
            # Om staden från prog_b inte redan finns i offspring (ingen dubblett),
            offspring = np.concatenate((offspring, [city]))
            # Läggs staden till offspring genom att kombinera (concatenate) listan med den nya staden.

    return offspring
    # Returnerar den färdiga avkomman.

def mate_population(progenitor_list):
    # Denna funktion genererar en ny population genom att para alla progenitorpar.
    
    new_population_set = []
    # Skapar en tom lista för att lagra den nya populationen (avkommor).

    for i in range(progenitor_list.shape[1]):
        # Loopar igenom varje par av progenitorer i progenitor_list.
        
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        # Hämtar respektive progenitor (prog_a och prog_b) från progenitor_list för index i.

        offspring = mate_progenitors(prog_a, prog_b)
        # Skapar en avkomma genom att para prog_a och prog_b.

        new_population_set.append(offspring)
        # Lägger till den nya avkomman i new_population_set.

    return new_population_set
    # Returnerar den nya populationen (listan med alla avkommor).


def mutate_offspring(offspring):
    # Denna funktion applicerar mutation på en enskild avkomma genom att byta plats på städer i resvägen.

    for q in range(int(n_cities * mutation_rate)):
        # Loopar ett antal gånger som är baserat på antalet städer multiplicerat med mutationsfrekvensen.
        # Det anger hur många mutationer (bytesoperationer) som ska göras.

        a = np.random.randint(0, n_cities)
        b = np.random.randint(0, n_cities)
        # Väljer två slumpmässiga index inom avkommans lista över städer.

        offspring[a], offspring[b] = offspring[b], offspring[a]
        # Byter plats på städerna vid de två slumpmässigt valda indexen (a och b).

    return offspring
    # Returnerar den muterade avkomman.

def mutate_population(new_population_set):
    # Denna funktion applicerar mutation på hela den nya populationen av avkommor.

    mutated_pop = []
    # Skapar en tom lista för att lagra den muterade populationen.

    for offspring in new_population_set:
        # Loopar igenom varje avkomma i den nya populationen.
        mutated_pop.append(mutate_offspring(offspring))
        # Muterar varje avkomma med funktionen mutate_offspring och lägger till den i listan mutated_pop.

    return mutated_pop
    # Returnerar den muterade populationen.

for n_population in population_sizes:
    for mutation_rate in mutation_rates:
        print(f"Running GA with population size: {n_population} and mutation rate: {mutation_rate}")
        
        # Initiera en tom variabel för bästa lösningen
        best_solution = [-1, np.inf, np.array([])]  
        
        # Generera initial population
        population_set = genesis(names_list, n_population)
        
        # Initiera populationen
        mutated_pop = population_set
        
        # Kör genetiska algoritmen i 10 000 generationer
        for i in range(10000):
            if i % 50 == 0:
                fitnes_list = get_all_fitnes(mutated_pop, cities_dict)  # Beräkna fitness varje gång det behövs
                print(f"Generation {i}: Best solution so far: {best_solution[1]}, Average fitness: {fitnes_list.mean()}, Time: {datetime.now().strftime('%d/%m/%y %H:%M')}")

            
            # Beräkna fitness för den muterade populationen
            fitnes_list = get_all_fitnes(mutated_pop, cities_dict)
            
            # Kontrollera om vi har hittat en bättre lösning
            if fitnes_list.min() < best_solution[1]:
                best_solution[0] = i  # Spara generationen där den bästa lösningen hittades
                best_solution[1] = fitnes_list.min()  # Spara bästa fitness-värdet
                best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]  # Spara bästa lösningen
            
            # Välj progenitorer baserat på fitness
            progenitor_list = progenitor_selection(population_set, fitnes_list)
            
            # Generera ny population via parning
            new_population_set = mate_population(progenitor_list)
            
            # Applicera mutation på den nya populationen
            mutated_pop = mutate_population(new_population_set)
        
        # Skriv ut den bästa lösningen som hittades för denna kombination av population och mutation rate
        print(f"Best solution for population size {n_population}, mutation rate {mutation_rate}: {best_solution[1]} at generation {best_solution[0]}")
        print(f"Best route: {best_solution[2]}")

