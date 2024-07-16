#importing required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history, plot_contour


#Dragonfly algorithm 
class DragonflyAlgorithm:
    def __init__(self, n_features, n_dragonflies=10, max_iterations=10, w=0.9, c=0.1, s=0.1):
        self.n_features = n_features
        self.n_dragonflies = n_dragonflies
        self.max_iterations = max_iterations
        self.w = w
        self.c = c
        self.s = s
        self.best_fitness_over_time = []
        self.cost_history = [] #initialize cost history
        
        print("Default values initialised for dragonfly algorithm....")

    def evaluate_features(self, features, X_train, X_test, y_train, y_test):
        if len(features) == 0:
            return 0
        selected_X_train = X_train[:, features]
        selected_X_test = X_test[:, features]
        # Compute mutual information scores for selected features
        mi_scores = mutual_info_classif(selected_X_train, y_train)
        # Return the mean of mutual information scores as fitness value
        return np.mean(mi_scores)

    def optimize(self, X_train, X_test, y_train, y_test):
        print("Executing optimize function of DA algorithm")
        positions = np.random.randint(0, 2, size=(self.n_dragonflies, self.n_features)).astype(float)
        velocities = np.random.uniform(-1, 1, size=(self.n_dragonflies, self.n_features))
        best_solution = positions[0]
        best_fitness = 0.0

        for iteration in range(self.max_iterations):
            for i in range(self.n_dragonflies):
                features = np.where(positions[i] >= 0.5)[0]
                fitness = self.evaluate_features(features, X_train, X_test, y_train, y_test)
                print("fitness in DA:", fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = positions[i].copy()
                    
                self.cost_history.append(best_fitness)  # Save cost value

                inertia = self.w * velocities[i]
                cognitive = self.c * np.random.uniform(0, 1, self.n_features) * (best_solution - positions[i])
                social = self.s * np.random.uniform(0, 1, self.n_features) * (positions.mean(axis=0) - positions[i])
                velocities[i] = inertia + cognitive + social
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                print(best_solution)
            self.best_fitness_over_time.append(best_fitness)

        return np.where(best_solution >= 0.5)[0]
    def plot_fitness_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_fitness_over_time) + 1), self.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
        plt.plot(self.cost_history)#plot of da cost history
        

#Artificial Bee Colony Algorithm
class ArtificialBeeColony:
    def __init__(self, n_features, n_bees=10, max_iterations=10, limit=5):
        self.n_features = n_features
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.limit = limit
        self.best_fitness_over_time=[]
        self.cost_history = []#initialize abc cost history
        print("Default values initialised for ABC algorithm")

    def evaluate_features(self, features, X_train, X_test, y_train, y_test):
        if len(features) == 0:
            return 0
        selected_X_train = X_train[:, features]
        selected_X_test = X_test[:, features]
        # Compute mutual information scores for selected features
        mi_scores = mutual_info_classif(selected_X_train, y_train)
        # Return the mean of mutual information scores as fitness value
        return np.mean(mi_scores)

    def optimize(self, X_train, X_test, y_train, y_test):
        print("Executing optimize function of ABC algorithm")
        positions = np.random.randint(0, 2, size=(self.n_bees, self.n_features)).astype(float)
        fitness = np.zeros(self.n_bees)
        trial = np.zeros(self.n_bees)
        best_solution = positions[0]
        best_fitness = 0.0

        for iteration in range(self.max_iterations):
            for i in range(self.n_bees):
                features = np.where(positions[i] >= 0.5)[0]
                fitness[i] = self.evaluate_features(features, X_train, X_test, y_train, y_test)
                if fitness[i] > best_fitness:
                    best_fitness = fitness[i]
                    best_solution = positions[i].copy()
                    print("best solution in ABC",best_solution)
                self.cost_history.append(best_fitness)  # Save cost value
            self.best_fitness_over_time.append(best_fitness)

            for i in range(self.n_bees):
                partner = np.random.randint(0, self.n_bees)
                candidate = np.copy(positions[i])
                phi = np.random.uniform(-1, 1, self.n_features)
                candidate += phi * (positions[i] - positions[partner])
                candidate = np.clip(candidate, 0, 1)
                candidate_fitness = self.evaluate_features(np.where(candidate >= 0.5)[0], X_train, X_test, y_train, y_test)
                
                if candidate_fitness > fitness[i]:
                    positions[i] = candidate
                    fitness[i] = candidate_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

            for i in range(self.n_bees):
                if trial[i] > self.limit:
                    positions[i] = np.random.randint(0, 2, self.n_features)
                    trial[i] = 0

        return np.where(best_solution >= 0.5)[0]
    def plot_fitness_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.best_fitness_over_time) + 1), self.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()
        plt.plot(self.cost_history) #plot cost history of abc

#data preprocessing and hybrid feature selection  
class DataPreprocessors:
    def __init__(self, data_path):
        self.data_path = data_path
        self.da_cost_history = []  # Initialize cost history for Dragonfly Algorithm
        self.abc_cost_history = []  # Initialize cost history for Artificial Bee Colony
        

    def preprocess_and_train(self, max_iterations_da=2, max_iterations_abc=2):
        df = pd.read_csv(self.data_path)
        print(df.head())
        df.dropna(inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        if 'label' not in df.columns:
            raise ValueError("The dataset does not contain a 'label' column.")
        X = df.drop(['label'], axis=1)
        y = df['label']
        print("X:", X)
        print("y:", y)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        selected_features = self.hybrid_feature_selection(X_scaled, y, max_iterations_da, max_iterations_abc)

        if len(selected_features) == 0:
            raise ValueError("No features selected by the hybrid feature selection method.")
        X_selected = X_scaled[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        selected_feature_names = X.columns[selected_features]
        print("Selected Features:", selected_feature_names)
        # Save the preprocessed data
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        subprocess.run(['python', 'code-implementation\classifier-svm.py'], check=True)
        # Visualize optimization history
        self.visualize_optimization_history()

    def hybrid_feature_selection(self, X, y, max_iterations_da, max_iterations_abc):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        da = DragonflyAlgorithm(n_features=X_train.shape[1], max_iterations=max_iterations_da)

        selected_features_da = da.optimize(X_train, X_test, y_train, y_test)
        da.plot_fitness_over_time()
        self.da_cost_history = da.cost_history  # Store cost history
        print("Selected features resulted after execution of DA:",selected_features_da)
        abc = ArtificialBeeColony(n_features=X_train.shape[1], max_iterations=max_iterations_abc)
        

        
        selected_features_abc = abc.optimize(X_train, X_test, y_train, y_test)
        abc.plot_fitness_over_time()
        self.abc_cost_history = abc.cost_history  # Store cost history
        print("Selected features resulted after execution of ABC:",selected_features_abc)
        final_selected_features = np.intersect1d(selected_features_da, selected_features_abc)
        print("Combining results rendered by DA and ABC.....")
        print("FEATURES EXTRACTED BY MOHADA ALGORITHM:", final_selected_features)
        self.visualize_optimization_history()
        
        # Plot fitness values for DA
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(da.best_fitness_over_time) + 1), da.best_fitness_over_time, marker='o', linestyle='-', color='b')
        plt.title('Best Fitness Value Over Iterations (DA)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)

        # Plot fitness values for ABC
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(abc.best_fitness_over_time) + 1), abc.best_fitness_over_time, marker='o', linestyle='-', color='r')
        plt.title('Best Fitness Value Over Iterations (ABC)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=final_selected_features)
        plt.title('Box Plot of Final Selected Features')
        plt.xlabel('Features')
        plt.ylabel('Value')
        plt.show()

        return final_selected_features
    def visualize_optimization_history(self):
        plt.figure(figsize=(12, 6))

        # Plot DA cost history
        plt.plot(self.da_cost_history, label='Dragonfly Algorithm Cost History')
        # Plot ABC cost history
        plt.plot(self.abc_cost_history, label='Artificial Bee Colony Cost History')

        plt.xlabel('Iteration')
        plt.ylabel('Cost Fitness Function)')
        plt.title('Optimization Cost History')
        plt.legend()
        plt.show()

import pandas as pd
# Usage example
if __name__ == "__main__":
    data_path = 'dataset/dataset_sdn.csv'
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessors(data_path)
    print("Executing the Hybrid feature selection module...")
    preprocessor.preprocess_and_train(max_iterations_da=2, max_iterations_abc=2)
    
    #selects rows from the DataFrame df where the value in the 'label' column is equal to 0 and 1 (indicating a normal or DDOS attack).
    malign = df[df['label'] == 1]
    benign = df[df['label'] == 0]

    #len calculates the number of rows where 'label' is 0 or 1 
    #calculates the percentage of DDOS attacks and normal flows relative to the total dataset size.
    print('Number of DDOS attacks that has occured :',round((len(malign)/df.shape[0])*100,2),'%')
    print('Number of DDOS attacks that has not occured :',round((len(benign)/df.shape[0])*100,2),'%')


    # Let's plot the Label class against the Frequency
    labels = ['benign','malign']
    # computes the percentage distribution of each label class
    #classes = pd.value_counts(df['label'], sort = True) / df['label'].count() *100
    classes = df['label'].value_counts(sort=True) / df['label'].count() * 100

    classes.plot(kind = 'bar') #creates a line plot of the label class distribution
    plt.title("Label class distribution")
    plt.xticks(range(2), labels)
    plt.xlabel("Label")
    plt.ylabel("Frequency %")


    #visualizing the distribution of a specific variable ('pktcount') across different categories defined by the 'label' column in the DataFrame. 
    #It leverages density plots to show the shape and spread of the data within each category#sets the transparency level of the density plot
    import matplotlib.pyplot as plt

    # Assuming 'df' is your DataFrame containing the data
    # Assuming 'label' is the column you want to use for coloring

    # Set the style of the plot
    #plt.style.use('seaborn-whitegrid')
    import seaborn as sns
    sns.set_style('whitegrid')


    # Plot density plots for each label category
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    for label_category in df['label'].unique():
        df[df['label'] == label_category]['pktcount'].plot.density(label=label_category, alpha=0.5)
        # You can replace 'pktcount' with 'flows', 'bytecount', etc. to plot other variables

    # Set plot title and labels
    plt.title('Density Plot for Variables by Label')
    plt.xlabel('Values')
    plt.ylabel('Density')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


        
    

        
        


