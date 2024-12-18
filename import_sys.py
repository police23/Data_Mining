from itertools import combinations
import math
import sys
from tkinter.font import Font
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFormLayout, QHBoxLayout, QLineEdit, QWidget, QTabWidget, QVBoxLayout, 
                             QLabel, QStackedWidget, QPushButton, QTextEdit, QFileDialog, QComboBox, QSizePolicy,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QAbstractItemView)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from graphviz import Digraph
from PyQt5.QtGui import QFont
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import defaultdict 
from pprint import pprint

class Apriori:
    def __init__(self, min_support, data):
        self.min_support = min_support
        self.data = data
        self.frequent_itemsets = []


    def generate_frequent_itemsets(self):
        item_counts = defaultdict(int)
        for transaction in self.data:
            for item in transaction:
                item_counts[item] += 1

        # Correctly generate frequent 1-itemsets
        self.frequent_itemsets = [
            {frozenset([item]): count for item, count in item_counts.items() if count >= self.min_support}
        ]

        k = 2
        while True:
            candidate_itemsets = self.generate_candidate_itemsets(self.frequent_itemsets[-1], k)  # Correct candidate generation


            if not candidate_itemsets:  # If no more candidates, stop
                break


            candidate_counts = defaultdict(int)  # Count candidate itemsets
            for transaction in self.data:  # Corrected support counting for k>2
                for itemset in candidate_itemsets:
                    if itemset.issubset(set(transaction)):
                       candidate_counts[itemset] += 1



            frequent_k_itemsets = {itemset: count for itemset, count in candidate_counts.items() if count >= self.min_support} # Check each itemset against min_support

            if not frequent_k_itemsets:  # No frequent itemsets, done
                break


            self.frequent_itemsets.append(frequent_k_itemsets) # Correctly add to self.frequent_itemsets
            k += 1



    def generate_candidate_itemsets(self, frequent_itemsets, k): # Must check if the subset of the candidate is in the frequent itemsets
         candidates = set()
         for itemset1 in frequent_itemsets:
             for itemset2 in frequent_itemsets:

                 union = itemset1.union(itemset2)

                 if len(union) == k:  # Check size after union

                     is_candidate = True
                     for subset in combinations(union, k - 1):  # Generate subsets of size k-1 and check if frequent

                         if frozenset(subset) not in frequent_itemsets:
                             is_candidate = False
                             break

                     if is_candidate:  #If all subsets are frequent, add it to candidates set
                         candidates.add(union)

         return list(candidates)


    def generate_association_rules(self, min_confidence):  # Corrected method
        rules = []
        for k_itemsets in self.frequent_itemsets[1:]:  # Iterate through frequent itemsets starting from k=2
            for itemset, support in k_itemsets.items():  # Get support of the current k-itemsets
                for i in range(1, len(itemset)):       # Iterate through all possible subset sizes for antecedents
                    for antecedent in combinations(itemset, i):   # Generate all possible antecedents

                        consequent = tuple(sorted(set(itemset) - set(antecedent)))  # Calculate the corresponding consequent



                        support_antecedent = 0
                        for transaction in self.data:
                            if set(antecedent).issubset(set(transaction)):
                                 support_antecedent += 1

                        confidence = support / support_antecedent if support_antecedent > 0 else 0 # Correct confidence calculation. Zero check

                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), confidence))

        return rules


class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []

    def fit(self, data):
        n_samples, n_features = data.shape

        # Initialize centroids randomly from data points
        self.centroids = data[np.random.choice(n_samples, self.k, replace=False)]

        # Initialize partition matrix U0
        U = np.zeros((n_samples, self.k))
        for i, point in enumerate(data):
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            cluster_index = np.argmin(distances)
            U[i, cluster_index] = 1

        for iteration in range(self.max_iterations):
            prev_U = U.copy()

            centroids = []
            for j in range(self.k):
                cluster_points_indices = np.where(U[:, j] == 1)[0]
                if cluster_points_indices.size > 0:
                    cluster_points = data[cluster_points_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
                else:
                    centroids.append(self.centroids[j])

            self.centroids = np.array(centroids)


            U = np.zeros((n_samples, self.k))
            for i, point in enumerate(data):
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                cluster_index = np.argmin(distances)
                U[i, cluster_index] = 1

            if np.array_equal(U, prev_U):
                break

        return U, self.centroids


class DecisionTreeGenerator:
    def __init__(self, label, criterion):  # Add criterion parameter
        self.label = label
        self.result_text = None
        self.criterion = criterion  # Store the chosen criterion

    def calc_gini_for_feature(self, data, feature, class_list):
        """Tính toán Gini Index cho một thuộc tính."""
        gini_feature = 0
        total_instances = len(data)
        for value in data[feature].unique():
            subset = data[data[feature] == value]
            gini_subset = 1
            for c in class_list:
                gini_subset -= (len(subset[subset[self.label] == c]) / len(subset))**2 if len(subset) > 0 else 0
            gini_feature += (len(subset) / total_instances) * gini_subset
        return gini_feature
    
    def calc_info_gain_for_feature(self, data, feature, class_list):
        total_instances = len(data)
        entropy_s = 0
        for c in class_list:
            p_c = len(data[data[self.label] == c]) / total_instances
            entropy_s -= p_c * math.log2(p_c) if p_c > 0 else 0

        info_gain = entropy_s
        for value in data[feature].unique():
            subset = data[data[feature] == value]
            subset_entropy = 0
            for c in class_list:
                p_c_subset = len(subset[subset[self.label] == c]) / len(subset) if len(subset) > 0 else 0
                subset_entropy -= p_c_subset * math.log2(p_c_subset) if p_c_subset > 0 else 0
            info_gain -= (len(subset) / total_instances) * subset_entropy

        return info_gain

    def select_best_feature(self, data, class_list, features):  # Modified to accept data and features
        if self.criterion == "gini":
            features_gini = {feature: self.calc_gini_for_feature(data, feature, class_list) for feature in features}
            return min(features_gini, key=features_gini.get) if features_gini else None
        elif self.criterion == "info_gain":
            features_info_gain = {feature: self.calc_info_gain_for_feature(data, feature, class_list) for feature in features}
            return max(features_info_gain, key=features_info_gain.get) if features_info_gain else None
        return None


    def generate_rules(self, dot, parent_name, parent_node, train_data, class_list, current_rule=None, rule_index=1):
        if train_data.empty:
            return rule_index  # Return rule_index if DataFrame is empty

        if len(train_data[self.label].unique()) == 1:  # Pure node (leaf)
            leaf_class = train_data[self.label].iloc[0]
            if current_rule:
                self.result_text.append(f"R{rule_index}: If {current_rule} Then Play={leaf_class}\n")
                rule_index += 1  # Increment after adding rule
            return rule_index  # Return updated rule_index

        if len(train_data.columns) == 1:  # All features used
            if current_rule:
                majority_class = train_data[self.label].mode().iloc[0] # Find majority class if no more features to split on.
                self.result_text.append(f"R{rule_index}: If {current_rule} Then Play={majority_class}\n")
                rule_index += 1 #Increment rule index if rule is generated
            return rule_index

        features = train_data.columns.drop(self.label)
        best_feature = self.select_best_feature(train_data, class_list, features)


        if best_feature:
            for feature_value in train_data[best_feature].unique(): # loop over unique features
                feature_value_data = train_data[train_data[best_feature] == feature_value] # get data for current value of best_feature
                node_name = f"{parent_node}_{feature_value}"

                new_rule = f"{best_feature}={feature_value}"
                if current_rule: # add to existing current rule
                    new_rule = f"{current_rule} AND {new_rule}"

                rule_index = self.generate_rules(dot, node_name, node_name, feature_value_data.drop(columns=best_feature), class_list, new_rule, rule_index) # Update and return rule_index



        elif current_rule:  # No best feature, but have a rule (edge case)

                majority_class = train_data[self.label].mode().iloc[0]
                self.result_text.append(f"R{rule_index}: If {current_rule} Then Play={majority_class}\n")
                rule_index += 1

        return rule_index #Always return rule_index


    def make_tree(self, dot, parent_name, parent_node, train_data, class_list, branch_condition=None):
        if train_data.empty:
            return

        if len(train_data[self.label].unique()) == 1:  # Pure node (single class)
            leaf_class = train_data[self.label].iloc[0]
            dot.node(parent_node, f"{leaf_class}", style='filled', color='lightgray', shape='ellipse')
            return

        if len(train_data.columns) == 1:  # No more features to split on, make leaf node based on majority class
            leaf_class = train_data[self.label].mode().iloc[0]
            dot.node(parent_name, f"{leaf_class}", style='filled', color='lightgray', shape='ellipse')
            return



        features = train_data.columns.drop(self.label)  #Get remaining features
        best_feature = self.select_best_feature(train_data, class_list, features) # Get best features using provided method


        if best_feature:
            dot.node(parent_name, best_feature, style='filled', color='olivedrab1', shape='ellipse')

            for feature_value in train_data[best_feature].unique():  # Iterate through unique feature values of best_feature
                feature_value_data = train_data[train_data[best_feature] == feature_value]
                node_name = f"{parent_node}_{feature_value}"

                current_branch_condition = f"{best_feature} = {feature_value}"
                if branch_condition:
                    current_branch_condition = f"{branch_condition} AND {current_branch_condition}"

                if len(feature_value_data[self.label].unique()) == 1: # Create leaf if pure.
                    leaf_class = feature_value_data[self.label].iloc[0]
                    dot.node(node_name, f"{leaf_class}", style='filled', color='lightgray', shape='ellipse')

                    dot.edge(parent_name, node_name, label=feature_value) # Edge to leaf node
                    continue  # Skip to next feature value


                if len(feature_value_data.columns) > 1:  # If more attributes available, calculate Gini/InfoGain

                    next_features = feature_value_data.drop(columns=best_feature).columns.drop(self.label)  #Drop best feature and class label

                    if self.criterion == "gini":  # Calculate and output Gini for remaining features if more than 1 column exists.

                       self.result_text.append(f"Xét {current_branch_condition}:") # Indicate condition for current node
                       gini_values = {feature: self.calc_gini_for_feature(feature_value_data, feature, class_list)
                                       for feature in next_features}

                       if next_features.empty:
                          majority_label = feature_value_data[self.label].mode().iloc[0] if not feature_value_data.empty else None

                          if majority_label is not None:

                               dot.node(node_name, f"{majority_label}", style="filled", color="lightgrey", shape="ellipse") #Create the leaf node with the majority class label
                               dot.edge(parent_name, node_name, label = feature_value) # Create the edge to the leaf node
                          continue  # Skip the loop and do not make a recursive call if there are no features to calculate the gini indices.
                          

                       for feature, gini in gini_values.items():
                            self.result_text.append(f"  Gini({feature}) = {gini:.4f}")

                       next_best_feature = self.select_best_feature(feature_value_data, class_list, next_features)  # Calculate the next best feature
                       self.result_text.append(f"=> Chọn {next_best_feature} làm thuộc tính phân nhánh\n") # Append to output

                    elif self.criterion == "info_gain":  # Calculate and output info gain for remaining features if more than 1 column exists.
                         self.result_text.append(f"Xét {current_branch_condition}:")

                         info_gain_values = {
                              feature: self.calc_info_gain_for_feature(feature_value_data, feature, class_list)
                              for feature in next_features
                         }
                         if next_features.empty: # Check to make sure a feature exists for calculation
                                  majority_label = feature_value_data[self.label].mode().iloc[0] if not feature_value_data.empty else None # Determine majority class

                                  if majority_label is not None:

                                        dot.node(node_name, f"{majority_label}", style="filled", color="lightgrey", shape="ellipse") # Create leaf node and edge
                                        dot.edge(parent_name, node_name, label=feature_value)
                                  continue # No features to calculate InfoGain values for, do not recurse.



                         for feature, info_gain in info_gain_values.items():

                              self.result_text.append(f"  InfoGain({feature}) = {info_gain:.4f}")

                         next_best_feature = self.select_best_feature(feature_value_data, class_list, next_features)
                         self.result_text.append(f"=> Chọn {next_best_feature} làm thuộc tính phân nhánh\n")

                dot.node(node_name, "?", shape='ellipse')  # Node for next level
                dot.edge(parent_name, node_name, label=feature_value)  # Edge to next level

                self.make_tree(dot, node_name, node_name, feature_value_data.drop(columns=best_feature), class_list, current_branch_condition)  # Recursive call
class NaiveBayes: 
    def __init__(self, label):
        self.label = label
        self.probabilities = {}  # Store conditional probabilities P(X|Y)
        self.class_priors = {}    # Store class priors P(Y)

    def train(self, data):
        class_counts = data[self.label].value_counts()  # Số lượng nhãn
        total_instances = len(data)  # Tổng số mẫu


        # Calculate class priors
        for class_value in class_counts.index:  # Corrected: Iterate through class values
            self.class_priors[class_value] = class_counts[class_value] / total_instances




        # Calculate feature probabilities (P(X|Y))
        self.feature_probs = {}  # Use feature_probs to store probabilities
        for feature in data.columns.drop(self.label):
            self.feature_probs[feature] = {}  # Initialize dictionaries
            for class_value in class_counts.index:
                 self.feature_probs[feature][class_value] = {} # Initialize nested dict
                 class_data = data[data[self.label] == class_value]
                 feature_counts = class_data[feature].value_counts()
                 for feature_value in feature_counts.index:

                      self.feature_probs[feature][class_value][feature_value] = feature_counts[feature_value] / class_counts[class_value]


    def predict(self, instance):
        # Same as NaiveBayesLaplace.predict, using self.feature_probs
        predictions = {}

        for class_value in self.class_priors:
            probability = self.class_priors[class_value]

            for feature, feature_value in instance.items():
                if feature in self.feature_probs:
                     probability *= self.feature_probs[feature][class_value].get(feature_value, 0) # Use .get to handle missing values

            predictions[class_value] = probability


        return max(predictions, key=predictions.get) if predictions else None

class NaiveBayesLaplace:
    def __init__(self, label):
        self.label = label  # Nhãn phân lớp
        self.class_priors = {}  # Xác suất tiên nghiệm P(Y)
        self.feature_probs = {}  # Xác suất có điều kiện P(X|Y)

    def train(self, data):
        class_counts = data[self.label].value_counts()  # Số lượng mỗi nhãn
        total_instances = len(data)  # Tổng số dòng dữ liệu
        num_classes = len(class_counts)  # Số lượng nhãn

       
        for class_value in class_counts.index:
            self.class_priors[class_value] = (class_counts[class_value] + 1) / (total_instances + num_classes)

        self.feature_probs = {}
        for feature in data.columns.drop(self.label): 
            self.feature_probs[feature] = {}

            for class_value in class_counts.index:
                # Lọc dữ liệu theo class_value
                class_data = data[data[self.label] == class_value]
                feature_counts = class_data[feature].value_counts()  
                unique_values = data[feature].unique() 
                num_feature_values = len(unique_values)

                self.feature_probs[feature][class_value] = {}
                for feature_value in unique_values:
                    # Áp dụng Laplace smoothing
                    count = feature_counts.get(feature_value, 0)
                    self.feature_probs[feature][class_value][feature_value] = (count + 1) / (
                        class_counts[class_value] + num_feature_values
                    )

    def predict(self, instance):
        predictions = {}

        for class_value in self.class_priors:
            probability = self.class_priors[class_value]
            for feature, feature_value in instance.items():
                if feature in self.feature_probs:  # Check if feature is in the dictionary
                     probability *= self.feature_probs[feature][class_value].get(feature_value, 1 / (sum(self.feature_probs[feature][class_value].values()) + len(self.class_priors))) # Access feature_probs correctly. Smoothing

            predictions[class_value] = probability

        # Return the class with the highest probability
        return max(predictions, key=predictions.get) if predictions else None
 
class RoughSetAnalyzer:
    def __init__(self, data):
        self.data = data
        self.label = data.columns[-1]  # Add label attribute
        self.result_text = []  # Initialize result_text as an empty list

    def approximate(self, X, B):
        X_indices = self.data[self.data['O'].isin(X)].index
        lower_approximation = []
        upper_approximation = []

        for index in X_indices:
            instance = self.data.loc[index, B]
            matching_indices = self.data[self.data[B].eq(instance).all(axis=1)].index
            approx_names = self.data.loc[matching_indices, 'O'].tolist()

            if set(approx_names).issubset(set(X)): 
                lower_approximation.extend(approx_names) 
            upper_approximation.extend(approx_names)  


        return list(set(lower_approximation)), list(set(upper_approximation))

    def dependency(self, C, B):
        try:
            first_column_name = self.data.columns[0]
            U = set(self.data[first_column_name])
        except IndexError:
            return "Lỗi: Tập dữ liệu trống hoặc không hợp lệ."

        # Calculate lower approximations for each value of C (Corrected)
        lower_approximations = []
        for c_val in self.data[C].unique(): 
           lower_approx_c = [] 
           for x in U:
               c_x = set(self.data[self.data[B].eq(self.data.set_index(first_column_name).loc[x, B]).all(axis=1)][first_column_name])

               if c_x.issubset(self.data[self.data[C] == c_val][first_column_name].tolist()): 
                   lower_approx_c.extend(list(c_x))



           lower_approximations.append(set(lower_approx_c)) 


        k = sum(len(approx) for approx in lower_approximations) / len(U) if len(U) > 0 else 0

        # Construct the dependency message
        if k == 1:
            return "Phụ thuộc hoàn toàn (k = 1)"
        elif 0 < k < 1:
            return f"Phụ thuộc một phần (k = {k:.2f})"
        else:
            return "Không phụ thuộc (k = 0)"


    def discernibility_matrix(self, decision_attr):  # Create method for computing discernibility matrix
        features = self.data.columns.drop(['O', decision_attr])
        matrix = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data)):
                if self.data.loc[i, decision_attr] != self.data.loc[j, decision_attr]: # different label so include different features
                    diff_features = set()
                    for feature in features:
                         if self.data.loc[i, feature] != self.data.loc[j, feature]:
                             diff_features.add(feature)  # Keep track of different features
                    if diff_features:  # If there is a feature that's different
                          row.append(diff_features)
                else: # Same label so no need to consider
                    row.append(None)

            matrix.append(row)
        return matrix


    def reducts(self, decision_attr):
        matrix = self.discernibility_matrix(decision_attr)  # Compute matrix
        discernibility_function = set()  # Use a set to ensure unique clauses
        
        for row in matrix:  # Loop through the rows of the discernibility matrix
            for item in row:  # Loop through items in the rows
                if item:
                    discernibility_function.add(frozenset(item))  # Use frozenset for immutability and uniqueness

        if not discernibility_function:  # Handle cases with no reducts
            return []

        # Simplify discernibility function (using absorption law)
        simplified_function = list(discernibility_function)  # Make a copy

        simplified = True  # Start with True to enter the loop
        while simplified:
            simplified = False  # Set it to False for each iteration of while loop
            new_simplified = []
            for clause1 in simplified_function:
                absorbed = False
                for clause2 in simplified_function:
                    if clause1 != clause2 and clause2.issubset(clause1): 
                        print(f'{clause1} bị hấp thụ')
                        absorbed = True  # Clause1 is absorbed so no need to add it to the output
                        break  # Clause1 absorbed by clause 2, exit the inner loop
                if not absorbed:
                    new_simplified.append(clause1)  # If clause1 is not absorbed, it is part of the reduct
            
            if set(simplified_function) != set(new_simplified):  # If changes made, update function and continue
                simplified_function = new_simplified  # Update the simplified_function
                simplified = True  # Changes made, so continue the loop

        # Sort the final simplified function to ensure consistent results
        sorted_simplified_function = sorted([sorted(list(clause)) for clause in simplified_function])

        # Ensure the output is always the same by sorting the final list of lists
        sorted_simplified_function.sort(key=lambda x: (len(x), x))

        return sorted_simplified_function  # Return the final simplified function as sorted list of lists

    def generate_rules(self, dot, parent_name, parent_node, train_data, class_list, current_rule=None, rule_index=1):
        if train_data.empty:
            return rule_index

        if len(train_data[self.label].unique()) == 1:
            leaf_class = train_data[self.label].iloc[0]
            if current_rule:
                self.result_text.append(f"{current_rule} --> {self.label}={leaf_class}\n")
                rule_index += 1
            return rule_index

        if len(train_data.columns) == 1:
            if current_rule:
                majority_class = train_data[self.label].mode().iloc[0]
                self.result_text.append(f"{current_rule} --> {self.label}={majority_class}\n")
                rule_index += 1
            return rule_index

        features = train_data.columns.drop([self.label, train_data.columns[0]])  # Exclude the first column and the label
        best_feature = self.select_best_feature(train_data, class_list, features)

        if best_feature:
            for feature_value in train_data[best_feature].unique():
                feature_value_data = train_data[train_data[best_feature] == feature_value]
                node_name = f"{parent_node}_{feature_value}"

                new_rule = f"{best_feature}={feature_value}"
                if current_rule:
                    new_rule = f"{current_rule} và {new_rule}"

                rule_index = self.generate_rules(dot, node_name, node_name, feature_value_data.drop(columns=best_feature), class_list, new_rule, rule_index)

        elif current_rule:
            majority_class = train_data[self.label].mode().iloc[0]
            self.result_text.append(f"{current_rule} --> {self.label}={majority_class}\n")
            rule_index += 1

        return rule_index

    def calculate_raw(self):
        try:
            self.raw_result_text_approx.clear()
            self.raw_result_text_dependency.clear()
            self.raw_result_text_reducts.clear()
            self.raw_result_text_rules.clear()

            selected_X = [item.text() for item in self.X_list.selectedItems()]
            selected_B = [item.text() for item in self.B_list.selectedItems()]
            selected_B1 = [item.text() for item in self.B1_list.selectedItems()]

            if not selected_X or not selected_B or not selected_B1:
                self.raw_result_text_approx.append("Vui lòng chọn X, B và B1.")
                return

            if len(self.raw_data.columns) > 0:
                decision_attribute = self.raw_data.columns[-1]
            else:
                self.raw_result_text_approx.append("Tập dữ liệu trống hoặc không hợp lệ.")
                return

            analyzer = RoughSetAnalyzer(self.raw_data)
            analyzer.result_text = []

            lower, upper = analyzer.approximate(selected_X, selected_B)
            self.raw_result_text_approx.append(f"Xấp xỉ dưới của X qua tập thuộc tính B là: {lower}")
            self.raw_result_text_approx.append(f"Xấp xỉ trên của X qua tập thuộc tính B là: {upper}")
            if len(upper) > 0:
                coeff = len(lower) / len(upper)
                self.raw_result_text_approx.append(f"Hệ số xấp xỉ: {coeff:.2f}")

            dependency_result_msg = analyzer.dependency(decision_attribute, selected_B1)
            self.raw_result_text_dependency.append(f"{dependency_result_msg}")

            reducts_result = analyzer.reducts(decision_attribute)
            if reducts_result:
                distributed_result = []
                single_clause = reducts_result[0]
                multi_clause = reducts_result[1]

                for item in multi_clause:
                    distributed_result.append(f"({single_clause[0]} ∧ {item})")

                distributed_laws = " ∨ ".join(distributed_result)
                self.raw_result_text_reducts.append(distributed_laws)
            else:
                self.raw_result_text_reducts.append("Không tìm thấy reduct nào.")

            dot = Digraph()
            class_list = self.raw_data[decision_attribute].unique()
            analyzer.generate_rules(dot, 'root', 'root', self.raw_data, class_list)

            # Display the rules in the raw_result_text_rules QTextEdit
            self.raw_result_text_rules.append("\n".join(analyzer.result_text))

        except Exception as e:
            self.raw_result_text_approx.append(f"Lỗi: {e}")

    def select_best_feature(self, data, class_list, features):
        # Implement a simple feature selection method, e.g., based on Gini index
        def calc_gini_for_feature(data, feature, class_list):
            gini_feature = 0
            total_instances = len(data)
            for value in data[feature].unique():
                subset = data[data[feature] == value]
                gini_subset = 1
                for c in class_list:
                    gini_subset -= (len(subset[subset[self.label] == c]) / len(subset))**2 if len(subset) > 0 else 0
                gini_feature += (len(subset) / total_instances) * gini_subset
            return gini_feature

        features_gini = {feature: calc_gini_for_feature(data, feature, class_list) for feature in features}
        return min(features_gini, key=features_gini.get) if features_gini else None

class Kohonen:
    def __init__(self, learning_rate, iterations, neighborhood_radius=0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.neighborhood_radius = neighborhood_radius
        self.weights = np.array([
            [19, 111, 21.5],
            [6.5, 88, 90.5],
            [8.5, 18, 65]
        ])

    def train(self, data, result_text):
        result_text.append("Khởi tạo giá trị của các vector trọng số:")
        for i, weight in enumerate(self.weights):
            result_text.append(f"w{i+1}: {[f'{w:.2f}'.rstrip('0').rstrip('.') for w in weight]}")
        
        for iteration in range(self.iterations):
            result_text.append(f"\n*Lần lặp thứ {iteration + 1}:")
            for i, sample in enumerate(data):
                distances = np.linalg.norm(self.weights - sample, axis=1)
                winner_index = np.argmin(distances)
                result_text.append(f" \nXét vector {i + 1} (Tranh {i + 1}) x{i + 1}\n")
                sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
                for j, distance in sorted_distances:
                    if j == winner_index:
                        result_text.append(f" Khoảng cách từ x{i + 1} đến w{j + 1}: {distance:.2f} -> ngắn nhất")
                    else:
                        result_text.append(f" Khoảng cách từ x{i + 1} đến w{j + 1}: {distance:.2f}")
                for j in range(len(self.weights)):
                    if np.linalg.norm(j - winner_index) <= self.neighborhood_radius:
                        self.weights[j] += self.learning_rate * (sample - self.weights[j])
                        self.weights[j] = np.round(self.weights[j], 2)  # Round weights to 2 decimal places
                        result_text.append(f"-> Cập nhật trọng số w{j + 1}: {[f'{w:.2f}'.rstrip('0').rstrip('.') for w in self.weights[j]]}")
            self.learning_rate /= 2
            result_text.append(f"\n**Giảm tốc độ học: alpha = {self.learning_rate}")

    def predict(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=1)
        return np.argmin(distances)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Mining Tools")
        self.setGeometry(100, 100, 900, 600)
        
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
        QTabWidget::pane {
            border-top: 2px solid #C0C0C0;
            position: absolute;
            top: -2px; 
            left: 0px; 
            right: 0px;
            bottom: 0px;
        }
        QTabWidget::tab-bar {
            alignment: left;
        }
        QTabBar::tab {
            background: #E0E0E0;
            border: 1px solid #C0C0C0;
            padding: 8px 20px;
            min-width: 120px;
            font-weight: 16px;
            border-radius : 8px;
        }
        QTabBar::tab:selected {
            background: #2E7D32; 
            color: white;
            border-bottom-color: #2E7D32; 
        }
        QTabBar::tab:hover {
            background: #A7D1AB; 
        }
    """)

        self.stacked_widget = QStackedWidget()

        items = [  # Your tab items
            ("Tiền xử lý dữ liệu", "preprocessing.png"),  # New tab for data preprocessing
            ("Tập phổ biến và luật kết hợp", "book.png"),
            ("Decision Tree", "ID3.png"),
            ("Naive Bayes", "NB.png"),
            ("Tập thô", "raw_data.png"),
            ("K-Means", "cluster.png"),
        ]

        items.append(("Kohonen", "kohonen.png"))  # Add Kohonen tab

        for i, (text, icon_file) in enumerate(items):
            tab = QWidget()
            icon = QIcon(icon_file)
            self.tabs.addTab(tab, icon, text)
            self.tabs.setIconSize(QSize(20, 20))
            
            if i == 0:  # Data Preprocessing Tab
                self.preprocessing_load_button = QPushButton("Upload file .CSV")
                self.preprocessing_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.preprocessing_load_button.clicked.connect(lambda: self.load_data('preprocessing'))

                self.preprocessing_filepath_display = QLineEdit()
                self.preprocessing_filepath_display.setReadOnly(True)
                self.preprocessing_filepath_display.setStyleSheet("font-size: 12pt;")

                self.preprocessing_calculate_button = QPushButton("Tính toán hệ số tương quan")
                self.preprocessing_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.preprocessing_calculate_button.clicked.connect(self.calculate_correlation)

                self.preprocessing_result_text = QTextEdit()
                self.preprocessing_result_text.setReadOnly(True)
                self.preprocessing_result_text.setStyleSheet("font-size: 13pt;")

                preprocessing_layout = QVBoxLayout()
                preprocessing_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                preprocessing_layout.addWidget(self.preprocessing_load_button)
                preprocessing_layout.addWidget(self.preprocessing_filepath_display)
                preprocessing_layout.addWidget(self.preprocessing_calculate_button)
                preprocessing_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                preprocessing_layout.addWidget(self.preprocessing_result_text, stretch=1)

                tab.setLayout(preprocessing_layout)

            if i == 1:  # Apriori Tab
                self.apriori_load_button = QPushButton("Upload file .CSV")
                self.apriori_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.apriori_load_button.clicked.connect(lambda: self.load_data('apriori')) # Connect

                self.apriori_calculate_button = QPushButton("Tính toán") #Apriori button
                self.apriori_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.apriori_calculate_button.clicked.connect(self.run_apriori) # connect

                self.apriori_result_text = QTextEdit()
                self.apriori_result_text.setReadOnly(True)
                self.apriori_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt; 
                    }
                """)

                self.min_support_input = QDoubleSpinBox()
                self.min_support_input.setRange(0.0, 1.0)
                self.min_support_input.setSingleStep(0.01)

                self.min_confidence_input = QDoubleSpinBox()
                self.min_confidence_input.setRange(0.0, 1.0)
                self.min_confidence_input.setSingleStep(0.01)
                self.apriori_calculate_button = QPushButton("Tính toán")
                self.apriori_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.apriori_calculate_button.clicked.connect(self.run_apriori)

                self.frequent_itemsets_text = QTextEdit() # Create QTextEdits *before* adding to layout
                self.frequent_itemsets_text.setReadOnly(True)
                self.frequent_itemsets_text.setFixedHeight(90)  # Set fixed height for two lines
                self.frequent_itemsets_text.setStyleSheet("font-size: 13pt;")


                self.maximal_itemsets_text = QTextEdit()
                self.maximal_itemsets_text.setReadOnly(True)
                self.maximal_itemsets_text.setFixedHeight(90)  # Set fixed height for two lines
                self.maximal_itemsets_text.setStyleSheet("font-size: 13pt;")



                self.association_rules_text = QTextEdit()
                self.association_rules_text.setReadOnly(True)
                self.association_rules_text.setStyleSheet("font-size: 13pt;")
                self.apriori_filepath_display = QLineEdit() 
                self.apriori_filepath_display.setReadOnly(True)
                self.apriori_filepath_display.setStyleSheet("font-size: 13pt;")

                apriori_layout = QVBoxLayout()
                apriori_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.apriori_load_button)
                apriori_layout.addWidget(self.apriori_filepath_display)

                #Inputs for Apriori
                input_layout = QHBoxLayout() # Horizontal layout for inputs
                min_support_layout = QVBoxLayout() # Vertical for min support
                min_support_layout.addWidget(QLabel("Min Support:", font=QFont("Arial", 12)))
                min_support_layout.addWidget(self.min_support_input)
                self.min_support_input.setStyleSheet("QDoubleSpinBox { font-size: 12pt;  height: 30px;}")

                input_layout.addLayout(min_support_layout)
                min_confidence_layout = QVBoxLayout() # Vertical for min confidence
                min_confidence_layout.addWidget(QLabel("Min Confidence:", font=QFont("Arial", 12)))
                min_confidence_layout.addWidget(self.min_confidence_input)
                self.min_confidence_input.setStyleSheet("QDoubleSpinBox { font-size: 12pt; height: 30px; }")
                input_layout.addLayout(min_confidence_layout)
                apriori_layout.addLayout(input_layout) # Now add to main layout
                apriori_layout.addWidget(self.apriori_calculate_button)

                apriori_layout.addWidget(QLabel("Các tập phổ biến thỏa ngưỡng:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.frequent_itemsets_text)  # No stretch needed

                apriori_layout.addWidget(QLabel("Tập phổ biến tối đại:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.maximal_itemsets_text)  # No stretch needed
                apriori_layout.addWidget(QLabel("Các luật kết hợp:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.association_rules_text, stretch=1)
                tab.setLayout(apriori_layout)


            if i == 2:  # ID3 Tab
                self.id3_load_button = QPushButton("Upload file .CSV")
                self.id3_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.id3_load_button.clicked.connect(lambda: self.load_data('decision_tree'))

                self.dt_filepath_display = QLineEdit() 
                self.dt_filepath_display.setReadOnly(True)
                self.dt_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt; 
                    }
                """)

                self.id3_result_text_nodes = QTextEdit()
                self.id3_result_text_nodes.setReadOnly(True)
                self.id3_result_text_nodes.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt; /* Increased text edit font size */
                    }
                """)

                self.id3_result_text_rules = QTextEdit()
                self.id3_result_text_rules.setReadOnly(True)
                self.id3_result_text_rules.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt; /* Increased text edit font size */
                    }
                """)

                # Criterion combobox *CREATED FIRST*
                self.criterion_combobox = QComboBox()
                self.criterion_combobox.addItem("Chỉ số Gini", "gini")
                self.criterion_combobox.addItem("Độ lợi thông tin", "info_gain")
                self.criterion_combobox.setStyleSheet("""
                    QComboBox {
                        font-size: 13pt;  /* Increased combobox font size */
                        margin-bottom: 15px;
                    }
                    QComboBox QAbstractItemView { /* ... */ }  # Dropdown styles
                """)
                self.dt_calculate_button = QPushButton("Tính toán")  # The calculate button
                self.dt_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.dt_calculate_button.clicked.connect(self.calculate_decision_tree)

                dt_layout = QVBoxLayout()
                dt_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                dt_layout.addWidget(self.id3_load_button)
                dt_layout.addWidget(self.dt_filepath_display) # Filepath display
                dt_layout.addWidget(QLabel("Chọn cách tính:", font=QFont("Arial", 12)))
                dt_layout.addWidget(self.criterion_combobox)

                dt_layout.addWidget(self.dt_calculate_button) # Add calculate button here

                dt_layout.addWidget(QLabel("Xét các nút chọn để phân nhánh:", font=QFont("Arial", 12)))
                result_container_nodes = QWidget()
                result_layout_nodes = QVBoxLayout(result_container_nodes)
                result_layout_nodes.addWidget(self.id3_result_text_nodes)
                result_layout_nodes.setContentsMargins(0, 0, 0, 0)  # Remove margins
                result_layout_nodes.setSpacing(0)  # Remove spacing within container
                result_container_nodes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text_nodes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text_nodes.setMinimumSize(0,0)  # Ensure no minimum size restriction on text edit

                dt_layout.addWidget(result_container_nodes, stretch=1) # stretch on the container

                dt_layout.addWidget(QLabel("Các luật được tạo:", font=QFont("Arial", 12)))
                result_container_rules = QWidget()
                result_layout_rules = QVBoxLayout(result_container_rules)
                result_layout_rules.addWidget(self.id3_result_text_rules)
                result_layout_rules.setContentsMargins(0, 0, 0, 0)  # Remove margins
                result_layout_rules.setSpacing(0)  # Remove spacing within container
                result_container_rules.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text_rules.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text_rules.setMinimumSize(0,0)  # Ensure no minimum size restriction on text edit

                dt_layout.addWidget(result_container_rules, stretch=1) # stretch on the container

                tab.setLayout(dt_layout)

            
            elif i == 3: 
                self.nb_load_button = QPushButton("Upload file .CSV")
                self.nb_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;    /* Rounded corners */
                        font-size : 20px;      
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.nb_result_text = QTextEdit()
                self.nb_result_text.setReadOnly(True)
                self.nb_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt;
                    }
                """)
                self.nb_filepath_display = QLineEdit()
                self.nb_filepath_display.setReadOnly(True)
                self.nb_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt;
                } 
                """)
                self.laplace_smoothing_checkbox = QCheckBox("Làm trơn Laplace") 
                self.laplace_smoothing_checkbox.setStyleSheet("""
                    QCheckBox {
                        font-size: 12pt;
                    }
                """)
                self.feature_comboboxes = {}
                
                self.predict_button = QPushButton("Tính toán")
                self.predict_button.clicked.connect(self.predict_laplace)
                self.predict_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;    /* Rounded corners */
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)

                nb_layout = QVBoxLayout()
                nb_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                nb_layout.addWidget(self.nb_load_button)
                nb_layout.addWidget(self.nb_filepath_display)
                nb_layout.addWidget(self.laplace_smoothing_checkbox)
                
    
                form_layout = QFormLayout()  # Form layout for dynamic comboboxes
                nb_layout.addWidget(QLabel("Chọn giá trị thuộc tính:", font=QFont("Arial", 12)))                
                nb_layout.addLayout(form_layout) 
                nb_layout.addWidget(self.predict_button)
    
                nb_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                
    
                result_container = QWidget()
                result_layout = QVBoxLayout(result_container)
                result_layout.addWidget(self.nb_result_text)
                result_layout.setContentsMargins(0, 0, 0, 0)
                result_layout.setSpacing(0)
                result_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.nb_result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Set size policy
                self.nb_result_text.setMinimumSize(0, 0)
    
                nb_layout.addWidget(result_container, stretch=1) # stretch=1 for result container
    
                self.nb_prediction_label = QLabel("Mẫu X được phân vào lớp:")
                self.nb_prediction_label.setFont(QFont("Arial", 14))  # Increased font size
                prediction_layout = QHBoxLayout()
                prediction_layout.addWidget(self.nb_prediction_label)
                nb_layout.addLayout(prediction_layout) 
    
                tab.setLayout(nb_layout)
    
    
                self.nb_load_button.clicked.connect(lambda: self.load_data('naive_bayes'))
            elif i == 4:
                self.raw_load_button = QPushButton("Upload file .CSV")
                self.raw_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;    /* Rounded corners */
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.raw_filepath_display = QLineEdit() 
                self.raw_filepath_display.setReadOnly(True)
                self.raw_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt;
                    }
                """)
                self.raw_result_text_approx = QTextEdit()
                self.raw_result_text_approx.setReadOnly(True)
                self.raw_result_text_approx.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt;
                    }
                """)
                self.raw_result_text_dependency = QTextEdit()
                self.raw_result_text_dependency.setReadOnly(True)
                self.raw_result_text_dependency.setFixedHeight(60)  # Set fixed height for two lines
                self.raw_result_text_dependency.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt;
                    }
                """)
                self.raw_result_text_reducts = QTextEdit()
                self.raw_result_text_reducts.setReadOnly(True)
                self.raw_result_text_reducts.setFixedHeight(60)  # Set fixed height for two lines
                self.raw_result_text_reducts.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt;
                    }
                """)
                self.raw_result_text_rules = QTextEdit()  # New QTextEdit for rules with 100% accuracy
                self.raw_result_text_rules.setReadOnly(True)
                self.raw_result_text_rules.setStyleSheet("""
                    QTextEdit {
                        font-size: 13pt;
                    }
                """)
                self.X_list = QListWidget()
                self.X_list.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Allow multiple selections
                self.B_list = QListWidget()
                self.B_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
                self.B1_list = QListWidget()
                self.B1_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
                self.C_combobox = QComboBox()

                self.raw_calculate_button = QPushButton("Tính toán") # button to trigger
                self.raw_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;    /* Rounded corners */
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)

                raw_layout = QVBoxLayout()
                
                self.decision_attr_display = QLineEdit() 
                self.decision_attr_display.setReadOnly(True) 
                self.decision_attr_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt;
                    }
                """)
                # Add widgets to the layout
                raw_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_load_button)
                raw_layout.addWidget(self.raw_filepath_display)  

                selection_layout = QHBoxLayout()

                # Left side (X selection)
                left_layout = QVBoxLayout()
                left_layout.addWidget(QLabel("Chọn tập X:", font=QFont("Arial", 12)))
                left_layout.addWidget(self.X_list, stretch=1)  # Stretch X_list vertically
                selection_layout.addLayout(left_layout)


                # Right side (B selection)
                right_layout = QVBoxLayout()
                right_layout.addWidget(QLabel("Chọn tập B:", font=QFont("Arial", 12)))
                right_layout.addWidget(self.B_list, stretch=1)  # Stretch B_list vertically
                selection_layout.addLayout(right_layout)

                # Right side (B1 selection)
                right_layout_B1 = QVBoxLayout()
                right_layout_B1.addWidget(QLabel("Chọn tập B1:", font=QFont("Arial", 12)))
                right_layout_B1.addWidget(self.B1_list, stretch=1)  # Stretch B1_list vertically
                selection_layout.addLayout(right_layout_B1)

                raw_layout.addLayout(selection_layout)


                # C selection
                raw_layout.addWidget(QLabel("Thuộc tính quyết định:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.decision_attr_display)



                raw_layout.addWidget(self.raw_calculate_button)  # Trigger button

                raw_layout.addWidget(QLabel("Xấp xỉ dưới, xấp xỉ trên, hệ số xấp xỉ của X qua tập thuộc tính B:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_result_text_approx, stretch=1) # Set result area to stretch

                raw_layout.addWidget(QLabel("Khảo sát sự phụ thuộc thuộc tính giữa thuộc tính quyết định và tập thuộc tính B1:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_result_text_dependency)  # No stretch needed

                raw_layout.addWidget(QLabel("Rút gọn của hệ quyết định:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_result_text_reducts)  # No stretch needed

                raw_layout.addWidget(QLabel("Một số luật có độ chính xác 100%:", font=QFont("Arial", 12)))  # Label for new section
                raw_layout.addWidget(self.raw_result_text_rules, stretch=1)  # Add new QTextEdit to layout

                tab.setLayout(raw_layout)

                self.raw_load_button.clicked.connect(lambda: self.load_data('raw'))
                self.raw_calculate_button.clicked.connect(self.calculate_raw)

            elif i == 5: 
                self.kmeans_load_button = QPushButton("Upload file .CSV")
                self.kmeans_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                       
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.kmeans_filepath_display = QLineEdit() 
                self.kmeans_filepath_display.setReadOnly(True)
                self.kmeans_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt;
                    }
                """)
                self.k_input = QSpinBox() # Use a spinbox
                self.k_input.setMinimum(1)  # Set minimum value (k must be at least 1)
                self.k_input.setValue(2) # default value
                self.k_input.setStyleSheet("font-size: 12pt;") # Increase font size
                self.kmeans_calculate_button = QPushButton("Tính toán")
                self.kmeans_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                       
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)


                self.kmeans_result_text = QTextEdit()
                self.kmeans_result_text.setReadOnly(True)
                self.kmeans_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 14pt;
                    }
                """)

                self.kmeans_cluster_text = QTextEdit()
                self.kmeans_cluster_text.setReadOnly(True)
                self.kmeans_cluster_text.setFixedHeight(90) 
                self.kmeans_cluster_text.setStyleSheet("font-size: 13pt;")
              

                kmeans_layout = QVBoxLayout()
                kmeans_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.kmeans_load_button)
                kmeans_layout.addWidget(self.kmeans_filepath_display)
                
                k_input_layout = QHBoxLayout()  # Create a horizontal layout for the k input
                k_input_label = QLabel("Nhập k:", font=QFont("Arial", 12))
                k_input_label.setFixedWidth(100)  # Set a fixed width for the label
                k_input_layout.addWidget(k_input_label)
                self.k_input.setFixedWidth(200)  # Set a fixed width for the input field
                k_input_layout.addWidget(self.k_input)
                k_input_layout.addStretch()  # Add a stretch to push the input to the left
                k_input_layout.setContentsMargins(0, 0, 50, 0)
               
                kmeans_layout.addLayout(k_input_layout)  # Add the horizontal layout to the main layout
                
                kmeans_layout.addWidget(self.kmeans_calculate_button)
                kmeans_layout.addWidget(QLabel("Kết quả tính toán:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.kmeans_result_text, stretch=1)
                kmeans_layout.addWidget(QLabel("Kết quả phân cụm:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.kmeans_cluster_text, stretch=1)

                tab.setLayout(kmeans_layout)

                self.kmeans_load_button.clicked.connect(lambda: self.load_data('kmeans'))
                self.kmeans_calculate_button.clicked.connect(self.run_kmeans)
           
            if i == len(items) - 1:  # Kohonen Tab
                self.kohonen_load_button = QPushButton("Upload file .CSV")
                self.kohonen_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.kohonen_load_button.clicked.connect(lambda: self.load_data('kohonen'))

                self.kohonen_filepath_display = QLineEdit()
                self.kohonen_filepath_display.setReadOnly(True)
                self.kohonen_filepath_display.setStyleSheet("font-size: 12pt;")

                self.learning_rate_input = QDoubleSpinBox()
                self.learning_rate_input.setRange(0.0, 1.0)
                self.learning_rate_input.setSingleStep(0.01)
                self.learning_rate_input.setStyleSheet("font-size: 12pt; height: 30px;")

                self.iterations_input = QSpinBox()
                self.iterations_input.setRange(1, 1000)
                self.iterations_input.setStyleSheet("font-size: 12pt; height: 30px;")

                self.neighborhood_radius_input = QSpinBox()
                self.neighborhood_radius_input.setRange(0, 10)
                self.neighborhood_radius_input.setStyleSheet("font-size: 12pt; height: 30px;")

                self.kohonen_calculate_button = QPushButton("Tính toán")
                self.kohonen_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 10px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.kohonen_calculate_button.clicked.connect(self.run_kohonen)

                self.kohonen_result_text = QTextEdit()
                self.kohonen_result_text.setReadOnly(True)
                self.kohonen_result_text.setStyleSheet("font-size: 13pt;")

                self.kohonen_cluster_text = QTextEdit()
                self.kohonen_cluster_text.setReadOnly(True)
                self.kohonen_cluster_text.setFixedHeight(110)  # Set fixed height for three lines
                self.kohonen_cluster_text.setStyleSheet("font-size: 13pt;")

                kohonen_layout = QVBoxLayout()
                kohonen_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                kohonen_layout.addWidget(self.kohonen_load_button)
                kohonen_layout.addWidget(self.kohonen_filepath_display)

                input_layout = QHBoxLayout()
                learning_rate_layout = QVBoxLayout()
                learning_rate_layout.addWidget(QLabel("Tốc độ học:", font=QFont("Arial", 12)))
                learning_rate_layout.addWidget(self.learning_rate_input)
                input_layout.addLayout(learning_rate_layout)

                iterations_layout = QVBoxLayout()
                iterations_layout.addWidget(QLabel("Số lần lặp:", font=QFont("Arial", 12)))
                iterations_layout.addWidget(self.iterations_input)
                input_layout.addLayout(iterations_layout)

                neighborhood_radius_layout = QVBoxLayout()
                neighborhood_radius_layout.addWidget(QLabel("Bán kính vùng lân cận:", font=QFont("Arial", 12)))
                neighborhood_radius_layout.addWidget(self.neighborhood_radius_input)
                input_layout.addLayout(neighborhood_radius_layout)

                kohonen_layout.addLayout(input_layout)
                kohonen_layout.addWidget(self.kohonen_calculate_button)
                kohonen_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                kohonen_layout.addWidget(self.kohonen_result_text, stretch=1)
                kohonen_layout.addWidget(QLabel("Ảnh thuộc cụm:", font=QFont("Arial", 12)))
                kohonen_layout.addWidget(self.kohonen_cluster_text, stretch=1)

                tab.setLayout(kohonen_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) #Remove margins on main layout if any

        main_layout.addWidget(self.tabs)

        # If using stacked widget:
        main_layout.addWidget(self.stacked_widget)  # Make sure to remove any stretch factors here
        self.tabs.currentChanged.connect(self.display_content)

        # Ensure MainWindow allows resizing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def load_data(self, tab_name):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filepath:
            try:
                if tab_name == 'preprocessing':
                    self.preprocessing_filepath_display.setText(filepath)
                    self.preprocessing_data = pd.read_csv(filepath)
                elif tab_name == 'apriori':
                    self.apriori_filepath_display.setText(filepath)
                    self.apriori_data = []
                    df = pd.read_csv(filepath)
                    for transaction_id in df['Mã hóa đơn'].unique():
                        transaction = df[df['Mã hóa đơn'] == transaction_id]['Mã hàng'].tolist()
                        self.apriori_data.append(transaction)
                elif tab_name == 'raw':
                    self.raw_filepath_display.setText(filepath)
                    self.raw_data = pd.read_csv(filepath)
                    if 'Day' in self.raw_data.columns:
                        self.raw_data = self.raw_data.drop(columns=['Day'])
                    self.X_list.clear()
                    self.X_list.addItems(self.raw_data.iloc[:, 0].unique().astype(str))
                    self.B_list.clear()
                    self.B1_list.clear()
                    if len(self.raw_data.columns) > 0:
                        decision_attribute = self.raw_data.columns[-1]
                        self.decision_attr_display.setText(decision_attribute)
                        columns_to_add = self.raw_data.columns.drop(self.raw_data.columns[0]).drop(decision_attribute).tolist()
                        self.B_list.addItems(columns_to_add)
                        self.B1_list.addItems(columns_to_add)
                elif tab_name == 'naive_bayes':
                    self.nb_filepath_display.setText(filepath)
                    self.nb_result_text.clear()
                    self.data = pd.read_csv(filepath)
                    if 'Day' in self.data.columns:
                        self.data = self.data.drop(columns=['Day'])
                    self.label = 'Play'
                    form_layout = self.findChild(QFormLayout)
                    for i in reversed(range(form_layout.count())):
                        form_layout.itemAt(i).widget().setParent(None)
                    self.feature_comboboxes = {}
                    for feature in self.data.columns.drop(self.label):
                        combobox = QComboBox()
                        combobox.addItems(self.data[feature].unique().astype(str))
                        combobox.setStyleSheet("""
                            QComboBox {
                                font-size: 12pt;        
                                height: 35px;          
                                padding: 3px 10px;      
                                min-width: 150px; 
                            }
                            QComboBox QAbstractItemView {
                                font-size: 12pt;
                            }
                        """)
                        self.feature_comboboxes[feature] = combobox
                        form_layout.addRow(QLabel(f"{feature}:", font=QFont("Arial", 12)), combobox)
                elif tab_name == 'kmeans':
                    self.kmeans_filepath_display.setText(filepath)
                    self.kmeans_result_text.clear()
                    self.kmeans_cluster_text.clear()
                    self.kmeans_data = pd.read_csv(filepath)
                    if 'X' in self.kmeans_data.columns:
                        self.kmeans_data = self.kmeans_data.drop(columns=['X'])
                    self.kmeans_data = self.kmeans_data.apply(pd.to_numeric, errors='coerce').dropna()
                    self.kmeans_data = self.kmeans_data.values
                elif tab_name == 'decision_tree':
                    self.dt_filepath_display.setText(filepath)
                    self.dt_data = pd.read_csv(filepath)
                    if 'Day' in self.dt_data.columns:
                        self.dt_data = self.dt_data.drop(columns=['Day'])
                elif tab_name == 'kohonen':
                    self.kohonen_filepath_display.setText(filepath)
                    self.kohonen_data = pd.read_csv(filepath).iloc[:, 1:].values
            except Exception as e:
                if tab_name == 'preprocessing':
                    self.preprocessing_result_text.append(f"Lỗi khi tải dữ liệu: {e}")
                elif tab_name == 'apriori':
                    self.apriori_result_text.append(f"Lỗi khi tải dữ liệu: {e}")
                elif tab_name == 'raw':
                    self.raw_result_text_approx.append(f"Lỗi: {e}")
                elif tab_name == 'naive_bayes':
                    self.nb_result_text.append(f"Lỗi: {e}")
                elif tab_name == 'kmeans':
                    self.kmeans_result_text.append(f"Lỗi khi tải dữ liệu: {e}")
                elif tab_name == 'decision_tree':
                    self.id3_result_text.append(f"Lỗi: {e}")
                elif tab_name == 'kohonen':
                    self.kohonen_result_text.append(f"Lỗi: {e}")

    def run_apriori(self):
        try:
            # Clear all result areas
            self.frequent_itemsets_text.clear()
            self.maximal_itemsets_text.clear()
            self.association_rules_text.clear()


            try:
                min_sup = float(self.min_support_input.value())
                min_conf = float(self.min_confidence_input.value())
            except ValueError:
                self.apriori_result_text.append("Lỗi: Min Support và Min Confidence phải là số.")
                return

            if not hasattr(self, 'apriori_data'):
                self.apriori_result_text.append("Lỗi: Vui lòng tải dữ liệu trước.")
                return



            apriori = Apriori(min_sup * len(self.apriori_data), self.apriori_data) # Multiply min_sup by num of transactions
            apriori.generate_frequent_itemsets()

            # Frequent Itemsets output
            frequent_itemsets_output = "{"
            for k_itemsets in apriori.frequent_itemsets:
                 for itemset, support in k_itemsets.items():
                    frequent_itemsets_output += str(set(itemset)) + ", "


            frequent_itemsets_output = frequent_itemsets_output[:-2] + "}" # Remove trailing comma and space
            self.frequent_itemsets_text.append(frequent_itemsets_output)


            # Maximal frequent itemsets
            maximal_itemsets = []
            for k_itemsets in reversed(apriori.frequent_itemsets): # Correct iteration through frequent itemsets starting with the largest k
                for itemset in k_itemsets:
                     is_maximal = True
                     for other_k_itemsets in reversed(apriori.frequent_itemsets): # Iterating through all other itemsets including the itemsets with same k
                         for other_itemset in other_k_itemsets:
                                if k_itemsets != other_k_itemsets and set(itemset).issubset(set(other_itemset)): # If current itemset is a subset of different itemsets at different k level, it is not maximal
                                     is_maximal = False # Set is_maximal to False because itemset is not maximal.
                                     break  # No need to check others
                         if not is_maximal: # if itemset is found not maximal, break out of the loop
                                 break
                     if is_maximal:  # Add if maximal
                           maximal_itemsets.append(itemset)
            
            for itemset in maximal_itemsets:
                 self.maximal_itemsets_text.append(f"{set(itemset)}")  # Correctly append to maximal_itemsets_text

            rules = apriori.generate_association_rules(min_conf)   

            if rules:  # Corrected display of association rules
                 for rule in rules:
                     antecedent, consequent, confidence = rule
                     self.association_rules_text.append(f"{antecedent} --> {consequent} (Confidence: {confidence:.2f})")  # Correctly append to association_rules_text

            else:
                 self.association_rules_text.append("Không tìm thấy luật kết hợp nào.") # Message when no rules found

        except Exception as e:
            self.apriori_result_text.append(f"Lỗi: {e}")
 
    def calculate_raw(self):
        try:
            self.raw_result_text_approx.clear()
            self.raw_result_text_dependency.clear()
            self.raw_result_text_reducts.clear()
            self.raw_result_text_rules.clear()

            selected_X = [item.text() for item in self.X_list.selectedItems()]
            selected_B = [item.text() for item in self.B_list.selectedItems()]
            selected_B1 = [item.text() for item in self.B1_list.selectedItems()]

            if not selected_X or not selected_B or not selected_B1:  # Check if X, B, and B1 are selected
                self.raw_result_text_approx.append("Vui lòng chọn X, B và B1.")
                return

            if len(self.raw_data.columns) > 0:
                decision_attribute = self.raw_data.columns[-1]
            else:
                self.raw_result_text_approx.append("Tập dữ liệu trống hoặc không hợp lệ.")
                return

            analyzer = RoughSetAnalyzer(self.raw_data)
            analyzer.result_text = []

            lower, upper = analyzer.approximate(selected_X, selected_B)
            self.raw_result_text_approx.append(f"Xấp xỉ dưới của X qua tập thuộc tính B là: {lower}")
            self.raw_result_text_approx.append(f"Xấp xỉ trên của X qua tập thuộc tính B là: {upper}")
            if len(upper) > 0:
                coeff = len(lower) / len(upper)
                self.raw_result_text_approx.append(f"Hệ số xấp xỉ: {coeff:.2f}")

            dependency_result_msg = analyzer.dependency(decision_attribute, selected_B1)  # Get dependency result string using B1
            self.raw_result_text_dependency.append(f"{dependency_result_msg}")  # Display message

            reducts_result = analyzer.reducts(decision_attribute)
            print(reducts_result)
            if reducts_result:  # Display Reducts if they exist
                distributed_result = []
                single_clause = reducts_result[0]  # Clause đơn
                multi_clause = reducts_result[1]  # Clause chứa nhiều phần tử

                # Tạo các thành phần phân bố
                for item in multi_clause:
                    distributed_result.append(f"({single_clause[0]} , {item})")

                # Kết hợp các thành phần phân bố bằng ∧
                distributed_laws = " và ".join(distributed_result)
                self.raw_result_text_reducts.append(distributed_laws) 

            else:
                self.raw_result_text_reducts.append("Không tìm thấy reduct nào.")

            # Generate rules with 100% accuracy
            dot = Digraph()
            class_list = self.raw_data[decision_attribute].unique()
            analyzer.generate_rules(dot, 'root', 'root', self.raw_data, class_list)

            # Display the rules in the raw_result_text_rules QTextEdit
            self.raw_result_text_rules.append("\n".join(analyzer.result_text))

        except Exception as e:
            self.raw_result_text_approx.append(f"Lỗi: {e}")

    def predict_laplace(self):
        try:
            instance = {}
            for feature, combobox in self.feature_comboboxes.items():
                selected_value = combobox.currentText()
                instance[feature] = selected_value

            use_laplace = self.laplace_smoothing_checkbox.isChecked()
            if use_laplace:
                nb_classifier = NaiveBayesLaplace(self.label)
            else:
                nb_classifier = NaiveBayes(self.label)

            nb_classifier.train(self.data)

            self.nb_result_text.clear()
            probabilities = {}
            for class_value in nb_classifier.class_priors:
                prob = nb_classifier.class_priors[class_value]
                prob_str = f"P(Play={class_value})"

                self.nb_result_text.append(f"Xét Play = {class_value}:")
                self.nb_result_text.append(f"\n  {prob_str} = {prob:.6f}\n")

                for feature, feature_value in instance.items():
                    if feature != self.label:
                        if feature in nb_classifier.feature_probs:
                            if feature_value not in nb_classifier.feature_probs[feature][class_value]:
                                if use_laplace:
                                    conditional_prob = nb_classifier.feature_probs[feature][class_value].get(feature_value, 0)
                                    prob *= conditional_prob
                                    prob_str += f" * P({feature}={feature_value}|Play={class_value}\n)"
                                    self.nb_result_text.append(
                                        f"  P({feature}={feature_value}|Play={class_value}) = {conditional_prob:.6f}\n")

                                else: 
                                    prob = 0
                                    self.nb_result_text.append(
                                        f"  P({feature}={feature_value}|Play={class_value}) = 0.000000\n")         
                            else:  
                                conditional_prob = nb_classifier.feature_probs[feature][class_value][feature_value]
                                prob *= conditional_prob
                                prob_str += f" * P({feature}={feature_value}|Play={class_value})"
                                self.nb_result_text.append(
                                    f"  P({feature}={feature_value}|Play={class_value}) = {conditional_prob:.6f}\n")
                        
                probabilities[class_value] = (prob, prob_str)
                self.nb_result_text.append(f"  {prob_str} = {prob:.6f}\n")


            prediction = max(probabilities, key=lambda x: probabilities[x][0]) if probabilities else None
            if prediction is not None:
                self.nb_prediction_label.setText(f"Mẫu X được phân vào lớp: {prediction}")
            else:
                self.nb_prediction_label.setText("Mẫu X được phân vào lớp: Không thể dự đoán.")

        except Exception as e:
            self.nb_result_text.append(f"Lỗi: {e}")

    def run_kmeans(self):
        try:
            self.kmeans_result_text.clear()
            self.kmeans_cluster_text.clear()
            k = int(self.k_input.text())

            kmeans = KMeans(k)
            kmeans.result_text = self.kmeans_result_text

            U, centroids = kmeans.fit(self.kmeans_data)  # Call fit() only ONCE

            data_point_labels = [f"X{i + 1}".rjust(3) for i in range(len(self.kmeans_data))]
            cluster_labels = [f"C{i + 1}".rjust(3) for i in range(k)]

            # 1. Display partition matrix
            self.kmeans_result_text.append("Ma trận phân hoạch U0:")
            header = "   \t" + "  \t".join(data_point_labels)
            matrix_string = ""
            for j in range(len(cluster_labels)):
                row = f"{cluster_labels[j]}\t"
                for i in range(len(data_point_labels)):
                    row += str(int(U[i][j])).rjust(3) + "\t"  # Pad matrix elements
                matrix_string += row + "\n"
            self.kmeans_result_text.append(f"{header}\n{matrix_string}")

            # 2. Display centroids
            self.kmeans_result_text.append("Trọng tâm các cụm:")
            for i, centroid in enumerate(centroids):
                formatted_centroid = [f"{val:.2f}" for val in centroid]  # Format each value
                self.kmeans_result_text.append(f"C{i+1}: [{', '.join(formatted_centroid)}]") 

            # 3. Calculate Euclidean distances and cluster assignments
            self.kmeans_result_text.append("\nKhoảng cách Euclidean :")
            header = " "+ "\t  " + "\t ".join(cluster_labels) + f" \t{'Cụm'.rjust(6)}"
            self.kmeans_result_text.append(header)


            for i, point in enumerate(self.kmeans_data):
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                formatted_distances = [f"{dist:.2f}".rjust(6) for dist in distances]

                min_distance_index = np.argmin(distances)  # Find closest cluster index
                assigned_cluster = f"C{min_distance_index + 1}"

                self.kmeans_result_text.append(f"X{i+1}\t" + "\t".join(formatted_distances) + f"\t{assigned_cluster.rjust(6)}")

            # 4. Display partition matrix U1
            self.kmeans_result_text.append("\nMa trận phân hoạch U1:")
            header = "   \t" + "  \t".join(data_point_labels)
            matrix_string = ""
            for j in range(len(cluster_labels)):
                row = f"{cluster_labels[j]}\t"
                for i in range(len(data_point_labels)):
                    row += str(int(U[i][j])).rjust(3) + "\t"  # Pad matrix elements
                matrix_string += row + "\n"
            self.kmeans_result_text.append(f"{header}\n{matrix_string}")


            # 5. Display cluster assignments
            clusters = [[] for _ in range(k)]  # Initialize clusters for final assignments
            for i, point in enumerate(self.kmeans_data):  # Cluster based on min distance
                min_distance_index = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])  # Assign based on min distance
                clusters[min_distance_index].append(data_point_labels[i])

            for i, cluster in enumerate(clusters):  # Display based on final assignments
                self.kmeans_cluster_text.append(f"Cụm {i + 1}: {cluster}")


        except Exception as e:
            self.kmeans_result_text.append(f"Lỗi: {e}")
    

    def display_content(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def calculate_decision_tree(self):
        try:
            self.id3_result_text_nodes.clear()
            self.id3_result_text_rules.clear()
            criterion = self.criterion_combobox.currentData()
            label = "Play"  # Or determine dynamically

            if not hasattr(self, 'dt_data'):
                self.id3_result_text_nodes.append("Vui lòng tải dữ liệu trước.")
                return

            if 'Day' in self.dt_data.columns:
                self.dt_data = self.dt_data.drop(columns=['Day'])

            class_list = self.dt_data[label].unique()

            dot = Digraph()

            decision_tree_gen = DecisionTreeGenerator(label, criterion)
            decision_tree_gen.result_text = self.id3_result_text_nodes

            features = self.dt_data.columns.drop(label)  # Define features here, outside the if/else blocks

            # Initial calculations at the root node
            if criterion == "gini":
                self.id3_result_text_nodes.append("Xét nút gốc:")
                gini_values = {feature: decision_tree_gen.calc_gini_for_feature(self.dt_data, feature, class_list) for feature in features}

                # Check if there are features available to split on
                if gini_values:
                     for feature, gini in gini_values.items():
                         self.id3_result_text_nodes.append(f"  Gini({feature}) = {gini:.4f}")
                     best_feature = decision_tree_gen.select_best_feature(self.dt_data, class_list, features)
                     self.id3_result_text_nodes.append(f"=> Chọn {best_feature} làm thuộc tính phân nhánh\n")
                else:  # No features to split, create leaf node based on majority class.
                      majority_class = self.dt_data[self.label].mode().iloc[0] if not self.dt_data.empty else None
                      if majority_class:
                        self.id3_result_text_nodes.append(f"Tạo nút lá: {majority_class}")

                      return  # Stop further processing


            elif criterion == "info_gain":  # Calculate info gain for each feature if criterion is info_gain
                 self.id3_result_text_nodes.append("Xét nút gốc:")


                 info_gain_values = {feature: decision_tree_gen.calc_info_gain_for_feature(self.dt_data, feature, class_list) for feature in features}
                 if info_gain_values: # Ensure that features exist for this calculation at the root node
                      for feature, info_gain in info_gain_values.items():
                           self.id3_result_text_nodes.append(f"  InfoGain({feature}) = {info_gain:.4f}")


                      best_feature = decision_tree_gen.select_best_feature(self.dt_data, class_list, features)
                      self.id3_result_text_nodes.append(f"=> Chọn {best_feature} làm thuộc tính phân nhánh\n")
                 else:
                     majority_class = self.dt_data[self.label].mode().iloc[0] if not self.dt_data.empty else None  # Get majority class in case there are no more features but more than one class
                     if majority_class:
                         self.id3_result_text_nodes.append(f"Tạo nút lá: {majority_class}")


                     return # No features to split on

            decision_tree_gen.make_tree(dot, 'root', 'root', self.dt_data, class_list)

            dot.render('decision_tree', view=True)
           
            dot_rules = Digraph()
            decision_tree_gen.result_text = self.id3_result_text_rules
            decision_tree_gen.generate_rules(dot_rules, 'root', 'root', self.dt_data, class_list)

        except Exception as e:
            self.id3_result_text_nodes.append(f"Lỗi: {e}")

    def load_data_preprocessing(self):
        self.load_data('preprocessing')

    def load_data_apriori(self):
        self.load_data('apriori')

    def load_data_raw(self):
        self.load_data('raw')

    def load_data_naive_bayes_laplace(self):
        self.load_data('naive_bayes')

    def load_data_kmeans(self):
        self.load_data('kmeans')

    def load_data_decision_tree(self):
        self.load_data('decision_tree')

    def calculate_correlation(self):
        try:
            if not hasattr(self, 'preprocessing_data'):
                self.preprocessing_result_text.append("Vui lòng tải dữ liệu trước.")
                return

            if self.preprocessing_data.shape[1] < 3:
                self.preprocessing_result_text.append("Tập dữ liệu phải có ít nhất 3 cột.")
                return

            x = self.preprocessing_data.iloc[:, 1]
            y = self.preprocessing_data.iloc[:, 2]

            r = np.corrcoef(x, y)[0, 1]

            self.preprocessing_result_text.clear()
            self.preprocessing_result_text.append(f"Hệ số tương quan r: {r:.4f}")

            if r > 0:
                self.preprocessing_result_text.append("X và Y tương quan thuận với nhau.")
            elif r == 0:
                self.preprocessing_result_text.append("X và Y không tương quan với nhau (độc lập).")
            else:
                self.preprocessing_result_text.append("X và Y tương quan nghịch với nhau, loại trừ lẫn nhau.")

            if abs(r) < 0.2:
                self.preprocessing_result_text.append("Mối tương quan: Quá thấp.")
            elif abs(r) < 0.4:
                self.preprocessing_result_text.append("Mối tương quan: Thấp.")
            elif abs(r) < 0.6:
                self.preprocessing_result_text.append("Mối tương quan: Trung bình.")
            elif abs(r) < 0.8:
                self.preprocessing_result_text.append("Mối tương quan: Cao.")
            else:
                self.preprocessing_result_text.append("Mối tương quan: Rất cao.")

        except Exception as e:
            self.preprocessing_result_text.append(f"Lỗi: {e}")

    def run_kohonen(self):
        try:
            self.kohonen_result_text.clear()
            self.kohonen_cluster_text.clear()
            learning_rate = self.learning_rate_input.value()
            iterations = self.iterations_input.value()
            neighborhood_radius = self.neighborhood_radius_input.value()

            kohonen = Kohonen(learning_rate, iterations, neighborhood_radius)
            kohonen.train(self.kohonen_data, self.kohonen_result_text)

            self.kohonen_result_text.append("\nTrọng số sau các lần lặp:")
            for i, weight in enumerate(kohonen.weights):
                self.kohonen_result_text.append(f"w{i+1}: {[f'{w:.2f}'.rstrip('0').rstrip('.') for w in weight]}")

            clusters = {1: [], 2: [], 3: []}
            for i, sample in enumerate(self.kohonen_data):
                cluster = kohonen.predict(sample) + 1
                clusters[cluster].append(f"Tranh {i + 1}")

            for cluster, images in clusters.items():
                self.kohonen_cluster_text.append(f"Cụm {cluster}: {', '.join(images)}")

        except Exception as e:
            self.kohonen_result_text.append(f"Lỗi: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())