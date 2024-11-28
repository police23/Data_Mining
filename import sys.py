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


    def generate_rules(self, dot, parent_name, parent_node, train_data, class_list, current_rule=None):
        if train_data.empty or len(train_data.columns) == 1:  # Check for empty DataFrame or only 'Play' remaining
            return


        if len(train_data[self.label].unique()) == 1:  # Check for single class *before* features
            leaf_class = train_data[self.label].iloc[0]
            if current_rule:
                self.result_text.append(f"R: If {current_rule} Then Play={leaf_class}\n") # Add newline for clarity
            return

        features = train_data.columns.drop(self.label)  # Get available features
        best_feature = self.select_best_feature(train_data, class_list, features)


        if best_feature:
            dot.node(parent_name, best_feature, style='filled', color='olivedrab1')
            for feature_value in train_data[best_feature].unique():
                feature_value_data = train_data[train_data[best_feature] == feature_value]
                node_name = f"{parent_node}_{feature_value}"

                new_rule = f"{best_feature}={feature_value}"
                if current_rule:
                    new_rule = f"{current_rule} AND {new_rule}"  # Changed '^' to 'AND' for better readability

                self.generate_rules(dot, node_name, node_name, feature_value_data.drop(columns=best_feature),
                                   class_list, new_rule)
        else:  # Handle cases where no best feature is found
            # Determine the majority class in the current data
            majority_class = train_data[self.label].mode().iloc[0] if not train_data.empty else None
            if majority_class and current_rule:
                self.result_text.append(f"R: If {current_rule} Then Play={majority_class}\n")

    def make_tree(self, dot, parent_name, parent_node, train_data, class_list, branch_condition=None):
        if train_data.shape[0] != 0:
            if len(train_data[self.label].unique()) == 1:
                leaf_class = train_data[self.label].iloc[0]
                dot.node(parent_node, f"{leaf_class}", style='filled', color='lightgray', shape='ellipse')
                return

            if len(train_data.columns) == 1:  # Check for remaining columns/features
                return

            if len(train_data.columns) > 1:  # Only calculate best_feature if columns remain
                features = train_data.columns.drop(self.label)
                best_feature = self.select_best_feature(train_data, class_list, features)
            else:
                best_feature = None  # No more features to split on

            if best_feature:
                dot.node(parent_name, best_feature, style='filled', color='olivedrab1', shape='ellipse')

                feature_value_count_dict = train_data[best_feature].value_counts(sort=False)
                for feature_value, count in feature_value_count_dict.items():
                    feature_value_data = train_data[train_data[best_feature] == feature_value]
                    node_name = f"{parent_node}_{feature_value}"

                    assigned_to_node = False
                    for c in class_list:
                        class_count = feature_value_data[feature_value_data[self.label] == c].shape[0]
                        if class_count == count:
                            leaf_node_name = f"{node_name}_{c}"
                            dot.node(leaf_node_name, c, style='filled', color='lightgray', shape='ellipse')
                            dot.edge(parent_name, leaf_node_name, label=feature_value)
                            assigned_to_node = True
                            break

                    if not assigned_to_node:
                        current_branch_condition = f"{best_feature} = {feature_value}"
                        if branch_condition:
                            current_branch_condition = f"{branch_condition} AND {current_branch_condition}"

                        if len(feature_value_data.columns) > 1:
                            next_features = feature_value_data.drop(columns=best_feature).columns.drop(self.label)
                        else:
                            next_features = pd.Index([])  # Use empty Index if no features remain

                        if len(feature_value_data.drop(columns=best_feature)) > 0 and not next_features.empty:
                            next_best_feature = self.select_best_feature(feature_value_data.drop(columns=best_feature), class_list, next_features)
                        elif len(feature_value_data) > 0 and len(feature_value_data.columns) == 1 : # Corrected condition: only 'Play' remains
                            leaf_class = feature_value_data[self.label].iloc[0]
                            dot.node(node_name, f"{leaf_class}", style='filled', color='lightgray', shape='ellipse')                           
                        else:
                            next_best_feature = None
                        dot.node(node_name, "?", shape='ellipse')
                        dot.edge(parent_name, node_name, label=feature_value)
                        self.make_tree(dot, node_name, node_name, feature_value_data.drop(columns=best_feature), class_list, current_branch_condition)

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
        dependency_text = f"Phụ thuộc của {C} vào B: "

        if k == 1:
            dependency_text += "Phụ thuộc hoàn toàn (k = 1)"
        elif 0 < k < 1:
            dependency_text += f"Phụ thuộc một phần (k = {k:.2f})"
        else:
            dependency_text += "Không phụ thuộc (k = 0)"

        return dependency_text


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
        matrix = self.discernibility_matrix(decision_attr) # Compute matrix
        #Calculate discernibility function
        discernibility_function = set()  # Use a set to ensure unique clauses



        for row in matrix:  # Loop through the rows of the discernibility matrix
            for item in row:  # Loop through items in the rows
                if item:
                    discernibility_function.add(tuple(sorted(item)))  #Sort for easier comparision

        if not discernibility_function: # Handle cases with no reducts
           return []

        # Simplify discernibility function (using absorption law)
        simplified_function = list(discernibility_function)  # Make a copy


        simplified = True  # Start with True to enter the loop

        while simplified:
             simplified = False # Set it to False for each iteration of while loop
             new_simplified = []
             for clause1 in simplified_function:
                 absorbed = False
                 for clause2 in simplified_function:
                     if clause1 != clause2 and set(clause1).issubset(clause2):  # Apply absorption if subset
                           absorbed = True #clause1 is absorbed so no need to add it to the output
                           break  # Clause1 absorbed by clause 2, exit the inner loop
                 if not absorbed:
                    new_simplified.append(clause1) # if clause1 is not absorbed, it is part of the reduct
                    if set(simplified_function) != set(new_simplified):  # If changes made, update function and continue
                           simplified_function = new_simplified # Update the simplified_function
                           simplified = True  # Changes made, so continue the loop



        return [list(clause) for clause in simplified_function]


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
            padding: 5px 15px;
            min-width: 80px;
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
            ("Tập phổ biến và luật kết hợp", "book.png"),
            ("Decision Tree", "ID3.png"),
            ("Naive Bayes", "NB.png"),
            ("Tập thô", "raw_data.png"),
            ("K-Means", "cluster.png"),
        ]

        for i, (text, icon_file) in enumerate(items):
            tab = QWidget()
            icon = QIcon(icon_file)
            self.tabs.addTab(tab, icon, text)
            self.tabs.setIconSize(QSize(20, 20))
            
            if i == 0:  # Apriori Tab
                self.apriori_load_button = QPushButton("Upload file .CSV")
                self.apriori_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.apriori_load_button.clicked.connect(self.load_data_apriori) # Connect

                self.apriori_calculate_button = QPushButton("Tính toán") #Apriori button
                self.apriori_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
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
                        font-size: 15pt; 
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
                        border-radius: 4px;
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
                self.frequent_itemsets_text.setStyleSheet("font-size: 12pt;")


                self.maximal_itemsets_text = QTextEdit()
                self.maximal_itemsets_text.setReadOnly(True)
                self.maximal_itemsets_text.setStyleSheet("font-size: 12pt;")



                self.association_rules_text = QTextEdit()
                self.association_rules_text.setReadOnly(True)
                self.association_rules_text.setStyleSheet("font-size: 12pt;")
                self.apriori_filepath_display = QLineEdit() 
                self.apriori_filepath_display.setReadOnly(True)
                self.apriori_filepath_display.setStyleSheet("font-size: 12pt;")

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
                apriori_layout.addWidget(self.frequent_itemsets_text, stretch=1) # Added to layout

                apriori_layout.addWidget(QLabel("Tập phổ biến tối đại:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.maximal_itemsets_text, stretch=1)  # Added to layout
                apriori_layout.addWidget(QLabel("Luật kết hợp:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.association_rules_text, stretch=1)
                tab.setLayout(apriori_layout)


            if i == 1:  # ID3 Tab
                self.id3_load_button = QPushButton("Upload file .CSV")
                self.id3_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.id3_load_button.clicked.connect(self.load_data_decision_tree)

                self.dt_filepath_display = QLineEdit() 
                self.dt_filepath_display.setReadOnly(True)
                self.dt_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt; 
                    }
                """)

                self.id3_result_text = QTextEdit()
                self.id3_result_text.setReadOnly(True)
                self.id3_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 18pt; /* Increased text edit font size */
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
                        border-radius: 4px;
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

                dt_layout.addWidget(QLabel("Các luật của cây quyết định:", font=QFont("Arial", 12)))
                result_container = QWidget()
                result_layout = QVBoxLayout(result_container)
                result_layout.addWidget(self.id3_result_text)
                result_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
                result_layout.setSpacing(0)  # Remove spacing within container
                result_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text.setMinimumSize(0,0)  # Ensure no minimum size restriction on text edit

                dt_layout.addWidget(result_container, stretch=1) # stretch on the container
                tab.setLayout(dt_layout)

            
            elif i == 2:  # Naive Bayes
                self.nb_load_button = QPushButton("Upload file .CSV")
                self.nb_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
                        font-size : 20px;      /* Rounded corners */
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
                        border-radius: 4px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)

                self.nb_prediction_label = QLabel("Mẫu X được phân vào lớp:")
                self.nb_prediction_label.setFont(QFont("Arial", 12))

               

                nb_layout = QVBoxLayout()
                nb_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                nb_layout.addWidget(self.nb_load_button)
                nb_layout.addWidget(self.nb_filepath_display)
                nb_layout.addWidget(self.laplace_smoothing_checkbox)
                

                form_layout = QFormLayout()  # Form layout for dynamic comboboxes
                nb_layout.addWidget(QLabel("Chọn giá trị thuộc tính:", font=QFont("Arial", 12)))                
                nb_layout.addLayout(form_layout) 
                nb_layout.addWidget(self.predict_button)

                prediction_layout = QHBoxLayout()
                prediction_layout.addWidget(self.nb_prediction_label)
                nb_layout.addLayout(prediction_layout) 
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
                tab.setLayout(nb_layout)


                self.nb_load_button.clicked.connect(self.load_data_naive_bayes_laplace)
            elif i == 3:
                self.raw_load_button = QPushButton("Upload file .CSV")
                self.raw_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
                        font-size : 20px;      /* Rounded corners */
                    }
                    QPushButton:hover {
                        background-color: #1B5E20; /* Darker green on hover */
                    }
                    QPushButton:pressed {
                        background-color: #4CAF50; /* Brighter green when pressed */
                    }
                """)
                self.raw_filepath_display = QLineEdit()  # LineEdit to display filepath
                self.raw_filepath_display.setReadOnly(True)
                self.raw_filepath_display.setStyleSheet("""
                    QLineEdit {
                        font-size: 12pt;
                    }
                """)
                self.raw_result_text = QTextEdit()
                self.raw_result_text.setReadOnly(True)

                # Create list widgets for selecting X, B, and C
                self.X_list = QListWidget()
                self.X_list.setSelectionMode(QAbstractItemView.ExtendedSelection)  # Allow multiple selections
                self.B_list = QListWidget()
                self.B_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
                self.C_combobox = QComboBox()

                self.raw_calculate_button = QPushButton("Tính toán") # button to trigger
                self.raw_calculate_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
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
                
                self.decision_attr_display = QLineEdit()  # Create a QLineEdit
                self.decision_attr_display.setReadOnly(True)  # Make it read-only


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

                raw_layout.addLayout(selection_layout)


                # C selection
                raw_layout.addWidget(QLabel("Thuộc tính quyết định:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.decision_attr_display)


                raw_layout.addWidget(self.raw_calculate_button)  # Trigger button

                raw_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_result_text, stretch=1) # Set result area to stretch


                tab.setLayout(raw_layout)

                self.raw_load_button.clicked.connect(self.load_data_raw)
                self.raw_calculate_button.clicked.connect(self.calculate_raw)

            elif i == 4: 
                self.kmeans_load_button = QPushButton("Upload file .CSV")
                self.kmeans_load_button.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32; /* Green background */
                        color: white;            /* White text */
                        font-weight: bold;       /* Bold text */
                        border: none;            /* No border */
                        padding: 8px 16px;      /* Padding around text */
                        border-radius: 4px;
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
                        border-radius: 4px;
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
                        font-size: 18pt;
                    }
                """)

                kmeans_layout = QVBoxLayout()
                kmeans_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.kmeans_load_button)
                kmeans_layout.addWidget(self.kmeans_filepath_display)
                kmeans_layout.addWidget(QLabel("Nhập k:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.k_input)
                kmeans_layout.addWidget(self.kmeans_calculate_button)
                kmeans_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                kmeans_layout.addWidget(self.kmeans_result_text, stretch=1)


                tab.setLayout(kmeans_layout)

                self.kmeans_load_button.clicked.connect(self.load_data_kmeans)
                self.kmeans_calculate_button.clicked.connect(self.run_kmeans)
           



            



        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) #Remove margins on main layout if any

        main_layout.addWidget(self.tabs)

        # If using stacked widget:
        main_layout.addWidget(self.stacked_widget)  # Make sure to remove any stretch factors here
        self.tabs.currentChanged.connect(self.display_content)

        # Ensure MainWindow allows resizing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def load_data_apriori(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filepath:
            try:
                self.apriori_filepath_display.setText(filepath)
                self.apriori_data = []  # Initialize an empty list for transactions
                df = pd.read_csv(filepath)

                # Correctly create transactions by grouping items
                for transaction_id in df['Mã hóa đơn'].unique(): # Get each transaction and loop through each transaction's items
                    transaction = df[df['Mã hóa đơn'] == transaction_id]['Mã hàng'].tolist()
                    self.apriori_data.append(transaction)

            except Exception as e:
                self.apriori_result_text.append(f"Lỗi khi tải dữ liệu: {e}")

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
 
    def load_data_raw(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if filepath:
            try:
                 self.raw_result_text.clear()  # Clear any existing result text
                 self.raw_filepath_display.setText(filepath) 
                 self.raw_data = pd.read_csv(filepath) # Load raw data
                 if 'Day' in self.raw_data.columns:
                     self.raw_data = self.raw_data.drop(columns=['Day'])


                 # Populate list widgets
                 self.X_list.clear()
                 self.X_list.addItems(self.raw_data["O"].unique().astype(str))

                 self.B_list.clear()
                # Exclude 'Tên' and the dynamically determined decision attribute
                 if len(self.raw_data.columns) > 0:
                    decision_attribute = self.raw_data.columns[-1]  # Determine decision attribute
                    self.decision_attr_display.setText(decision_attribute) # Set the display
                    columns_to_add = self.raw_data.columns.drop(["O", decision_attribute]).tolist()  # Exclude these
                    self.B_list.addItems(columns_to_add)


            except Exception as e:
                self.raw_result_text.append(f"Lỗi: {e}")


    def calculate_raw(self):
        try:
            self.raw_result_text.clear()

            selected_X = [item.text() for item in self.X_list.selectedItems()]
            selected_B = [item.text() for item in self.B_list.selectedItems()]

            if not selected_X or not selected_B :  # Check if X and B are selected
                self.raw_result_text.append("Vui lòng chọn X và B.")
                return

            if len(self.raw_data.columns) > 0:
                decision_attribute = self.raw_data.columns[-1]
            else:
                self.raw_result_text.append("Tập dữ liệu trống hoặc không hợp lệ.")
                return


            analyzer = RoughSetAnalyzer(self.raw_data)

            lower, upper = analyzer.approximate(selected_X, selected_B)
            self.raw_result_text.append(f"Xấp xỉ dưới của X qua tập thuộc tính B là: {lower}")
            self.raw_result_text.append(f"Xấp xỉ trên của X qua tập thuộc tính B là: {upper}")
            if len(upper) > 0:
                  coeff = len(lower) / len(upper)
                  self.raw_result_text.append(f"Hệ số xấp xỉ: {coeff:.2f}")

            dependency_result_msg = analyzer.dependency(decision_attribute, selected_B)  # Get dependency result string
            self.raw_result_text.append(f"\n{dependency_result_msg}")  # Display message

            reducts_result = analyzer.reducts(decision_attribute)
            if reducts_result:  # Display Reducts if they exist
                self.raw_result_text.append("\nReducts:")

                reduct_strings = []
                for reduct in reducts_result:
                    reduct_strings.append("(" + " ∧ ".join(reduct) + ")")

                self.raw_result_text.append(" ∨ ".join(reduct_strings)) 

            else:
                 self.raw_result_text.append("\nKhông tìm thấy reduct nào.")


        except Exception as e:
            self.raw_result_text.append(f"Lỗi: {e}")

    def load_data_naive_bayes_laplace(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if filepath:
            try:
                self.nb_filepath_display.setText(filepath)
                self.nb_result_text.clear()
                self.data = pd.read_csv(filepath) # Store the data
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


            except Exception as e:
                 self.nb_result_text.append(f"Lỗi: {e}")

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
                self.nb_result_text.append(f"  {prob_str} = {prob:.6f}")

                for feature, feature_value in instance.items():
                    if feature != self.label:
                        if feature in nb_classifier.feature_probs:
                            if feature_value not in nb_classifier.feature_probs[feature][class_value]:
                                if use_laplace:
                                    conditional_prob = nb_classifier.feature_probs[feature][class_value].get(feature_value, 0)
                                    prob *= conditional_prob
                                    prob_str += f" * P({feature}={feature_value}|Play={class_value})"
                                    self.nb_result_text.append(
                                        f"  P({feature}={feature_value}|Play={class_value}) = {conditional_prob:.6f}")

                                else: 
                                    prob = 0
                                    self.nb_result_text.append(
                                        f"Giá trị '{feature_value}' của thuộc tính '{feature}' không có trong dữ liệu huấn luyện.")
                                    break 
                            else:  
                                conditional_prob = nb_classifier.feature_probs[feature][class_value][feature_value]
                                prob *= conditional_prob
                                prob_str += f" * P({feature}={feature_value}|Play={class_value})"
                                self.nb_result_text.append(
                                    f"  P({feature}={feature_value}|Play={class_value}) = {conditional_prob:.6f}")
                        
                probabilities[class_value] = (prob, prob_str)
                self.nb_result_text.append(f"  {prob_str} = {prob:.6f}\n")


            prediction = max(probabilities, key=lambda x: probabilities[x][0]) if probabilities else None
            if prediction is not None:
                self.nb_prediction_label.setText(f"Mẫu X được phân vào lớp: {prediction}")
            else:
                self.nb_prediction_label.setText("Mẫu X được phân vào lớp: Không thể dự đoán.")

        except Exception as e:
            self.nb_result_text.append(f"Lỗi: {e}")

    def load_data_kmeans(self): 

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if filepath:
            try:
               self.kmeans_result_text.clear()
               self.kmeans_data = pd.read_csv(filepath) # Loads numerical data
               if 'X' in self.kmeans_data.columns: # Remove the unnecessary column 'X' if it's in the dataframe
                    self.kmeans_data = self.kmeans_data.drop(columns=['X'])

               self.kmeans_data = self.kmeans_data.apply(pd.to_numeric, errors='coerce').dropna()
               self.kmeans_data = self.kmeans_data.values # Convert to numpy array for more efficient calculation
               self.kmeans_filepath_display.setText(filepath)




            except Exception as e:
               self.kmeans_result_text.append(f"Lỗi khi tải dữ liệu: {e}")


    def run_kmeans(self):
        try:
            self.kmeans_result_text.clear()
            k = int(self.k_input.text())

            kmeans = KMeans(k)
            kmeans.result_text = self.kmeans_result_text

            U, centroids = kmeans.fit(self.kmeans_data)  # Call fit() only ONCE

            data_point_labels = [f"X{i + 1}".rjust(3) for i in range(len(self.kmeans_data))]
            cluster_labels = [f"C{i + 1}".rjust(3) for i in range(k)]

            # 1. Display partition matrix
            self.kmeans_result_text.append("Ma trận phân hoạch:")
            header = "  M\t" + "  \t".join(data_point_labels)
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
            header = " "+ "\t  " + "\t\t ".join(cluster_labels) + f" \t\t{'Cụm'.rjust(6)}"
            self.kmeans_result_text.append(header)


            for i, point in enumerate(self.kmeans_data):
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                formatted_distances = [f"{dist:.2f}".rjust(6) for dist in distances]

                min_distance_index = np.argmin(distances)  # Find closest cluster index
                assigned_cluster = f"C{min_distance_index + 1}"

                self.kmeans_result_text.append(f"X{i+1}\t" + "\t".join(formatted_distances) + f"\t{assigned_cluster.rjust(6)}")




            # 4. Display cluster assignments
            self.kmeans_result_text.append("\nKết quả phân cụm:")
            clusters = [[] for _ in range(k)]  # Initialize clusters for final assignments
            for i, point in enumerate(self.kmeans_data):  # Cluster based on min distance
                min_distance_index = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])  # Assign based on min distance
                clusters[min_distance_index].append(data_point_labels[i])

            for i, cluster in enumerate(clusters):  # Display based on final assignments
                self.kmeans_result_text.append(f"Cụm {i + 1}: {cluster}")


        except Exception as e:
            self.kmeans_result_text.append(f"Lỗi: {e}")
    

    def display_content(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def load_data_decision_tree(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filepath:
            try:
                self.dt_filepath_display.setText(filepath)  # Update filepath display
                self.dt_data = pd.read_csv(filepath) #Load and store data
                if 'Day' in self.dt_data.columns:
                    self.dt_data = self.dt_data.drop(columns=['Day'])
            except Exception as e:
                self.id3_result_text.append(f"Lỗi: {e}")
    
    def calculate_decision_tree(self):
        try:
            self.id3_result_text.clear()  # Clear previous results

            criterion = self.criterion_combobox.currentData()
            label = "Play"  # Or determine dynamically if needed

            if not hasattr(self, 'dt_data'): # Check if data is loaded
                 self.id3_result_text.append("Vui lòng tải dữ liệu trước.")
                 return

            if 'Day' in self.dt_data.columns:  # Check and drop Day if exist
                self.dt_data = self.dt_data.drop(columns=['Day'])

            class_list = self.dt_data[label].unique()
            dot = Digraph()
            dot.node('root', 'root')

            decision_tree_gen = DecisionTreeGenerator(label, criterion)
            decision_tree_gen.result_text = self.id3_result_text

            decision_tree_gen.make_tree(dot, 'root', 'root', self.dt_data, class_list) # Correctly use self.dt_data

            dot.render('decision_tree', view=True)

            self.id3_result_text.append("Các luật được tạo:\n")
            dot_rules = Digraph()
            decision_tree_gen.generate_rules(dot_rules, 'root', 'root', self.dt_data, class_list)

        except Exception as e:
            self.id3_result_text.append(f"Lỗi: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())