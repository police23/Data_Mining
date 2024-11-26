from itertools import combinations
import math
import sys
from tkinter.font import Font
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QFormLayout, QLineEdit, QWidget, QTabWidget, QVBoxLayout, 
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
        self.min_support = min_support * len(data)
        self.data = data
        self.frequent_itemsets = []

    def generate_frequent_itemsets(self):
        unique_items = sorted(set(item for transaction in self.data for item in transaction))
        all_frequent_itemsets = []
        k = 1
        while True:
            frequent_k_itemsets = self.find_frequent_itemsets(unique_items, k)
            if not frequent_k_itemsets:
                break
            all_frequent_itemsets.extend(frequent_k_itemsets)


            items_for_next_k = set() # To remove duplicate
            for itemset, _ in frequent_k_itemsets:  # Use all frequent items from current k-level
                items_for_next_k.update(itemset)
            unique_items = sorted(list(items_for_next_k))

            k += 1

        self.frequent_itemsets = all_frequent_itemsets



    def find_frequent_itemsets(self, items, k):
        candidate_itemsets = list(combinations(items, k))
        frequent_k_itemsets = []

        for itemset in candidate_itemsets:
            count = 0 # reset count for each itemset
            for transaction in self.data: # Count based on transactions in self.data
                if set(itemset).issubset(transaction):
                    count += 1


            if count >= self.min_support:
                frequent_k_itemsets.append((set(itemset), count))

        return frequent_k_itemsets

    def generate_association_rules(self, min_confidence):

        rules = []
        for itemset, support in self.frequent_itemsets:  # Corrected iteration
             if isinstance(itemset, tuple) and len(itemset) > 1:  # Filter size 2 or greater

                for i in range(1, len(itemset)):
                     for antecedent in combinations(itemset, i):
                         consequent = tuple(sorted(set(itemset) - set(antecedent)))
                         support_antecedent = sum(1 for transaction in self.data if set(antecedent).issubset(set(transaction)))

                         confidence = support / support_antecedent if support_antecedent > 0 else 0


                         if confidence >= min_confidence:
                               rules.append((antecedent, consequent, confidence))

        return rules


class KMeansClustering:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        for _ in range(self.max_iterations):
            self.clusters = self.create_clusters(X)
            new_centroids = self.calculate_new_centroids(X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((sample - self.centroids) ** 2, axis=1)))
            clusters[closest_centroid].append(idx)
        return clusters

    def calculate_new_centroids(self, X):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for idx, cluster in enumerate(self.clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            new_centroids[idx] = new_centroid
        return new_centroids

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for idx, sample in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((sample-self.centroids)**2, axis=1)))
            predictions[idx] = closest_centroid
        return predictions

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

class NaiveBayesLaplace:  # New class for Laplace smoothing
    def __init__(self, label):
        self.label = label
        self.probabilities = {}  # Store conditional probabilities
        self.class_priors = {}    # Store class priors

    def train(self, data):

        class_counts = data[self.label].value_counts()
        total_instances = len(data)


        for class_value in class_counts.index:
            self.class_priors[class_value] = class_counts[class_value] / total_instances
            class_data = data[data[self.label] == class_value]
            self.probabilities[class_value] = {}

            for feature in data.columns.drop(self.label): # Drop label from columns when calculate
                self.probabilities[class_value][feature] = {}
                feature_counts = class_data[feature].value_counts()
                num_feature_values = len(data[feature].unique())

                for feature_value in data[feature].unique():  # Iterate over ALL unique feature values
                    count = feature_counts[feature_value] if feature_value in feature_counts else 0  # Check if val exists
                    self.probabilities[class_value][feature][feature_value] = (count + 1) / (class_counts[class_value] + num_feature_values)  # Laplace smoothing

    def predict(self, instance):

        predictions = {}
        for class_value in self.class_priors:
            probability = self.class_priors[class_value]
            for feature, feature_value in instance.items():
                if feature != self.label:
                    if feature_value in self.probabilities[class_value][feature]:
                        probability *= self.probabilities[class_value][feature][feature_value]
                    else:  # Feature value not seen in training, set to a small probability
                        probability = 0 #  If feature value not seen, class prob is zero (no Laplace)
                        break   # No need to check other features for this class

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
        unique_C = self.data[C].unique()
        unique_combinations_B = self.data[B].drop_duplicates()
        dependency_exists = True

        for c_val in unique_C:
            indices_c = self.data[self.data[C] == c_val].index

            for index, row in unique_combinations_B.iterrows():
                b_combination = row
                indices_b = self.data[self.data[B].eq(b_combination).all(axis=1)].index
                intersection = indices_c.intersection(indices_b)

                if not intersection.empty and len(self.data.loc[intersection, C].unique()) > 1:
                    dependency_exists = False
                    break 
            if not dependency_exists:
                break

        return dependency_exists

    def reducts(self, decision_attr):
        if self.data.empty:
            return []

        features = self.data.columns.drop(['O', decision_attr]).tolist()
        n = len(features)
        reducts_found = []

    # Precompute the dependency for all combinations of features
        for i in range(1, 1 << n):  # Iterate over all non-empty subsets
            current_features = [features[j] for j in range(n) if (i >> j) & 1]

        # Check dependency only if current_features is not empty
        if current_features and self.dependency(decision_attr, current_features):
            # Check if the current reduct is minimal
            is_minimal = all(not set(existing_reduct).issubset(set(current_features)) for existing_reduct in reducts_found)

            if is_minimal:
                # Remove any existing reducts that are supersets of the current reduct
                reducts_found = [
                    reduct for reduct in reducts_found if not set(current_features).issubset(set(reduct))
                ]
                reducts_found.append(current_features)

        readable_reducts = [", ".join(reduct) for reduct in reducts_found]
        print("Reducts found:")
        for idx, reduct in enumerate(readable_reducts, 1):
           print(f"Reduct {idx}: {reduct}")

        return reducts_found


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
            ("Naive Bayes không dùng Laplace", "NB.png"),
            ("Naive Bayes dùng Laplace", "NB.png"),
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



                apriori_layout = QVBoxLayout()
                apriori_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.apriori_load_button)

                # Add input fields for min_support and min_confidence
                apriori_layout.addWidget(QLabel("Min Support:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.min_support_input)
                apriori_layout.addWidget(QLabel("Min Confidence:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.min_confidence_input)
                apriori_layout.addWidget(self.apriori_calculate_button)  # Trigger calculation



                apriori_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                apriori_layout.addWidget(self.apriori_result_text, stretch=1)  # Allow results to expand

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
                self.id3_result_text = QTextEdit()
                self.id3_result_text.setReadOnly(True)
                self.id3_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 25pt; /* Increased text edit font size */
                    }
                """)


                # Criterion combobox *CREATED FIRST*
                self.criterion_combobox = QComboBox()
                self.criterion_combobox.addItem("Chỉ số Gini", "gini")
                self.criterion_combobox.addItem("Độ lợi thông tin", "info_gain")
                self.criterion_combobox.setStyleSheet("""
                    QComboBox {
                        font-size: 14pt;  /* Increased combobox font size */
                        margin-bottom: 15px;
                    }
                    QComboBox QAbstractItemView { /* ... */ }  # Dropdown styles
                """)




                # Now create the layout and add widgets
                id3_layout = QVBoxLayout()
                id3_layout.addWidget(QLabel("Chọn cách tính:", font=QFont("Arial", 14)))  # Set font size here
                id3_layout.addWidget(self.criterion_combobox, stretch=1)
                id3_layout.addWidget(self.id3_load_button)
                id3_layout.addWidget(QLabel("Các luật của cây quyết định:", font=QFont("Arial", 14)))
                result_container = QWidget()
                result_layout = QVBoxLayout(result_container)
                result_layout.addWidget(self.id3_result_text)


                result_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
                result_layout.setSpacing(0)  # Remove spacing within container
                result_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.id3_result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                self.id3_result_text.setMinimumSize(0,0)  # Ensure no minimum size restriction on text edit


                id3_layout.addWidget(result_container, stretch=1) # stretch on the container
                tab.setLayout(id3_layout)

                self.id3_load_button.clicked.connect(self.calculate_gini_and_select_feature)
            
            elif i == 3:  # Naive Bayes (with Laplace) Tab
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
                self.nb_laplace_result_text = QTextEdit()
                self.nb_laplace_result_text.setReadOnly(True)
                self.nb_laplace_result_text.setStyleSheet("""
                    QTextEdit {
                        font-size: 25pt;
                    }
                """)

                self.feature_comboboxes = {}

                self.predict_button = QPushButton("Dự đoán")
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
                nb_laplace_layout = QVBoxLayout()
                nb_laplace_layout.addWidget(QLabel("Tải tập dữ liệu:", font=QFont("Arial", 12)))
                nb_laplace_layout.addWidget(self.nb_load_button)

                form_layout = QFormLayout()  # Form layout for dynamic comboboxes
                nb_laplace_layout.addWidget(QLabel("Chọn giá trị thuộc tính:", font=QFont("Arial", 12)))
                nb_laplace_layout.addLayout(form_layout)  # Add form layout

                nb_laplace_layout.addWidget(self.predict_button)
                nb_laplace_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))

                # Container for result text (for scrolling)
                result_container = QWidget()
                result_layout = QVBoxLayout(result_container)
                result_layout.addWidget(self.nb_laplace_result_text)
                result_layout.setContentsMargins(0, 0, 0, 0)
                result_layout.setSpacing(0)

                # ... (set size policies as shown in previous responses)

                nb_laplace_layout.addWidget(result_container, stretch=1) # stretch=1 for result container
                tab.setLayout(nb_laplace_layout)


                self.nb_load_button.clicked.connect(self.load_data_naive_bayes_laplace)
            elif i == 4:
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

                # X selection
                raw_layout.addWidget(QLabel("Chọn tập X:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.X_list)


                # B selection
                raw_layout.addWidget(QLabel("Chọn tập B:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.B_list)



                # C selection
                raw_layout.addWidget(QLabel("Thuộc tính quyết định:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.decision_attr_display)


                raw_layout.addWidget(self.raw_calculate_button)  # Trigger button

                raw_layout.addWidget(QLabel("Kết quả:", font=QFont("Arial", 12)))
                raw_layout.addWidget(self.raw_result_text, stretch=1) # Set result area to stretch


                tab.setLayout(raw_layout)

                self.raw_load_button.clicked.connect(self.load_data_raw)
                self.raw_calculate_button.clicked.connect(self.calculate_raw)

            elif i == 5: 
                kmeans_layout = QVBoxLayout()

                self.k_spinbox = QSpinBox()
                self.k_spinbox.setMinimum(2)
                self.k_spinbox.setValue(3)  # Default k
                kmeans_layout.addWidget(QLabel("Giá trị của k:"))
                kmeans_layout.addWidget(self.k_spinbox)

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
                self.kmeans_load_button.clicked.connect(self.perform_kmeans)
                kmeans_layout.addWidget(self.kmeans_load_button)

                self.kmeans_result_text = QTextEdit()
                self.kmeans_result_text.setReadOnly(True)
                kmeans_layout.addWidget(self.kmeans_result_text) 

                self.figure = plt.figure()
                self.canvas = FigureCanvas(self.figure)
                kmeans_layout.addWidget(self.canvas)

                tab.setLayout(kmeans_layout)

           



            



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
                self.apriori_data = pd.read_csv(filepath)  # Store Apriori data separately
                self.apriori_result_text.clear()
                # ... (process apriori_data if needed, e.g., drop columns, etc.)
            except Exception as e:
                self.apriori_result_text.append(f"Lỗi: {e}")

    def run_apriori(self):
        try:
            self.apriori_result_text.clear()
            min_support = self.min_support_input.value()
            min_confidence = self.min_confidence_input.value()

            # Data preprocessing: Convert to list of sets
            transactions = []
            for _, row in self.apriori_data.iterrows():
                transaction = set(row.dropna().astype(str))
                transactions.append(transaction)

            apriori = Apriori(min_support, transactions)
            apriori.generate_frequent_itemsets()

            # Display frequent itemsets (Corrected and simplified)
            frequent_itemsets_to_display = []
            for itemset, _ in apriori.frequent_itemsets:  # Ignore support counts here, using '_'
                frequent_itemsets_to_display.append(itemset)

            self.apriori_result_text.append("Các tập phổ biến:")
            for itemset in frequent_itemsets_to_display:  #Iterate over each set in the filtered list.
                self.apriori_result_text.append(str(itemset)) 


            # Calculate and display maximal frequent itemsets (corrected)

            if apriori.frequent_itemsets: # Only proceed if frequent itemsets exist
                self.apriori_result_text.append("\nCác tập phổ biến tối đại:")
                maximal_frequent_itemsets = []
                for itemset1, _ in apriori.frequent_itemsets:
                    is_maximal = True
                    for itemset2, _ in apriori.frequent_itemsets:
                        if itemset1 != itemset2 and itemset1.issubset(itemset2):
                            is_maximal = False
                            break
                    if is_maximal:
                        maximal_frequent_itemsets.append(itemset1)

                for itemset in maximal_frequent_itemsets:
                    self.apriori_result_text.append(f"{itemset}")


                # Generate and display association rules (only if frequent itemsets exist)
                association_rules = apriori.generate_association_rules(min_confidence)
                self.apriori_result_text.append("\nLuật kết hợp:")

                if association_rules: # Display if at least one rule is found.
                    for rule in association_rules:
                        self.apriori_result_text.append(f"{set(rule[0])} => {set(rule[1])} (Độ tin cậy = {rule[2]:.2f})")

                else:
                    self.apriori_result_text.append("Không tìm thấy luật kết hợp")



        except Exception as e:
            self.apriori_result_text.append(f"Lỗi: {e}")
    
    def load_data_raw(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if filepath:
            try:
                 self.raw_result_text.clear()  # Clear any existing result text

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

            if not selected_X or not selected_B:
                self.raw_result_text.append("Vui lòng chọn X và B.")
                return

            # Determine decision attribute (last column)
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
               approximation_coefficient = len(lower) / len(upper)
               self.raw_result_text.append(f"Hệ số xấp xỉ: {approximation_coefficient:.2f}")

            dependency_result = analyzer.dependency(decision_attribute, selected_B)
            self.raw_result_text.append(f"\nPhụ thuộc của {decision_attribute} vào B: {dependency_result}")

            reducts_result = analyzer.reducts(decision_attribute)
            self.raw_result_text.append("\nReducts:")
            for reduct in reducts_result:
                self.raw_result_text.append(str(reduct))

        except Exception as e:
            self.raw_result_text.append(f"Lỗi: {e}")

    def load_data_naive_bayes_laplace(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if filepath:
            try:
                 self.nb_laplace_result_text.clear()
                 self.data = pd.read_csv(filepath) # Store the data
                 if 'Day' in self.data.columns:
                     self.data = self.data.drop(columns=['Day'])


                 self.label = 'Play'


                 # Create comboboxes dynamically
                 form_layout = self.findChild(QFormLayout)  # Find the form layout
                 for i in reversed(range(form_layout.count())):
                    form_layout.itemAt(i).widget().setParent(None) # Delete existing comboboxes if any

                 self.feature_comboboxes = {}
                 for feature in self.data.columns.drop(self.label):
                     combobox = QComboBox()
                     combobox.addItems(self.data[feature].unique().astype(str))
                     self.feature_comboboxes[feature] = combobox
                     form_layout.addRow(QLabel(f"{feature}:", font=QFont("Arial", 12)), combobox) # Add to layout





            except Exception as e:
                 self.nb_laplace_result_text.append(f"Lỗi: {e}")

    def predict_laplace(self):
        try:
            instance = {}
            for feature, combobox in self.feature_comboboxes.items():
                selected_value = combobox.currentText()
                instance[feature] = selected_value

            nb_laplace = NaiveBayesLaplace(self.label)
            nb_laplace.train(self.data)


            self.nb_laplace_result_text.clear()

            probabilities = {}
            for class_value in nb_laplace.class_priors:
                prob = nb_laplace.class_priors[class_value]
                prob_str = f"P(Play={class_value})"

                self.nb_laplace_result_text.append(f"\nXét Play = {class_value}:")  # Indicate which class is being considered
                self.nb_laplace_result_text.append(f"  {prob_str} = {prob:.6f}") # Display the prior probability

                for feature, feature_value in instance.items():
                    if feature != self.label:
                        if feature_value in nb_laplace.probabilities[class_value][feature]:
                            conditional_prob = nb_laplace.probabilities[class_value][feature][feature_value]
                            prob *= conditional_prob
                            prob_str += f" * P({feature}={feature_value}|Play={class_value})"
                            self.nb_laplace_result_text.append(f"  P({feature}={feature_value}|Play={class_value}) = {conditional_prob:.6f}")  # Display individual conditional probabilities

                        else:
                            prob = 0
                            break  # No need to proceed if any value not found
                probabilities[class_value] = (prob, prob_str)
                self.nb_laplace_result_text.append(f"  {prob_str} = {prob:.6f}\n") # Final calculation for each Play value


            prediction = max(probabilities, key=lambda x: probabilities[x][0]) if probabilities else None



            self.nb_laplace_result_text.append(f"Dự đoán: Play={prediction}")



        except Exception as e:
            self.nb_laplace_result_text.append(f"Lỗi: {e}")

    def perform_kmeans(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                df = pd.read_csv(file_path)
                k = self.k_spinbox.value()

                if len(df.columns) < 2:
                    raise ValueError("Dữ liệu phải có ít nhất hai cột cho phân cụm K-Means.")



                X = df.values  # NumPy array of your data

                kmeans = KMeansClustering(k)
                kmeans.fit(X)
                predictions = kmeans.predict(X)



                self.figure.clear() # Clear any previous plot
                ax = self.figure.add_subplot(111) # Get the axes

                for i in range(k):
                    ax.scatter(X[predictions == i, 0], X[predictions == i, 1], label=f'Cụm {i+1}')
                ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=100, c='black', marker='x', label='Tâm')

                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(df.columns[1])
                ax.legend()
                self.canvas.draw() # Redraw the canvas


            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))
    
    

    def display_content(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def calculate_gini_and_select_feature(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tập dữ liệu", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            try:
                df = pd.read_csv(file_path)
                if 'Day' in df.columns:  # Remove 'Day' column if it exists
                    df = df.drop(columns=['Day'])

                label = "Play"
                class_list = df[label].unique()

                self.id3_result_text.clear()

                dot = Digraph()
                dot.node('root', 'root')


                criterion = self.criterion_combobox.currentData() #Corrected line

                decision_tree_gen = DecisionTreeGenerator(label, criterion)
                decision_tree_gen.result_text = self.id3_result_text
                decision_tree_gen.make_tree(dot, 'root', 'root', df, class_list)

                dot.render('decision_tree', view=True) # or  'decision_tree_info_gain'


                self.id3_result_text.append("Các luật được tạo:\n")
                dot_rules = Digraph()
                decision_tree_gen.generate_rules(dot_rules, 'root', 'root', df, class_list)



            except Exception as e:
                self.id3_result_text.append(f"Lỗi: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())