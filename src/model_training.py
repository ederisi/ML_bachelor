import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import openpyxl

class HydraulicDiagnostic:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = self.load_data()
        self.targets = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Accumulator_Condition']
        self.stable_flag = 'Stable_Flag'

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data was not found: {self.data_path}")
        return pd.read_csv(self.data_path)

    def train_and_evaluate(self, target_col, features, title_suffix=""):
        """Teaches model for one feature and returns accuracy"""
        X = self.df[features]
        y = self.df[target_col]

        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=49
        )

        model = RandomForestClassifier(n_estimators=50, max_depth=8,min_samples_leaf=10,  random_state=49, n_jobs=-1)
        model.fit(X_train, y_train)

        # check overfitting (training vs test accuracy)
        train_acc = model.score(X_train, y_train) 
        test_acc = model.score(X_test, y_test)   

        # Predictions needed for the confusionmatrix
        y_pred = model.predict(X_test)

        # prints accuracies and visualisations
        if title_suffix != "":
            print(f"\n--- {target_col} ({title_suffix}) ---")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            
            # Draw matrix
            if title_suffix != "":
                y_pred = model.predict(X_test)
                self.plot_confusion_matrix(y_test, y_pred, f"{target_col} - {title_suffix}")

        return test_acc
    
    def run_prepruning_test(self, target_col, depth_list=range(1, 21)):
        """
        Tests model performance with different max_depth values.
        Identifies the 'sweet spot' where training and validation accuracy are balanced.
        """
        print(f"\n--- Running Pre-pruning Test (max_depth) for: {target_col} ---")
        
        all_features = [f for f in self.df.columns if f not in self.targets + [self.stable_flag]]
        X = self.df[all_features]
        y = self.df[target_col]
        
        # Split data for a clean train/test comparison in addition to CV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
        
        results = []

        for depth in depth_list:
            # Use the predetermined tree amount 50
            model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=depth, 
                random_state=49, 
                n_jobs=-1
            )
            
            # Train accuracy
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            
            # Crossvalidation accuracy
            # skf ensures class sizes remain "strict"
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=49)
            val_scores = cross_val_score(model, X, y, cv=skf)
            val_acc = val_scores.mean()

            results.append({
                'Depth': depth,
                'Train Acc (%)': train_acc * 100,
                'Validation Acc (%)': val_acc * 100,
                'Gap (%)': (train_acc - val_acc) * 100
            })
            print(f"Depth: {depth:>2} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {(train_acc-val_acc):.4f}")

        res_df = pd.DataFrame(results)
        
        # Save to excel
        os.makedirs("results/tables", exist_ok=True)
        res_df.to_excel(f"results/tables/PrePruning_{target_col}.xlsx", index=False)

        # Draw validation curve
        plt.figure(figsize=(10, 6))
        plt.plot(res_df['Depth'], res_df['Train Acc (%)'], 'o-', label='Training Accuracy', color='red')
        plt.plot(res_df['Depth'], res_df['Validation Acc (%)'], 'o-', label='Validation Accuracy', color='blue')
        
        plt.title(f"Pre-pruning Analysis (Bias-Variance Trade-off): {target_col}")
        plt.xlabel("Tree Depth (max_depth)")
        plt.ylabel("Accuracy (%)")
        plt.xticks(depth_list)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        os.makedirs("results/plots/pruning", exist_ok=True)
        plt.savefig(f"results/plots/pruning/Pruning_{target_col}.png")
        plt.close()
        
        return res_df
    

    def plot_generalized_importance(self, importance_dict, title, subfolder=""):
        """Generates grouped sensor importance plot to specific subfolder."""
        if not importance_dict: return
        
        importance_df = pd.DataFrame(importance_dict).mean(axis=1).sort_values(ascending=False)
        sensor_groups = importance_df.groupby(lambda x: x.split('_')[0]).sum().sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sensor_groups.values, y=sensor_groups.index, hue=sensor_groups.index, palette="viridis", legend=False)
        plt.title(f"Generalized Importance: {title}")
        plt.xlabel("Aggregated Importance Score")
        
        # Create a path for folder
        path = os.path.join("results/plots/importance", subfolder)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"Importance_{title.replace(' ', '_')}.png"))
        plt.close()


    def run_quantile_stress_test(self, target_col, use_temp=True):
        """Returns results and feature importance for the stress test."""
        # Split data by temperature (80/20)
        temp_cols = [c for c in self.df.columns if c.startswith('TS')]
        avg_temp = self.df[temp_cols].mean(axis=1)
        threshold = avg_temp.quantile(0.80)
        
        train_df = self.df[avg_temp <= threshold].reset_index(drop=True)
        val_df = self.df[avg_temp > threshold].reset_index(drop=True)

        features = [f for f in self.df.columns if f not in self.targets + [self.stable_flag]]
        if not use_temp:
            features = [f for f in features if not f.startswith('TS')]
        
        X_train, y_train = train_df[features], train_df[target_col]
        X_val, y_val = val_df[features], val_df[target_col]

        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=49, n_jobs=-1)
        model.fit(X_train, y_train)

        # Collect importance
        importance = pd.Series(model.feature_importances_, index=features)
        
        acc = model.score(X_val, y_val)
        gap = model.score(X_train, y_train) - acc
        
        # Save matrix
        y_pred = model.predict(X_val)
        sub = "stress/with_temp" if use_temp else "stress/without_temp"
        self.plot_confusion_matrix(y_val, y_pred, f"Stress_{target_col}", subfolder=sub)

        return {'Target': target_col, 'Acc': acc, 'Gap': gap}, importance


    def plot_confusion_matrix(self, y_true, y_pred, title, subfolder=""):
        """Saves matrix to specific subfolder."""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {title}')
        plt.ylabel('True state')
        plt.xlabel('Predicted state')
        
        path = os.path.join("results/plots/matrices", subfolder)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{title.replace(' ', '_')}.png"))
        plt.close()


    def run_comparative_analysis(self, k=5):
        """Makes the comparison of the different faults with/without temperature
            using crossvalidation"""
        # All sensors (take out the labels)
        all_features = self.df.drop(self.targets + [self.stable_flag], axis=1).columns.tolist()
        
        # Without temperature all the (TS) starting away
        no_temp_features = [f for f in all_features if not f.startswith('TS')]

        feature_sets = {
            "All sensors": all_features,
            "Without temperature": no_temp_features
        }

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=49)
        comparison_results = []

        # Dictionaries to hold aggregated importance for the standard analysis
        standard_all_importances = {}
        standard_no_temp_importances = {}

        print("\n--- Comparative Analysis & Matrices ---")
        

        for target in self.targets:
            Y = self.df[target]
            for set_name, f_list in feature_sets.items():
                X = self.df[f_list]
                model = RandomForestClassifier(n_estimators=50, max_depth=8,min_samples_leaf=10,  random_state=49, n_jobs=-1)
                

                # Capture importance from a single fit for visualization
                model.fit(X, Y)
                importance_series = pd.Series(model.feature_importances_, index=f_list)
                if set_name == "All sensors":
                    standard_all_importances[target] = importance_series
                else:
                    standard_no_temp_importances[target] = importance_series

                # Run the crossvalidation
                scores = cross_val_score(model, X, Y, cv=skf)
                mean_acc = scores.mean()
                std_acc = scores.std()

                # Call the existing function to draw the confusion matrices
                self.train_and_evaluate(target, f_list, title_suffix=set_name)

                comparison_results.append({
                    'Target': target,
                    'Set': set_name,
                    'Mean Accuracy': mean_acc,
                    'Std_dev' : std_acc
                })


        # Generate plots for the Standard Comparison
        self.plot_generalized_importance(standard_all_importances, "Standard All Sensors")
        self.plot_generalized_importance(standard_no_temp_importances, "Standard Without Temperature")
        
        # Print the Overall analysis to the end
        print("\n" + "="*65)
        print(f"{'Target':<22} | {'Set':<15} | {'Mean Acc':<10} | {'Std Dev'}")
        print("-" * 65)
        for res in comparison_results:
            print(f"{res['Target']:<22} | {res['Set']:<15} | {res['Mean Accuracy']:<10.2%} | {res['Std_dev']:.4f}")

        return pd.DataFrame(comparison_results)
    
    def run_convergence_test(self, target_col, n_list=[1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150]):
        """
        Tests RF convergence and saves results to Excel for Word.
        """
        print(f"\n--- Running Convergence Test (1-150) for: {target_col} ---")
        
        all_features = [f for f in self.df.columns if f not in self.targets + [self.stable_flag]]
        X = self.df[all_features]
        y = self.df[target_col]
        
        results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=49)

        for n in n_list:
            model = RandomForestClassifier(
                n_estimators=n, 
                max_depth=8, 
                min_samples_leaf=10, 
                random_state=49, 
                n_jobs=-1
            )
            scores = cross_val_score(model, X, y, cv=skf)
            results.append({
                'Trees': n,
                'Mean Acc (%)': scores.mean() * 100,
                'Std Dev': scores.std()
            })
        
        res_df = pd.DataFrame(results)
        
        # save in excel
        os.makedirs("results/tables", exist_ok=True)
        excel_file = f"results/tables/Convergence_{target_col}.xlsx"
        
        # needs openpyx1
        try:
            res_df.to_excel(excel_file, index=False)
            print(f"Excel-taulukko tallennettu: {excel_file}")
        except ImportError:
            # if openpyx1 missing, opens in csv
            csv_file = excel_file.replace(".xlsx", ".csv")
            res_df.to_csv(csv_file, index=False)
            print(f"Excel-kirjasto puuttui, tallennettu CSV: {csv_file}")

        # create a curve
        plt.figure(figsize=(10, 6))
        plt.plot(res_df['Trees'], res_df['Mean Acc (%)'], marker='o', linestyle='-', color='b')
        plt.title(f"Random Forest Convergence: {target_col}")
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        os.makedirs("results/plots/convergence", exist_ok=True)
        plt.savefig(f"results/plots/convergence/Convergence_{target_col}.png")
        plt.close()
        
        return res_df


    def run_all_diagnostics(self):
        """Main runner that handles diagnostics and generalized visualizations."""
        
        # Model selection based tests
        self.run_convergence_test('Accumulator_Condition')
        self.run_prepruning_test('Accumulator_Condition')

        # Standard Comparison (Random 80/20 Split)
        print("\n--- Running Standard Comparison & Saving to Excel ---")
        std_results = []
        std_with_imp = {}
        std_no_imp = {}
        
        all_features = [f for f in self.df.columns if f not in self.targets + [self.stable_flag]]
        no_temp_features = [f for f in all_features if not f.startswith('TS')]

        for target in self.targets:
            # With Temp
            X, y = self.df[all_features], self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=49)
            model.fit(X_train, y_train)
            
            acc_w = model.score(X_test, y_test)
            std_with_imp[target] = pd.Series(model.feature_importances_, index=all_features)
            std_results.append({'Target': target, 'Type': 'Standard (With Temp)', 'Acc': acc_w})
            self.plot_confusion_matrix(y_test, model.predict(X_test), f"Std_{target}", "standard/with_temp")
            
            # Without Temp
            model_no = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=49)
            model_no.fit(X_train[no_temp_features], y_train)
            acc_n = model_no.score(X_test[no_temp_features], y_test)
            std_no_imp[target] = pd.Series(model_no.feature_importances_, index=no_temp_features)
            std_results.append({'Target': target, 'Type': 'Standard (No Temp)', 'Acc': acc_n})
            self.plot_confusion_matrix(y_test, model_no.predict(X_test[no_temp_features]), f"Std_{target}", "standard/without_temp")

        # Tallenna Standard-tulokset Exceliin
        pd.DataFrame(std_results).to_excel("results/tables/Standard_Comparison_Results.xlsx", index=False)
        self.plot_generalized_importance(std_with_imp, "Standard_With_Temp", "standard/with_temp")
        self.plot_generalized_importance(std_no_imp, "Standard_No_Temp", "standard/without_temp")

        # Stress Tests (80/20 Quantile Split)
        print("\n--- Running Quantile Stress Tests & Saving to Excel ---")
        stress_results = []
        stress_with_imp = {}
        stress_no_imp = {}

        for target in self.targets:
            # Stress With Temp
            res_w, imp_w = self.run_quantile_stress_test(target, use_temp=True)
            stress_with_imp[target] = imp_w
            stress_results.append({**res_w, 'Type': 'Stress (With Temp)'})
            
            # Stress Without Temp
            res_n, imp_n = self.run_quantile_stress_test(target, use_temp=False)
            stress_no_imp[target] = imp_n
            stress_results.append({**res_n, 'Type': 'Stress (No Temp)'})

        
        pd.DataFrame(stress_results).to_excel("results/tables/Stress_Test_Results.xlsx", index=False)
        self.plot_generalized_importance(stress_with_imp, "Stress_With_Temp", "stress/with_temp")
        self.plot_generalized_importance(stress_no_imp, "Stress_No_Temp", "stress/without_temp")

if __name__ == "__main__":
    DATA_FILE = "data/processed/master_data.csv"
    if os.path.exists(DATA_FILE):
        trainer = HydraulicDiagnostic(DATA_FILE)
        trainer.run_all_diagnostics()
