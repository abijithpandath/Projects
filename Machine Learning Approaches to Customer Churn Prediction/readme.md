    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    df=pd.read_csv('Customer_Churn.csv')
    
    
    # Step 1: Handling Duplicate Entries
    df = df.drop_duplicates()
    
    # Step 2: Handling Missing Values (if any)
    missing_values = df.isnull().sum()
    
    # Step 3: Renaming Columns
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    # Step 4: Outlier Detection and Treatment 
    Q1 = df['seconds_of_use'].quantile(0.25)
    Q3 = df['seconds_of_use'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['seconds_of_use'] = df['seconds_of_use'].clip(upper=upper_bound)
    
    #Step 4: #Dropping Irrelavant columns
    df.drop("customer_value", axis=1, inplace=True)
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    
    #feature importance using correlation
    df.drop('churn', axis=1).corrwith(df.churn).abs().plot(kind='barh',
                                                               figsize=(8, 6),
                                                               color='grey',
                                                               title="Churn vs all Features")
    plt.show()
    
    #Splitting into features and target variable
    X = df.drop('churn', axis=1)  # Drop the 'churn' column to get the features
    Y = df['churn']  # Assign the 'churn' column as the target variable
    
    # Create an instance of the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the features and transform the data
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
    
    # Random Forest Regression
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, Y_train)
    Y_pred_rf = rf_reg.predict(X_test)
    r2_rf = r2_score(Y_test, Y_pred_rf)
    mse_rf = mean_squared_error(Y_test, Y_pred_rf)
    rmse_rf = mean_squared_error(Y_test, Y_pred_rf, squared=False)
    mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
    
    # Decision Tree Regression
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, Y_train)
    Y_pred_tree = decision_tree.predict(X_test)
    r2_tree = r2_score(Y_test, Y_pred_tree)
    mse_tree = mean_squared_error(Y_test, Y_pred_tree)
    rmse_tree = mean_squared_error(Y_test, Y_pred_tree, squared=False)
    mae_tree = mean_absolute_error(Y_test, Y_pred_tree)
    
    # Gradient Boosting Regression
    gradient_boosting = GradientBoostingRegressor()
    gradient_boosting.fit(X_train, Y_train)
    Y_pred_gradient = gradient_boosting.predict(X_test)
    r2_gradient = r2_score(Y_test, Y_pred_gradient)
    mse_gradient = mean_squared_error(Y_test, Y_pred_gradient)
    rmse_gradient = mean_squared_error(Y_test, Y_pred_gradient, squared=False)
    mae_gradient = mean_absolute_error(Y_test, Y_pred_gradient)
    
    # Print the evaluation metrics
    print("Random Forest Regression:")
    print("R-squared value:", r2_rf)
    print("Mean Squared Error:", mse_rf)
    print("Root Mean Squared Error:", rmse_rf)
    print("Mean Absolute Error:", mae_rf)
    print()
    
    print("Decision Tree Regression:")
    print("R-squared value:", r2_tree)
    print("Mean Squared Error:", mse_tree)
    print("Root Mean Squared Error:", rmse_tree)
    print("Mean Absolute Error:", mae_tree)
    print()
    
    print("Gradient Boosting Regression:")
    print("R-squared value:", r2_gradient)
    print("Mean Squared Error:", mse_gradient)
    print("Root Mean Squared Error:", rmse_gradient)
    print("Mean Absolute Error:", mae_gradient)
    
    # Create a list of model names for labeling the plot
    model_names = ['Random Forest Regression', 'Decision Tree Regression', 'Gradient Boosting Regression']
    
    # Create a list of R-squared values for each model
    r2_values = [r2_rf, r2_tree, r2_gradient]
    
    # Create a bar plot to compare the R-squared values
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, r2_values)
    plt.xlabel('Regression Model')
    plt.ylabel('R-squared Value')
    plt.title('Comparison of R-squared Values for Regression Models')
    plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1 for R-squared values
    plt.show()
    
    # Scatter plot of actual vs. predicted values for Random Forest Regression
    plt.figure(figsize=(5, 3))
    plt.scatter(Y_test, Y_pred_rf, color='blue', alpha=0.5)
    plt.xlabel('Actual Churn')
    plt.ylabel('Predicted Churn (Random Forest)')
    plt.title('Actual vs. Predicted Churn (Random Forest Regression)')
    plt.show()
    
    # Scatter plot of actual vs. predicted values for Decision Tree Regression
    plt.figure(figsize=(5, 3))
    plt.scatter(Y_test, Y_pred_tree, color='green', alpha=0.5)
    plt.xlabel('Actual Churn')
    plt.ylabel('Predicted Churn (Decision Tree)')
    plt.title('Actual vs. Predicted Churn (Decision Tree Regression)')
    plt.show()
    
    # Scatter plot of actual vs. predicted values for Gradient Boosting Regression
    plt.figure(figsize=(5, 3))
    plt.scatter(Y_test, Y_pred_gradient, color='red', alpha=0.5)
    plt.xlabel('Actual Churn')
    plt.ylabel('Predicted Churn (Gradient Boosting)')
    plt.title('Actual vs. Predicted Churn (Gradient Boosting Regression)')
    plt.show()
