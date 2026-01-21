import pandas as pd
import numpy as np

def vdp(data: pd.DataFrame, intercept: bool, log_variables = None) -> pd.DataFrame:

    """
    Computes the Variance Decomposition Proportions (V.D.P.) as illustrated in the following book: Besley (1991) "Conditioning Diagonostics. Collinearity and Weak Data in Regression". Wiley. 
    V.D.P. is meant to be a collineary diagnostic tool for linear regression. With this tool, you have a greater ability to see which variables are collinear with one another and to what degree.

    Args:
        data (pd.DataFrame): A dataframe with only continuous input variables that are linear/log based. Data must be cleaned (e.g. no missing values). 
        intercept (bool): Indicate if you want to include an intercept term based on what you want to do with your regression analysis.
        log_variables (None): If it is the case that you must log transform an input variable that will lose its meaning/interpretability, list the relevant variables. For more information, read Ch.9 in Besley's book.
    
    Returns:
        result (pd.Dataframe): A table that displays the variance decomposition proportions for regression variables as well as the condition indexes.
    """

    #Create copy to prevent shenanigans
    df = data.copy(deep=True)

    # Insert Column of 1s
    if intercept:
        df.insert(0, 'intercept', 1)

    # Log Base e Scaling to help with interpretation issues with log-transformed variables
    def log_scale(columns: pd.Series) -> pd.Series:
        return np.e / np.exp(np.log(columns).mean())
    if log_variables is not None:
        df[log_variables] = df[log_variables].apply(log_scale, axis=0)

    # Unit Length Scaling 
    numpy_array = df.to_numpy()
    column_normalization = np.linalg.norm(numpy_array, axis=0).reshape(1,-1)
    numpy_array = numpy_array / column_normalization

    # SVD
    u, d, vt = np.linalg.svd(numpy_array)
    singular_values = np.array([d[i] for i in range(len(d))]).reshape(1,-1)

    # Calculate Variance Decomposition Proportions
    variance_components = (vt.T * vt.T) / (singular_values ** 2)
    max_singular_value = np.max(singular_values)
    scaled_condition_indexes = max_singular_value / singular_values
    phi = np.sum(variance_components, axis=1).reshape(-1,1)
    pi = (variance_components / phi).T
    result = pd.DataFrame(data=pi, columns=df.columns)
    scaled_condition_indexes = scaled_condition_indexes = scaled_condition_indexes.flatten().tolist()
    result.insert(0, 'scaled_condition_indexes', scaled_condition_indexes)
    return result

