import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import sklearn.metrics as metrics

class PredWeather:
    def __init__(self, file_name: str, forcast_days: int, target_col: str = "TMAX"):
        self.file_name = file_name
        self.forcast_days = forcast_days
        self.target_col = target_col
            
    def read_data(self):
        df = pd.read_csv(self.file_name, index_col=False)    #no index column
        df.index = df.DATE                              #set index to date column
        df["TARGET"] = df[self.target_col].shift(-self.forcast_days)  #create target column
        print(f"Your data (length of {len(df)}) has been read successfully.")
        self.data = df
        
    def isnull(self):
        """To see how many missing values are in the dataset"""
        print("The percentage of missing values in the dataset is: ")
        print(round(self.data.isnull().sum()/len(self.data)*100, 2))
        
    def drop_columns(self, columns: list):
        """To drop columns from the dataset"""
        try:
            self.data = self.data.drop(columns, axis=1)
        except KeyError:
            print("The columns are not existing or you have already dropped them.")
            
    def dropna(self, how: str = "all"):
        """To drop a row where all values are missing"""
        self.data = self.data.dropna(how=how)
        
    def fillna(self, method: str = "ffill"):
        """To fill missing values in the dataset"""
        if method == "ffill":
            self.data = self.data.ffill()
        if method == "bfill":
            self.data = self.data.bfill()
            
    def split_data(self, test_size: float = 0.2):
        x = self.data.drop("TARGET", axis=1)
        y = self.data["TARGET"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)
        print(f"Your data has been split successfully. The training data has {len(self.x_train)} and the test data has {len(self.x_test)}")
        
    def train_model(self, alpha: float = 0.2):
        reg = Ridge(alpha=0.1)
        reg.fit(self.x_train, self.y_train)
        self.model = reg
        print("Your model has been trained successfully.")
        
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_df = pd.DataFrame(self.y_pred, columns=["PREDICTION"], index=self.y_test.index)
        self.combined = pd.concat([pd.DataFrame(self.y_test), self.y_pred_df], axis=1)
        self.combined.sort_index(inplace=True)
        self.combined.to_csv("combined.csv")
        
    def evaluate(self):
        print(f"The mean squared error is: {metrics.mean_squared_error(self.y_test, self.y_pred)}")
        print(f"The mean absolute error is: {metrics.mean_absolute_error(self.y_test, self.y_pred)}")
        print(f"The r2 score is: {metrics.r2_score(self.y_test, self.y_pred)}")    
            
    