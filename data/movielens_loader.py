import os 
import numpy as np
import pandas as pd

class MovieLensLoader:
    def __init__(self, data_dir="dataset"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one level
        self.data_dir = os.path.join(parent_dir, data_dir)
    
    def load_dataset(self, dataset_name):
        print(f"Loading {dataset_name} dataset...")
        path = os.path.join(self.data_dir, dataset_name, "ratings.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Where is the dataset? This is your path: {path}")
        
        df = pd.read_csv(path, names=["user","item","rating","timestamp"])
        return self._process_dataframe(df)
    
    def _process_dataframe(self, df: pd.DataFrame):
        df = df.drop(columns=["timestamp"])
        
        df["user"] = df["user"].astype(int) - 1
        df["item"] = df["item"].astype(int) - 1
        df["rating"] = df["rating"].astype(float)
        
        n_users = df["user"].max() + 1
        n_items = df["item"].max() + 1
        
        return {
            'ratings_df': df,
            'n_users': n_users,
            'n_items': n_items,
        }
        
    def get_ratings_matrix(self, ratings_df):
        n_users = ratings_df['user'].max() + 1
        n_items = ratings_df['item'].max() + 1
        
        R = np.full((n_users, n_items), np.nan)
        R[ratings_df['user'].values, ratings_df['item'].values] = ratings_df['rating'].values
        
        mask = ~np.isnan(R)
        return R, mask