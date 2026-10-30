import numpy as np
import pandas as pd

def train_test_split(indices, test_size, random_state):
    np.random.seed(random_state)
    n_test = int(len(indices) * test_size)
    test_index = np.random.choice(indices, n_test, replace=False)
    train_index = np.setdiff1d(indices, test_index)
    return train_index, test_index

class DataPreprocessor:
    def __init__(self, R, mask):
        self.R = R
        self.mask = mask
        self.n_users, self.n_items = R.shape
        
        self.user_ids, self.item_ids = np.where(mask)
        self.ratings = R[self.user_ids, self.item_ids]
        
    def split_train_test(self, test_size=0.1, random_state=42):
        indices = np.arange(len(self.ratings))
        train_index, test_index = train_test_split(indices, test_size=test_size, random_state=random_state)
        
        R_train = np.full_like(self.R, np.nan)
        R_test = np.full_like(self.R, np.nan)
        
        R_train[self.user_ids[train_index], self.item_ids[train_index]] = self.ratings[train_index]
        R_test[self.user_ids[test_index], self.item_ids[test_index]] = self.ratings[test_index]
        
        mask_train = ~np.isnan(R_train)
        mask_test = ~np.isnan(R_test)
        return R_train, mask_train, R_test, mask_test
    
    def split_train_test_by_user(self, n_test_per_user=10, random_state=42):
        np.random.seed(random_state)
        R_train= np.full_like(self.R, np.nan)
        R_test = np.full_like(self.R, np.nan)
        for user in range(self.n_users):
            user_items = np.where(self.mask[user])[0]
            user_ratings = self.R[user, user_items]
            
            if(len(user_items) > n_test_per_user):
                test_index = np.random.choice(len(user_items), n_test_per_user, replace=False)
                train_index = np.setdiff1d(np.arange(len(user_items)), test_index)
                R_train[user, user_items[train_index]] = user_ratings[train_index]
                R_test[user, user_items[test_index]] = user_ratings[test_index]
                
            else:
                R_train[user, user_items] = user_ratings
                
        mask_train = ~np.isnan(R_train)
        mask_test = ~np.isnan(R_test)
        
        return R_train, mask_train, R_test, mask_test
    
    def create_partitions(self, n_partitions, mode='horizontal'):
        partitions = []
        if mode == 'horizontal':
            user_groups = np.array_split(np.arange(self.n_users), n_partitions)
            for users in user_groups:
                R_part = self.R[users]
                mask_part = self.mask[users]
                partitions.append((R_part, mask_part))
                
        elif mode == 'vertical':
            item_groups = np.array_split(np.arange(self.n_items), n_partitions)
            for items in item_groups:
                R_part = self.R[:, items]
                mask_part = self.mask[:, items]
                partitions.append((R_part, mask_part))
                
        return partitions