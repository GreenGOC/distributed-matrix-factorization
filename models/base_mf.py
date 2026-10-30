import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, rank=10, reg=0.01, learning_rate=0.01, use_clipping=True):
        self.n_users = n_users
        self.n_items = n_items 
        self.rank = rank
        self.reg = reg
        self.learning_rate = learning_rate
        
        self.R_min = 0.5
        self.R_max = 5.0
        
        self.scale = np.sqrt((self.R_max - self.R_min) / rank)
        
        self.U = np.random.uniform(0, self.scale, size=(n_users, rank))
        self.V = np.random.uniform(0, self.scale, size=(n_items, rank))
        self.b_u = np.full(n_users, self.R_min / 2)
        self.b_i = np.full(n_items, self.R_min / 2)
        
        self.use_clipping = use_clipping
        
    def predict(self, user_index, item_index):
        return self.b_u[user_index] + self.b_i[item_index] + np.dot(self.U[user_index], self.V[item_index])
    
    def compute_loss(self, R, mask):
        predictions = self.b_u[:, None] + self.b_i[None, :] + self.U @ self.V.T
        R_filled = np.nan_to_num(R, nan=0.0)       
        error = (predictions - R_filled) * mask  # NaN * False = 0
        mse = np.sum(error ** 2)
        reg_term = self.reg * (np.sum(self.U ** 2) + np.sum(self.V ** 2) + 
                            np.sum(self.b_u ** 2) + np.sum(self.b_i ** 2))
        return mse + reg_term
    
    def sgd_step(self, user_index, item_index, rating):
        prediction = self.predict(user_index, item_index)
        error = prediction - rating
        
        grad_u = error * self.V[item_index] + self.reg * self.U[user_index]
        grad_v = error * self.U[user_index] + self.reg * self.V[item_index]
        grad_bu = error + self.reg * self.b_u[user_index]
        grad_bi = error + self.reg * self.b_i[item_index]
        
        self.U[user_index] -= self.learning_rate * grad_u
        self.V[item_index] -= self.learning_rate * grad_v
        self.b_u[user_index] -= self.learning_rate * grad_bu
        self.b_i[item_index] -= self.learning_rate * grad_bi
        
        if(self.use_clipping):
            self.U = self._clip_embeddings(self.U)
            self.V = self._clip_embeddings(self.V)
        
    def train(self, R, mask, epochs=10, verbose=True):
        loss_history = []
        
        user_indices, item_indices = np.where(mask)
        ratings = R[user_indices, item_indices]
        for epoch in range(epochs):
            indices = np.arange(len(ratings))
            np.random.shuffle(indices)
            
            for index in indices:
                u = user_indices[index]
                i = item_indices[index]
                r = ratings[index]
                
                self.sgd_step(u, i, r)
            
            loss = self.compute_loss(R, mask)
            loss_history.append(loss)
            
            if verbose or epoch == epochs-1:
                rmse = np.sqrt(loss/np.sum(mask))
                print(f"Epoch {epoch}: Loss = {loss:.2f}, RMSE = {rmse:.4f}")
                
        return loss_history
    
    def evaluate(self, R_test, mask_test):
        loss = self.compute_loss(R_test, mask_test)
        rmse = np.sqrt(loss / np.sum(mask_test))
        return rmse
    
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling

                