
import torch
import torch.nn as nn
import numpy as np
from typing import List, Set, Tuple
import json
from datetime import datetime
import os


def save_dataset(X: torch.Tensor, y: torch.Tensor, path: str, rank: int, min_size_bytes: int = 1000):
    """Helper function to save dataset with built-in verification"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'X': X.cpu(),
            'y': y.cpu(),
            'shape_X': X.shape,
            'shape_y': y.shape,
            'saved_by_rank': rank,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(save_dict, path)
        
        # Verify the save
        if not os.path.exists(path):
            raise RuntimeError(f"File does not exist after save: {path}")
        if os.path.getsize(path) < min_size_bytes:
            raise RuntimeError(f"File too small after save: {path} ({os.path.getsize(path)} bytes)")
            
        print(f"Rank {rank}: Successfully saved and verified dataset at {path}")
        return True
    except Exception as e:
        print(f"Rank {rank}: Error saving dataset: {e}")
        return False


def save_results(results: List[dict], results_dir: str, timestamp: str):
    """Helper function to save results with error handling"""
    try:
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")

def save_model(model: nn.Module, path: str):
    """Helper function to save model with error handling"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save state dict directly without moving model
        torch.save(model.state_dict(), path)
    except Exception as e:
        print(f"Error saving model: {e}")

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]], device=None):
        self.P = P
        self.sets = sets
        self.device = device
    
    def to(self, device):
        """Move function to specified device"""
        self.device = device
        return self
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        if self.device is not None:
            z = z.to(self.device)
        
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result
    
class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]], device=None):
        self.P = P
        self.sets = sets
        self.device = device
    
    def to(self, device):
        """Move function to specified device"""
        self.device = device
        return self
    
    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the correct device
        if self.device is not None:
            z = z.to(self.device)
        
        device = z.device
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        for S in self.sets:
            term = torch.ones(batch_size, dtype=torch.float32, device=device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term
            
        return result
    


def generate_master_dataset(P, d, master_size, n_test, msp, seed=42):
    """Generate master training set and test set with fixed seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate master training set
    X_train_master = (2 * torch.bernoulli(0.5 * torch.ones((master_size, d), dtype=torch.float32)) - 1).to(device)
    y_train_master = msp.evaluate(X_train_master)
    
    # Generate test set
    X_test = (2 * torch.bernoulli(0.5 * torch.ones((n_test, d), dtype=torch.float32)) - 1).to(device)
    y_test = msp.evaluate(X_test)
    
    return X_train_master, y_train_master, X_test, y_test