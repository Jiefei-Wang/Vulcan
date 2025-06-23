import math
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

class BlockTokenizer:
    def __init__(self, dataset, buffer_size, batch_size, tokenizer, device, max_length=512):
        """
        Initialize BlockTokenizer.
        
        Args:
            dataset: CombinedDataset to tokenize
            buffer_size: Size of the block to be tokenized together
            batch_size: Size of each batch
            tokenizer: Tokenizer to use for tokenization
            device: Device to move tensors to
        """
        # Validate batch size is not larger than buffer size
        if batch_size > buffer_size:
            raise ValueError("Batch size cannot be larger than buffer size.")
        
        # bufer size must be a multiple of batch size
        if buffer_size % batch_size != 0:
            raise ValueError("Buffer size must be a multiple of batch size.")
        
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Calculate number of blocks and total batches
        self.num_blocks = math.ceil(len(dataset) / buffer_size)
        self.total_batches = math.ceil(len(dataset) / batch_size)
        
        # Thread pool for async tokenization
        self.executor = ThreadPoolExecutor(max_workers=1)
          # Cache for tokenized blocks (max 2 blocks)
        self.reset_cache()
    
    def __len__(self):
        """Return the number of batches."""
        return self.total_batches
    
    def __getitem__(self, batch_idx):
        """
        Get a batch by index.
        
        Args:
            batch_idx: Index of the batch to retrieve
            
        Returns:
            Dictionary containing tokenized batch data
        """
        if batch_idx >= self.total_batches:
            raise IndexError(f"Batch index {batch_idx} out of range (total batches: {self.total_batches})")
        
        # Calculate which block this batch belongs to
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        
        block_idx = start_idx // self.buffer_size
        
        # Get or create tokenized block (moves to GPU here)
        tokenized_block = self._get_tokenized_block(block_idx)
        
        # Extract batch from the tokenized block
        block_start = block_idx * self.buffer_size
        batch_start_in_block = start_idx - block_start
        batch_end_in_block = end_idx - block_start
        
        # Extract batch from block
        batch_data = {
            'labels': tokenized_block['labels'][batch_start_in_block:batch_end_in_block],
            'tokenized_sentence1s': {
                k: v[batch_start_in_block:batch_end_in_block] 
                for k, v in tokenized_block['tokenized_sentence1s'].items()
            },            
            'tokenized_sentence2s': {
                k: v[batch_start_in_block:batch_end_in_block] 
                for k, v in tokenized_block['tokenized_sentence2s'].items()
            }
        }
        
        return batch_data
    
    def _get_tokenized_block(self, block_idx):
        """
        Get a tokenized block, using cache or creating new one.
        Moves data to GPU when requested.
        
        Args:
            block_idx: Index of the block to get
            
        Returns:
            Dictionary containing tokenized block data on GPU
        """
        # Check if block is already cached on GPU
        if block_idx in self.gpu_cache:
            # Schedule next block for loading if not already scheduled
            self._schedule_next_block(block_idx)
            return self.gpu_cache[block_idx]
        
        # Check if block is being loaded
        if block_idx in self.future_cache:
            # Wait for the future to complete and get CPU data
            cpu_data = self.future_cache[block_idx].result()
            del self.future_cache[block_idx]
            
            # Move to GPU
            gpu_data = self._move_to_gpu(cpu_data)
            
            # Cache on GPU (maintain max 2 blocks)
            self._cache_on_gpu(block_idx, gpu_data)
            
            # Schedule next block for loading
            self._schedule_next_block(block_idx)
            return gpu_data
        
        # Block not cached, need to load synchronously
        cpu_data = self._tokenize_block_cpu(block_idx)
        gpu_data = self._move_to_gpu(cpu_data)
        self._cache_on_gpu(block_idx, gpu_data)
        self._schedule_next_block(block_idx)
        
        return gpu_data
    
    def _schedule_next_block(self, current_block_idx):
        """Schedule the next block for asynchronous loading."""
        next_block_idx = current_block_idx + 1
        
        # Don't schedule if already scheduled/cached or if beyond dataset
        if (next_block_idx >= self.num_blocks or 
            next_block_idx in self.future_cache or 
            next_block_idx in self.gpu_cache):
            return
        
        # Schedule async loading of next block (CPU only)
        future = self.executor.submit(self._tokenize_block_cpu, next_block_idx)
        self.future_cache[next_block_idx] = future
    
    def _cache_on_gpu(self, block_idx, gpu_data):
        """Cache block data on GPU, maintaining max 2 blocks."""
        # Remove oldest cached block if we have 2 already
        if len(self.gpu_cache) >= 2:
            oldest_idx = min(self.gpu_cache.keys())
            del self.gpu_cache[oldest_idx]
            torch.cuda.empty_cache()
        
        self.gpu_cache[block_idx] = gpu_data
    
    def _move_to_gpu(self, cpu_data):
        """Move tokenized data from CPU to GPU."""
        return {k:v.to(self.device) for k, v in cpu_data.items() }
    
    
    def _tokenize_block_cpu(self, block_idx):
        """
        Tokenize a block and return CPU tensors.
        
        Args:
            block_idx: Index of the block to tokenize
            
        Returns:
            Dictionary containing tokenized block data on CPU
        """
        # Get block data
        start_idx = block_idx * self.buffer_size
        end_idx = min(start_idx + self.buffer_size, len(self.dataset))
        
        block_data = self.dataset[start_idx:end_idx]
        
        # Extract sentences and labels
        labels = [item['label'] for item in block_data]
        sentence1s = [item['sentence1'] for item in block_data]
        sentence2s = [item['sentence2'] for item in block_data]
        
        # Tokenize sentences (keep on CPU)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        tokenized_sentence1s = self.tokenizer(
            sentence1s, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
        )
        tokenized_sentence2s = self.tokenizer(
            sentence2s, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
        )
        
        return {
            'labels': labels_tensor,
            'tokenized_sentence1s': tokenized_sentence1s,
            'tokenized_sentence2s': tokenized_sentence2s
        }    
        
    def reset_cache(self):
        """Clear the tokenized cache to free memory."""
        # Clear GPU cache
        self.gpu_cache = {}
        # Clear future cache
        self.future_cache = {}
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
    def cleanup(self):
        """Clean up resources, including the thread pool executor."""
        # Wait for any pending futures to complete
        for future in self.future_cache.values():
            future.cancel()
        
        # Shutdown the executor
        self.executor.shutdown(wait=True)
          # Clear caches
        self.reset_cache()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor
    
    def debug_block(self):
        return f"GPU Cache: {list(self.gpu_cache.keys())}, Future Cache: {list(self.future_cache.keys())}"
