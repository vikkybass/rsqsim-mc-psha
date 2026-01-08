"""
SIMPLIFIED Memory Management for RSQSim - FIXED VERSION
Removes complexity while keeping essential monitoring
"""
import psutil
import gc
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryManager:
    """Simple, reliable memory monitoring and cleanup"""
    
    def __init__(self, max_memory_gb: float = 16, warning_threshold: float = 0.8):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
        self.start_time = time.time()
        self.peak_memory = 0
        self.cleanup_count = 0
        
        logger.info(f"🧠 MemoryManager initialized: {max_memory_gb}GB limit, {warning_threshold*100}% warning threshold")
        
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            memory_bytes = self.process.memory_info().rss
            memory_gb = memory_bytes / (1024**3)
            self.peak_memory = max(self.peak_memory, memory_gb)
            return memory_gb
        except Exception as e:
            logger.warning(f"⚠️  Could not get memory usage: {e}")
            return 0.0
    
    def get_system_memory_info(self) -> dict:
        """Get system memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent_used': memory.percent
            }
        except Exception as e:
            logger.warning(f"⚠️  Could not get system memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent_used': 0}
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is approaching limits"""
        current_memory = self.get_memory_usage_gb()
        threshold_memory = self.max_memory_gb * self.warning_threshold
        
        if current_memory > threshold_memory:
            logger.warning(f"⚠️  High memory usage: {current_memory:.2f}GB > {threshold_memory:.2f}GB threshold")
            logger.warning(f"⚠️  Peak memory so far: {self.peak_memory:.2f}GB")
            return True
        
        return False
    
    def force_cleanup(self) -> float:
        """Force garbage collection and memory cleanup"""
        self.cleanup_count += 1
        memory_before = self.get_memory_usage_gb()
        
        logger.info(f"🧹 Forcing memory cleanup #{self.cleanup_count}...")
        logger.info(f"   Memory before cleanup: {memory_before:.2f}GB")
        
        # Multiple cleanup passes for better results
        for i in range(3):
            gc.collect()
        
        # Try to release memory back to OS (Linux/Unix)
        try:
            if hasattr(os, 'sync'):
                os.sync()
        except:
            pass
        
        memory_after = self.get_memory_usage_gb()
        memory_freed = memory_before - memory_after
        
        logger.info(f"   Memory after cleanup: {memory_after:.2f}GB")
        if memory_freed > 0.01:  # Only log if meaningful cleanup
            logger.info(f"   ✅ Freed: {memory_freed:.2f}GB")
        else:
            logger.info(f"   ⚠️  Minimal cleanup achieved")
        
        return memory_after
    
    def log_memory_summary(self):
        """Log comprehensive memory usage summary"""
        runtime = (time.time() - self.start_time) / 60
        current = self.get_memory_usage_gb()
        system_info = self.get_system_memory_info()
        
        logger.info("=" * 60)
        logger.info("🧠 MEMORY USAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Runtime: {runtime:.1f} minutes")
        logger.info(f"📊 Process Memory:")
        logger.info(f"   Current: {current:.2f}GB")
        logger.info(f"   Peak: {self.peak_memory:.2f}GB")
        logger.info(f"   Limit: {self.max_memory_gb:.2f}GB")
        logger.info(f"   Efficiency: {(current/self.max_memory_gb)*100:.1f}% of limit used")
        logger.info(f"🖥️  System Memory:")
        logger.info(f"   Total: {system_info['total_gb']:.1f}GB")
        logger.info(f"   Available: {system_info['available_gb']:.1f}GB")
        logger.info(f"   Used: {system_info['percent_used']:.1f}%")
        logger.info(f"🧹 Cleanup Operations: {self.cleanup_count}")
        logger.info("=" * 60)

class BatchProcessor:
    """Simple batch processing for memory-efficient site processing"""
    
    def __init__(self, batch_size: int = 100, max_events_per_site: int = 10000):
        self.batch_size = batch_size
        self.max_events_per_site = max_events_per_site
        self.memory_manager: Optional[MemoryManager] = None
        
        logger.info(f"📦 BatchProcessor initialized:")
        logger.info(f"   Batch size: {batch_size} sites per batch")
        logger.info(f"   Max events per site: {max_events_per_site:,}")
    
    def set_memory_manager(self, memory_manager: MemoryManager):
        """Attach a memory manager for monitoring during batch processing"""
        self.memory_manager = memory_manager
        logger.info("🔗 Memory manager attached to BatchProcessor")
    
    def log_batch_progress(self, current_batch: int, total_batches: int, 
                          current_site: int, total_sites: int):
        """Log batch processing progress with memory info"""
        progress = (current_site / total_sites) * 100
        
        progress_msg = f"📈 Batch {current_batch}/{total_batches} - {progress:.1f}% complete ({current_site}/{total_sites} sites)"
        
        if self.memory_manager:
            memory_gb = self.memory_manager.get_memory_usage_gb()
            progress_msg += f", Memory: {memory_gb:.2f}GB"
        
        logger.info(progress_msg)
        
        # Check memory limits if manager is available
        if self.memory_manager and self.memory_manager.check_memory_limit():
            logger.warning("🧹 High memory detected during batch processing")
            self.memory_manager.force_cleanup()

# Simple usage example for testing
def test_memory_management():
    """Test the simplified memory management"""
    import numpy as np
    
    print("🧪 Testing simplified memory management...")
    
    # Initialize memory manager
    memory_manager = MemoryManager(max_memory_gb=1, warning_threshold=0.5)  # Low limits for testing
    
    print(f"📊 Initial memory: {memory_manager.get_memory_usage_gb():.3f}GB")
    
    # Create some memory usage
    print("🔄 Allocating memory...")
    large_arrays = []
    for i in range(3):
        arr = np.random.random((500, 500))
        large_arrays.append(arr)
        print(f"   After allocation {i+1}: {memory_manager.get_memory_usage_gb():.3f}GB")
    
    # Test memory limit checking
    print("🔍 Checking memory limits...")
    if memory_manager.check_memory_limit():
        print("⚠️  Memory limit exceeded (as expected for test)")
    
    # Test cleanup
    print("🧹 Testing cleanup...")
    del large_arrays
    memory_manager.force_cleanup()
    print(f"📊 After cleanup: {memory_manager.get_memory_usage_gb():.3f}GB")
    
    # Test batch processor
    print("📦 Testing batch processor...")
    batch_processor = BatchProcessor(batch_size=10, max_events_per_site=1000)
    batch_processor.set_memory_manager(memory_manager)
    batch_processor.log_batch_progress(1, 5, 25, 100)
    
    # Final summary
    memory_manager.log_memory_summary()
    
    print("✅ Memory management test completed!")

if __name__ == "__main__":
    test_memory_management()