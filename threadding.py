
"""
Multi-threaded PWS Runner for Phast Web Services Automation
Handles concurrent dispersion modeling calculations with timeout and memory management
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import threading
import time
import logging
import psutil
import gc
from queue import Queue
from dataclasses import dataclass, field
from typing import List, Callable, Any, Optional, Dict
import json
import os
from datetime import datetime
from main import disp_model


@dataclass
class CalculationTask:
    """Represents a single PWS calculation task"""
    case_id: str
    disp_object: Any
    timeout: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class MultiThreadPWSRunner:
    """
    Multi-threaded runner for PWS calculations with memory management and timeout handling
    """
    
    def __init__(self, max_workers=10, default_timeout=300, max_retries=1, 
                 memory_threshold_gb=7.0, log_level=logging.INFO):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.memory_threshold = memory_threshold_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.results = {}
        self.failed_cases = []
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
    def _setup_logging(self, log_level):
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # File handler with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(f'pws_automation_{timestamp}.log')
            file_handler.setLevel(log_level)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_memory_usage_gb(self):
        """Get current memory usage in GB"""
        memory = psutil.virtual_memory()
        return memory.used / (1024**3)
    
    def check_memory_usage(self):
        """Check current memory usage in bytes"""
        memory = psutil.virtual_memory()
        return memory.used
    
    def log_memory_stats(self):
        """Log detailed memory statistics"""
        memory = psutil.virtual_memory()
        self.logger.info(f"Memory Stats - Used: {memory.used/1024**3:.1f}GB, "
                        f"Available: {memory.available/1024**3:.1f}GB, "
                        f"Usage: {memory.percent:.1f}%")
    
    def wait_for_memory(self, max_wait=300):
        """Wait until memory usage drops below threshold"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            current_usage = self.check_memory_usage()
            if current_usage < self.memory_threshold:
                return True
            
            self.logger.warning(f"Memory usage high: {current_usage / 1024**3:.1f}GB, "
                              f"threshold: {self.memory_threshold / 1024**3:.1f}GB. Waiting...")
            time.sleep(10)
            
            # Force garbage collection while waiting
            gc.collect()
        
        self.logger.error(f"Memory usage remained high after {max_wait}s wait")
        return False
    
    def cleanup_disp_object(self, disp_object):
        """Clean up dispersion object to free memory"""
        try:
            # Try common cleanup methods that PWS objects might have
            cleanup_methods = ['cleanup', 'clear', 'reset', 'close']
            for method_name in cleanup_methods:
                if hasattr(disp_object, method_name):
                    method = getattr(disp_object, method_name)
                    if callable(method):
                        method()
                        self.logger.debug(f"Called {method_name}() on dispersion object")
        except Exception as e:
            self.logger.debug(f"Error during cleanup: {e}")
        finally:
            try:
                del disp_object
            except:
                pass
            gc.collect()
    
    def run_single_calculation(self, task: CalculationTask):
        """Run a single calculation with timeout and memory checks"""
        case_id = task.case_id
        timeout = task.timeout or self.default_timeout
        
        # Check memory before starting
        if not self.wait_for_memory():
            self.logger.error(f"Case {case_id}: Memory threshold exceeded, skipping")
            return {
                'case_id': case_id,
                'result': None,
                'metadata': task.metadata,
                'success': False,
                'error': 'Memory threshold exceeded'
            }
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"Case {case_id}: Starting calculation "
                               f"(attempt {attempt + 1}/{self.max_retries + 1})")
                
                start_time = time.time()
                
                # Run the calculation with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(task.disp_object.run)
                    result = future.result(timeout=timeout)
                
                execution_time = time.time() - start_time
                self.logger.info(f"Case {case_id}: Completed successfully in {execution_time:.1f}s")
                
                # Cleanup
                self.cleanup_disp_object(task.disp_object)
                
                return {
                    'case_id': case_id,
                    'result': result,
                    'metadata': task.metadata,
                    'success': True,
                    'execution_time': execution_time
                }
                
            except TimeoutError:
                self.logger.warning(f"Case {case_id}: Attempt {attempt + 1} timed out after {timeout}s")
                if attempt < self.max_retries:
                    self.logger.info(f"Case {case_id}: Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    self.logger.error(f"Case {case_id}: All retry attempts failed due to timeout")
                    
            except Exception as e:
                self.logger.error(f"Case {case_id}: Failed with error: {str(e)}")
                break
        
        # Cleanup on failure
        self.cleanup_disp_object(task.disp_object)
            
        return {
            'case_id': case_id,
            'result': None,
            'metadata': task.metadata,
            'success': False,
            'error': 'Timeout or execution error'
        }
    
    def run_batch(self, tasks: List[CalculationTask], 
                  progress_callback: Callable = None,
                  save_intermediate_results: bool = True):
        """Run multiple calculations concurrently"""
        
        self.logger.info(f"Starting batch of {len(tasks)} calculations with {self.max_workers} workers")
        self.log_memory_stats()
        
        completed_count = 0
        total_tasks = len(tasks)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_calculation, task): task 
                for task in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                result = future.result()
                completed_count += 1
                
                with self.lock:
                    if result['success']:
                        self.results[result['case_id']] = result
                        if save_intermediate_results:
                            self._save_single_result(result)
                    else:
                        self.failed_cases.append(result['case_id'])
                
                # Log progress
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_count
                eta = avg_time_per_task * (total_tasks - completed_count)
                
                self.logger.info(f"Progress: {completed_count}/{total_tasks} "
                               f"({completed_count/total_tasks*100:.1f}%) "
                               f"Memory: {self.get_memory_usage_gb():.1f}GB "
                               f"ETA: {eta/60:.1f}min")
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed_count, total_tasks, result)
                
                # Periodic memory logging
                if completed_count % 10 == 0:
                    self.log_memory_stats()
        
        total_time = time.time() - start_time
        self.logger.info(f"Batch complete in {total_time/60:.1f} minutes: "
                        f"{len(self.results)} successful, {len(self.failed_cases)} failed")
        
        return self.results, self.failed_cases
    
    def _save_single_result(self, result):
        """Save individual result to file"""
        try:
            results_dir = "pws_results"
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"{results_dir}/{result['case_id']}_result.json"
            with open(filename, 'w') as f:
                # Convert result to JSON-serializable format
                serializable_result = {
                    'case_id': result['case_id'],
                    'success': result['success'],
                    'metadata': result['metadata'],
                    'execution_time': result.get('execution_time'),
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(serializable_result, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save result for {result['case_id']}: {e}")
    
    def save_summary(self, filename=None):
        """Save summary of all results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pws_summary_{timestamp}.json"
        
        summary = {
            'total_cases': len(self.results) + len(self.failed_cases),
            'successful_cases': len(self.results),
            'failed_cases': len(self.failed_cases),
            'success_rate': len(self.results) / (len(self.results) + len(self.failed_cases)) * 100,
            'failed_case_ids': self.failed_cases,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {filename}")
        return filename


def run_large_study_in_batches(all_tasks: List[CalculationTask], 
                              batch_size=50, 
                              max_workers=8):
    """Process very large studies in batches to manage memory"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    all_results = {}
    all_failed = []
    
    total_batches = (len(all_tasks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_tasks), batch_size):
        batch_num = i // batch_size + 1
        batch = all_tasks[i:i+batch_size]
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} cases)")
        
        # Create runner for this batch
        runner = MultiThreadPWSRunner(
            max_workers=max_workers,
            default_timeout=600,
            max_retries=1,
            memory_threshold_gb=7.0
        )
        
        results, failed = runner.run_batch(batch)
        all_results.update(results)
        all_failed.extend(failed)
        
        # Force cleanup between batches
        del runner
        gc.collect()
        time.sleep(2)  # Brief pause between batches
        
        logger.info(f"Batch {batch_num} complete. "
                   f"Running totals: {len(all_results)} successful, {len(all_failed)} failed")
    
    return all_results, all_failed


# Example usage and setup functions
def create_sample_tasks(num_cases=100):
    """Create sample calculation tasks - replace with your actual case setup"""
    tasks = []
    
    for i in range(num_cases):
        # This is where you'd create your actual PWS dispersion object
        # disp = setup_your_dispersion_case(case_parameters)
        
        # Placeholder - replace with your actual dispersion object
        class MockDispObject:
            def __init__(self, case_id):
                self.case_id = case_id
            
            def run(self):
                # Simulate calculation time
                import random
                time.sleep(random.uniform(1, 5))
                return f"Result for {self.case_id}"
        
        disp = MockDispObject(f"case_{i:04d}")
        
        task = CalculationTask(
            case_id=f"case_{i:04d}",
            disp_object=disp,
            timeout=300,  # 5 minutes
            metadata={
                'case_number': i,
                'description': f'Sample case {i}',
                'parameters': {'param1': i * 2, 'param2': i * 3}
            }
        )
        tasks.append(task)
    
    return tasks


def progress_callback(completed, total, result):
    """Example progress callback function"""
    if result['success']:
        print(f"✓ {result['case_id']} completed in {result.get('execution_time', 0):.1f}s")
    else:
        print(f"✗ {result['case_id']} failed: {result.get('error', 'Unknown error')}")


def main():
    """Main execution example"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    )
    
    # Create calculation tasks (replace with your actual case setup)
    print("Setting up calculation tasks...")
    tasks = create_sample_tasks(50)  # Adjust number as needed
    
    # Create runner
    runner = MultiThreadPWSRunner(
        max_workers=10,          # Start with 10, reduce if memory issues
        default_timeout=600,     # 10 minutes default timeout
        max_retries=1,          # Retry once on failure
        memory_threshold_gb=7.0  # Leave 1GB free on 8GB system
    )
    
    print(f"Starting {len(tasks)} calculations with {runner.max_workers} workers...")
    
    # Run the batch
    results, failed_cases = runner.run_batch(
        tasks, 
        progress_callback=progress_callback,
        save_intermediate_results=True
    )
    
    # Save summary
    summary_file = runner.save_summary()
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"Study Complete!")
    print(f"{'='*50}")
    print(f"Total cases: {len(tasks)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed_cases)}")
    print(f"Success rate: {len(results)/(len(results)+len(failed_cases))*100:.1f}%")
    print(f"Summary saved to: {summary_file}")
    
    if failed_cases:
        print(f"\nFailed cases: {failed_cases}")
    
    return results, failed_cases


if __name__ == "__main__":
    results, failed_cases = main()
