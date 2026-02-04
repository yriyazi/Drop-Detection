import time
import tracemalloc
from functools import wraps

def measure_performance(func):
    """
    <p style="background-color:tomato;"><b> Decorator:</b></p> that measures the execution time and memory usage of a function.
    
    It uses:
    - `time.perf_counter()` for high-resolution timing.
    - `tracemalloc` to track memory allocations during execution.
    
    Args:
        func (Callable): The function to be wrapped.
    
    Returns:
        Callable: The wrapped function with performance tracking.
    """
    @wraps(func)  # Preserve function metadata like __name__ and __doc__
    def wrapper(*args, **kwargs):
        # Start tracking memory allocations
        tracemalloc.start()
        
        # Record start time
        start_time = time.perf_counter()
        
        # Execute the target function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.perf_counter()
        
        # Retrieve current and peak memory usage during execution
        current, peak = tracemalloc.get_traced_memory()
        
        # Stop tracking memory allocations
        tracemalloc.stop()
        
        # Report the performance metrics
        print(f"[{func.__name__}] Execution time: {end_time - start_time:.6f} seconds")
        print(f"[{func.__name__}] Current memory usage: {current / 1024:.2f} KiB")
        print(f"[{func.__name__}] Peak memory usage: {peak / 1024:.2f} KiB")
        
        # Return the result of the original function
        return result

    return wrapper


def average_performance(runs:int=1000):
    """
    <p style="background-color:tomato;"><b> Decorator:</b></p> factory that returns a decorator to measure average performance
    of a function over multiple runs.
    
    It reports:
    - Average execution time over all runs
    - Average peak memory usage
    
    Args:
        runs (int): Number of times to run the function for averaging (default: 1000)
    
    Returns:
        Callable: The performance-measuring decorator.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            total_time = 0.0             # Accumulator for total execution time
            total_peak_memory = 0.0      # Accumulator for total peak memory usage

            for i in range(runs):
                # Start tracking memory
                tracemalloc.start()
                
                # Record start time
                start_time = time.perf_counter()

                # Execute the function
                result = func(*args, **kwargs)

                # Record end time
                end_time = time.perf_counter()

                # Get peak memory usage (in bytes)
                _, peak = tracemalloc.get_traced_memory()
                
                # Stop memory tracking
                tracemalloc.stop()

                # Update accumulators
                total_time += (end_time - start_time)
                total_peak_memory += peak

            # Calculate averages
            avg_time = total_time / runs
            avg_peak_memory = total_peak_memory / runs / 1024  # Convert to KiB

            # Print averaged performance results
            print(f"[{func.__name__}] Ran {runs} times")
            print(f"[{func.__name__}] Avg execution time: {avg_time:.6f} seconds")
            print(f"[{func.__name__}] Avg peak memory usage: {avg_peak_memory:.2f} KiB")

            return result  # Return the result of the last run

        return wrapper
    return decorator
