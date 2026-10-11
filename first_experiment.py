import time
import psutil

# Start measuring time
start_time = time.perf_counter()

# Perform a computation
total = 0

for i in range(10_000_000):
    total += i

# Stop measuring time
end_time = time.perf_counter()

# Calculate execution time
execution_time = end_time - start_time

print("Computation result:", total)
print("Execution time:", execution_time, "seconds")
print("CPU cores:", psutil.cpu_count(logical=True))
print("CPU usage:", psutil.cpu_percent(interval=1), "%")