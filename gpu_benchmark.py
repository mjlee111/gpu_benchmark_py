import torch
import time
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm
def gpu_benchmark(max_test_number, epochs_per_test):
    if not torch.cuda.is_available():
        print("GPU is not available")
        return
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) 
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3) 
    print(f"Using GPU: {device_name}")
    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"GPU Memory Usage: {allocated_memory:.2f} GB")
    
    matrix_sizes = [500 * i for i in range(1, max_test_number)]
    times = []
    total_times = []
    gpu_usages = []  
    temp_mins = [] 
    temp_maxs = []  
    temp_avgs = [] 
    
    result_dir = f'./results/{device_name}'
    os.makedirs(result_dir, exist_ok=True) 
    
    for size in matrix_sizes:
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        
        initial_memory = torch.cuda.memory_allocated(0) / (1024**3)
        temperatures = []
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in tqdm(range(epochs_per_test), desc="Running Matmul " + str(size) + "x" + str(size)):
            c = torch.matmul(a, b)
            temp = float(os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader').read())
            temperatures.append(temp)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        final_memory = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_usage = final_memory - initial_memory
        
        total_time = end_time - start_time
        avg_time = total_time / epochs_per_test
        
        times.append(avg_time)
        total_times.append(total_time)
        gpu_usages.append(gpu_usage)
        temp_mins.append(min(temperatures))
        temp_maxs.append(max(temperatures))
        temp_avgs.append(sum(temperatures) / len(temperatures))
        
        print(f"Matrix Size: {size}x{size}")
        print(f"Epochs: {epochs_per_test}")
        print(f"Average Time: {avg_time:.4f} seconds, Total Time: {total_time:.4f} seconds")
        print(f"GPU Usage: {gpu_usage:.2f} GB")
        print(f"Temperature - Min: {min(temperatures)}°C, Max: {max(temperatures)}°C, Avg: {sum(temperatures)/len(temperatures):.1f}°C\n")
    
    with open(f'{result_dir}/gpu_benchmark_result_{device_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Matrix Size', 'Epochs', 'Average Time (s)', 'Total Time (s)', 
                        'GPU Usage (GB)', 'Min Temp (°C)', 'Max Temp (°C)', 'Avg Temp (°C)'])
        for i, size in enumerate(matrix_sizes):
            writer.writerow([f"{size}x{size}", f"{epochs_per_test}", f"{times[i]:.4f}", f"{total_times[i]:.4f}",
                           f"{gpu_usages[i]:.2f}", f"{temp_mins[i]:.1f}", 
                           f"{temp_maxs[i]:.1f}", f"{temp_avgs[i]:.1f}"])
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(matrix_sizes, times, 'bo-')
    plt.title('Average Time per Operation')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Average Time (seconds)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(matrix_sizes, total_times, 'ro-')
    plt.title(f'Total Time for {epochs_per_test} Operations')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(matrix_sizes, gpu_usages, 'go-')
    plt.title('GPU Memory Usage')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Memory Usage (GB)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(matrix_sizes, temp_mins, 'b-', label='Min Temperature')
    plt.plot(matrix_sizes, temp_maxs, 'r-', label='Max Temperature')
    plt.plot(matrix_sizes, temp_avgs, 'g-', label='Avg Temperature')
    plt.title('GPU Temperature')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{result_dir}/gpu_benchmark_result_{device_name}.png')
    plt.show()

if __name__ == "__main__":
    input("Press Enter to start GPU benchmark...")
    max_test_number = int(input("Enter max test number:"))
    epochs_per_test = int(input("Enter epochs per test:"))
    gpu_benchmark(max_test_number, epochs_per_test) 