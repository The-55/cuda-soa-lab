import ctypes
import numpy as np
import time
import subprocess
from ctypes import c_void_p, c_int
import os

# Chargement de la bibliothèque CUDA
try:
    lib = ctypes.CDLL('./libgpuadd.so')
    lib.gpu_add.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]
    lib.gpu_add.restype = None
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA library not available: {e}")
    CUDA_AVAILABLE = False

def gpu_matrix_add(matrix_a, matrix_b):
    """
    Addition de matrices sur GPU via bibliothèque C/CUDA
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Matrices must have same shape")
    
    height, width = matrix_a.shape
    size = width * height
    
    # Conversion en float32 et flatten
    a_flat = matrix_a.astype(np.float32).flatten()
    b_flat = matrix_b.astype(np.float32).flatten()
    c_flat = np.zeros(size, dtype=np.float32)
    
    # Appel de la fonction CUDA
    start_time = time.perf_counter()
    
    lib.gpu_add(
        a_flat.ctypes.data_as(c_void_p),
        b_flat.ctypes.data_as(c_void_p),
        c_flat.ctypes.data_as(c_void_p),
        c_int(width),
        c_int(height)
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # Reshape du résultat
    result = c_flat.reshape((height, width))
    
    return result, elapsed_time

def get_gpu_info():
    """Récupère les informations GPU via nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line and ',' in line:
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_id = parts[0].strip()
                    memory_used = int(parts[1]) if parts[1].strip() else 0
                    memory_total = int(parts[2]) if parts[2].strip() else 0
                    utilization = int(parts[3]) if len(parts) > 3 and parts[3].strip() else 0
                    
                    gpus.append({
                        "gpu": gpu_id,
                        "memory_used_MB": memory_used,
                        "memory_total_MB": memory_total,
                        "utilization_percent": utilization
                    })
        
        return {"gpus": gpus}
    
    except Exception as e:
        return {"error": str(e), "gpus": []}

def compile_cuda_library():
    """Compile la bibliothèque CUDA"""
    try:
        result = subprocess.run([
            'nvcc', '-Xcompiler', '-fPIC', '-shared', '-o', 'libgpuadd.so', 'gpu_service.cu'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Compilation error: {e}")
        return False