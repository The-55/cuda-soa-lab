# ** Session – CI/CD & Monitoring of GPU-Accelerated Microservices**

- **Course:** Service Oriented Architecture (SOA)
- **GitHub Repository:** `https://github.com/HamzaGbada/cuda-soa-lab`

---

## **Learning Objectives**

By the end of this lab, you will:

1. Design and implement a **GPU-based microservice** using **FastAPI** and **CUDA/Numba**.
2. Expose REST endpoints for computation and GPU monitoring.
3. Containerize your service using **Docker (GPU runtime)**.
4. Deploy automatically to the instructor's server using **Jenkins**.
5. Collect and visualize real-time GPU and request metrics using **Prometheus and Grafana**.

---

## **Context**

Service-oriented systems often integrate high-performance GPU-accelerated components — for example, matrix computations or neural inference services. This lab simulates a real DevOps pipeline for such services.

You will:
- Implement a **matrix addition service on GPU**
- Expose metrics about GPU memory and request latency
- Deploy it via **Jenkins pipeline** to a shared GPU server
- Monitor it in **Grafana dashboards**

---

## **Tools & Technologies**

| Tool | Role | Access URL |
|------|------|------------|
| Jenkins | Continuous Deployment | `http://10.90.90.100:8090` |
| Docker + NVIDIA Toolkit | Container runtime | Installed on server |
| Prometheus | Metrics collector | `http://10.90.90.100:9090` |
| Grafana | Visualization dashboard | `http://10.90.90.100:3000` |
| GitHub | Version control | Personal account |
| FastAPI | Web framework | Port 8000+ |
| CUDA/Numba | GPU acceleration | NVIDIA GPU required |

---

## **Lab Setup**

### **1. Fork and Clone the Repository**

```bash
# Fork the template repository on GitHub first
# Then clone your fork locally
git clone https://github.com/<your-username>/cuda-soa-lab.git
cd cuda-soa-lab
```

## Project Structure
```
cuda-soa-lab/
├── main.py                 # FastAPI application
├── gpu_service.cu          # CUDA C++ implementation
├── pyproject.toml          # Python dependencies
├── requirements.txt        # Alternative dependencies
├── Dockerfile             # Container configuration
├── Jenkinsfile            # CI/CD pipeline
├── cuda_test.py           # Testing scripts
├── matrix1.npz            # Test data
├── matrix2.npz            # Test data
└── README.md              # This file
```

##  Environment Setup
pip install -r requirements.txt
pip install .

## Task 1 – GPU Matrix Addition Service

### Implementation Options
```
Option	Language	GPU API	Difficulty
🐍 Path A	Python	Numba (CUDA JIT)	Easier
⚙️ Path B	C/CUDA + FastAPI wrapper	CUDA C via ctypes	Advanced
```
### Service Specification
```
Feature	Description
Endpoint	POST /add
Port	Each student uses different port <student_port>
Input	Two uploaded .npz files containing NumPy matrices
Output	JSON: {"matrix_shape": [rows, cols], "elapsed_time": seconds, "device": "GPU"}
Validation	Reject matrices with different shapes
Health Check	/health → {"status": "ok"}
GPU Info	/gpu-info → GPU memory usage
```

## Task 2 – CUDA Implementation (Path B)
gpu_service.cu - CUDA Kernel

```
#include <cuda_runtime.h>

extern "C" {

__global__ void matrixAddKernel(float* A, float* B, float* C, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

void gpu_add(float* A, float* B, float* C, int width, int height) {
    float *d_A, *d_B, *d_C;
    int size = width * height * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    matrixAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // extern "C"

```
## FastAPI CUDA Wrapper
The wrapper in main.py handles:

- Loading the compiled CUDA library (libgpuadd.so)
- Memory management between CPU and GPU
- Error handling and validation
- Performance timing

## Task 3 – Containerization with Docker

```
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    nvidia-cuda-toolkit \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Compile CUDA library
RUN nvcc -Xcompiler -fPIC -shared -o libgpuadd.so gpu_service.cu

# Expose port
EXPOSE $PORT

# Start application
CMD ["python3", "main.py"]
```
## Task 4 – Jenkins CI/CD Pipeline
Jenkinsfile
```
pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'gpu-matrix-service'
        STUDENT_PORT = '8001'  // Change to your assigned port
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                url: 'https://github.com/<your-username>/cuda-soa-lab.git'
            }
        }
        
        stage('Test CUDA') {
            steps {
                sh '''
                # Test CUDA availability
                python3 -c "
                try:
                    from numba import cuda
                    print('CUDA available:', cuda.is_available())
                except Exception as e:
                    print('CUDA test failed:', e)
                    exit(1)
                "
                '''
            }
        }
        
        stage('Build Docker') {
            steps {
                sh "docker build -t ${DOCKER_IMAGE}:latest ."
            }
        }
        
        stage('Deploy') {
            steps {
                sh """
                # Stop existing container
                docker stop ${DOCKER_IMAGE} || true
                docker rm ${DOCKER_IMAGE} || true
                
                # Start new container
                docker run -d \
                    --name ${DOCKER_IMAGE} \
                    --gpus all \
                    -p ${STUDENT_PORT}:${STUDENT_PORT} \
                    ${DOCKER_IMAGE}:latest
                """
            }
        }
        
        stage('Smoke Test') {
            steps {
                sh """
                sleep 10
                curl -f http://localhost:${STUDENT_PORT}/health || exit 1
                echo "Deployment successful!"
                """
            }
        }
    }
}

```

## Task 5 – Monitoring & Visualization
Prometheus Metrics
- The service automatically exposes metrics at /metrics endpoint:
- request_count_total - Total API requests
- request_latency_seconds - Request duration histogram
- Custom GPU metrics

Grafana Dashboard
Access: http://10.90.90.100:3000
Monitor:

- GPU memory usage per person
- Request latency distribution
- API call frequency
- Error rates
