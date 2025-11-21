# **Lab Session – CI/CD & Monitoring of GPU-Accelerated Microservices**

## **Course Information**
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
