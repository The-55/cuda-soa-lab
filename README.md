# **Lab Session – CI/CD & Monitoring of GPU-Accelerated Microservices**
## **Learning Objectives**

By the end of this lab, you will:

1. Design and implement a **GPU-based microservice** using **FastAPI** and **CUDA/Numba**.
2. Expose REST endpoints for computation and GPU monitoring.
3. Containerize your service using **Docker (GPU runtime)**.
4. Deploy automatically to the instructor’s server using **Jenkins**.
5. Collect and visualize real-time GPU and request metrics using **Prometheus and Grafana**.

---

## **Context**

Service-oriented systems often integrate high-performance GPU-accelerated components — for example, matrix computations or neural inference services.
This lab simulates a real DevOps pipeline for such services.
