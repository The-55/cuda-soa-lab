from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io
import time
import ctypes
from ctypes import c_void_p, c_int
import subprocess
import os

# =============================================================================
# CONFIGURATION ÉTUDIANT - CHANGEZ LE PORT SELON VOTRE NUMÉRO
# =============================================================================
STUDENT_PORT = 8001  

# =============================================================================
# WRAPPER CUDA POUR LE FICHIER .cu
# =============================================================================

class CUDAService:
    def __init__(self):
        self.lib = None
        self.cuda_available = False
        self._load_cuda_library()
    
    def _load_cuda_library(self):
        """Charge la bibliothèque CUDA compilée"""
        try:
            # Compilation de la bibliothèque CUDA
            if not os.path.exists('libgpuadd.so'):
                self._compile_cuda_library()
            
            # Chargement de la bibliothèque
            self.lib = ctypes.CDLL('./libgpuadd.so')
            self.lib.gpu_add.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int]
            self.lib.gpu_add.restype = None
            self.cuda_available = True
            print("✅ CUDA library loaded successfully")
        except Exception as e:
            print(f"❌ CUDA library loading failed: {e}")
            self.cuda_available = False
    
    def _compile_cuda_library(self):
        """Compile le fichier CUDA en bibliothèque partagée"""
        try:
            result = subprocess.run([
                'nvcc', '-Xcompiler', '-fPIC', '-shared', '-o', 'libgpuadd.so', 'gpu_service.cu'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CUDA library compiled successfully")
                return True
            else:
                print(f"❌ CUDA compilation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ CUDA compilation error: {e}")
            return False
    
    def matrix_add(self, matrix_a, matrix_b):
        """
        Addition de matrices sur GPU via le wrapper CUDA
        """
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")
        
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Matrices must have same shape")
        
        height, width = matrix_a.shape
        size = width * height
        
        # Conversion en float32 et flatten
        a_flat = matrix_a.astype(np.float32).flatten()
        b_flat = matrix_b.astype(np.float32).flatten()
        c_flat = np.zeros(size, dtype=np.float32)
        
        # Appel de la fonction CUDA avec mesure du temps
        start_time = time.perf_counter()
        
        self.lib.gpu_add(
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

# =============================================================================
# INITIALISATION DU SERVICE CUDA
# =============================================================================
cuda_service = CUDAService()

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title=f"GPU Matrix Service - Port {STUDENT_PORT}",
    description=f"Service d'addition matricielle accéléré GPU - Étudiant port {STUDENT_PORT}",
    version="1.0.0"
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Endpoint racine avec informations du service"""
    return {
        "message": f"GPU Matrix Addition Service - Student Port {STUDENT_PORT}",
        "student_port": STUDENT_PORT,
        "cuda_available": cuda_service.cuda_available
    }

@app.get("/health")
async def health_check():
    """Endpoint de santé - REQUIS PAR LE TP"""
    return {
        "status": "ok",
        "student_port": STUDENT_PORT,
        "cuda_available": cuda_service.cuda_available,
        "timestamp": time.time()
    }

@app.get("/gpu-info")
async def gpu_info():
    """Endpoint d'information GPU - REQUIS PAR LE TP"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line and ',' in line:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        "gpu": parts[0],
                        "memory_used_MB": int(parts[1]),
                        "memory_total_MB": int(parts[2])
                    })
        
        return {
            "student_port": STUDENT_PORT,
            "gpus": gpus
        }
    
    except Exception as e:
        return {
            "student_port": STUDENT_PORT,
            "error": str(e),
            "gpus": []
        }

@app.post("/add")
async def matrix_add(
    file_a: UploadFile = File(..., description="Première matrice au format .npz"),
    file_b: UploadFile = File(..., description="Deuxième matrice au format .npz")
):
    """
    Addition de deux matrices sur GPU - ENDPOINT PRINCIPAL DU TP
    
    Accepte deux fichiers .npz contenant des matrices NumPy.
    Retourne le temps d'exécution et la forme de la matrice résultante.
    """
    # Validation des types de fichiers
    if not file_a.filename.endswith('.npz') or not file_b.filename.endswith('.npz'):
        raise HTTPException(
            status_code=400, 
            detail="Seuls les fichiers .npz sont acceptés"
        )
    
    try:
        # Lecture et chargement des matrices
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        with io.BytesIO(content_a) as buffer_a, io.BytesIO(content_b) as buffer_b:
            matrix_a_data = np.load(buffer_a)
            matrix_b_data = np.load(buffer_b)
            
            # Extraction des tableaux NumPy (gère n'importe quelle clé)
            matrix_a_keys = matrix_a_data.files
            matrix_b_keys = matrix_b_data.files
            
            if not matrix_a_keys or not matrix_b_keys:
                raise HTTPException(
                    status_code=400, 
                    detail="Fichiers .npz invalides - aucun tableau trouvé"
                )
            
            matrix_a = matrix_a_data[matrix_a_keys[0]]
            matrix_b = matrix_b_data[matrix_b_keys[0]]
        
        # Validation des dimensions
        if matrix_a.shape != matrix_b.shape:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimensions incompatibles: {matrix_a.shape} vs {matrix_b.shape}"
            )
        
        # Addition sur GPU via le wrapper CUDA
        result, elapsed_time = cuda_service.matrix_add(matrix_a, matrix_b)
        
        # Réponse conforme au format demandé
        return {
            "matrix_shape": list(matrix_a.shape),
            "elapsed_time": round(elapsed_time, 6),
            "device": "GPU",
            "student_port": STUDENT_PORT
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Erreur GPU: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# =============================================================================
# DÉMARRAGE DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting GPU Matrix Service on port {STUDENT_PORT}")
    print(f"📊 CUDA Available: {cuda_service.cuda_available}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=STUDENT_PORT,  
        log_level="info"
    )