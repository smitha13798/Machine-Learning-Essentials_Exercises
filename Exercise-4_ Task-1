import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def construct_X_optimized(M, alphas, Np=None):
    if Np is None:
        Np = int(np.ceil(np.sqrt(2) * M))
    
    # Convert angles to radians and precompute cos and sin values
    alphas_rad = np.radians(alphas)
    cos_alphas = np.cos(alphas_rad)
    sin_alphas = np.sin(alphas_rad)
    
    # Precompute pixel coordinates
    x = np.linspace(-M//2, M//2, M)
    y = np.linspace(-M//2, M//2, M)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    D = M * M
    pixel_indices = np.arange(D).reshape(M, M)
    
    # Preallocate arrays for indices and weights
    i_indices = []
    j_indices = []
    weights = []
    
    # Loop over each angle, but calculations are vectorized
    for io, (cos_alpha, sin_alpha) in enumerate(zip(cos_alphas, sin_alphas)):
        # Calculate the rotated coordinates for sensor projection
        p = xv * cos_alpha + yv * sin_alpha
        
        # Sensor position indices and ensuring they fall within the valid range
        sensor_positions = np.round(p + Np // 2).astype(int)
        valid = (sensor_positions >= 0) & (sensor_positions < Np)
        valid_positions = sensor_positions[valid]
        valid_pixel_indices = pixel_indices[valid]
        
        # Use broadcasting to fill arrays
        i_indices.extend((valid_positions + io * Np).flatten())
        j_indices.extend(valid_pixel_indices.flatten())
        weights.extend(np.ones(np.sum(valid)))
    
    # Create the sparse matrix in COO format and convert to CSR for better performance
    X = coo_matrix((weights, (i_indices, j_indices)), shape=(len(alphas) * Np, D), dtype=np.float32)
    X_csr = X.tocsr()  # Convert to CSR format for better subsequent matrix operations
    return X_csr

# Example usage
M = 195
alphas = [-90, 0, 90, 180]  # Example angles in degrees
X_optimized = construct_X_optimized(M, alphas, Np=275)
print(X_optimized)
print("Non-zero entries in X:", X_optimized.nnz)
