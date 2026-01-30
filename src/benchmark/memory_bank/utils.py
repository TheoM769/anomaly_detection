import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import faiss
from sklearn.manifold import TSNE
import umap

def dists2map(dists, img_shape):
    # resize and smooth the distance map
    # caution: cv2.resize expects the shape in (width, height) order (not (height, width) as in numpy, so indices here are swapped!
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation = cv2.INTER_LINEAR)
    dists = gaussian_filter(dists, sigma=4)
    return dists


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    mask = np.repeat(mask, imgd2, axis=1)
    return mask


def plot_ref_images(img_list, mask_list, grid_size, save_path, title = "Reference Images", img_names = None):
    k = min(len(img_list), 32)  # reduce max number of ref samples to plot to 32

    #n_aug = len(img_list)//len(img_names)

    fig, axs = plt.subplots(k, 2, figsize=(10, 3.5*k))
    if k == 1:
        axs = axs.reshape(1, -1)
    for i in range(k):
        axs[i, 0].imshow(img_list[i])
        axs[i, 1].imshow(img_list[i])
        axs[i, 1].imshow(resize_mask_img(mask_list[i], img_list[i].shape, grid_size), alpha=0.5)
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        # if i % n_aug == 0:
        #     axs[i, 0].title.set_text(f"Image: {img_names[i // n_aug]}")
        # else:
        #     axs[i, 0].title.set_text(f"Augmentation of Image {img_names[i // n_aug]}")
        #axs[i, 1].title.set_text("PCA + Mask")
        axs[i, 0].title.set_text(f"Reference Image {i}")
        axs[i, 1].title.set_text("Mask")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "reference_samples.png")
    plt.close()

def build_knn_index(features_ref, faiss_on_cpu=True, knn_metric="L2_normalized"):
    if faiss_on_cpu:
        # similariy search on CPU
        knn_index = faiss.IndexFlatL2(features_ref.shape[1])
    else:
        # similariy search on GPU
        res = faiss.StandardGpuResources()
        knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
        # knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        # knn_index = faiss.index_cpu_to_gpu(res, int(model.device[-1]), knn_index)


    if knn_metric == "L2_normalized":
        faiss.normalize_L2(features_ref)
    knn_index.add(features_ref)
    return knn_index

def compute_distances(knn_index, masked_features, knn_metric, knn_neighbors, mean_over_neighbors=True):
    # Compute distances to nearest neighbors in M
    if knn_metric == "L2":
        distances, indices = knn_index.search(masked_features, k = knn_neighbors)
        if knn_neighbors > 1 and mean_over_neighbors:
            distances = distances.mean(axis=1)
        distances = np.sqrt(distances)

    elif knn_metric == "L2_normalized":
        faiss.normalize_L2(masked_features) 
        distances, indices = knn_index.search(masked_features, k = knn_neighbors)
        if knn_neighbors > 1 and mean_over_neighbors:
            distances = distances.mean(axis=1)
        distances = distances / 2   # equivalent to cosine distance (1 - cosine similarity)

    return distances

def plot_memory_bank_tsne(memory_bank_features, new_samples=None, save_path=None, perplexity=30, n_components=2, random_state=42, title="Memory Bank t-SNE Visualization"):
    """
    Create a t-SNE plot of memory bank features with optional new samples overlay.
    
    Args:
        memory_bank_features (np.ndarray): Memory bank features of shape (N, D) where N is number of samples and D is feature dimension
        new_samples (np.ndarray, optional): New sample features of shape (M, D) to overlay on the plot
        save_path (str, optional): Path to save the plot. If None, displays the plot
        perplexity (float): t-SNE perplexity parameter, should be between 5 and 50
        n_components (int): Number of components for t-SNE (2 or 3)
        random_state (int): Random state for reproducibility
        title (str): Title for the plot
        
    Returns:
        tuple: (memory_bank_tsne, new_samples_tsne) where new_samples_tsne is None if new_samples not provided
    """
    print(f"Computing t-SNE for {memory_bank_features.shape[0]} memory bank features with {memory_bank_features.shape[1]} dimensions...")
    
    # Combine features for fitting t-SNE if new samples provided
    if new_samples is not None:
        print(f"Including {new_samples.shape[0]} new samples in t-SNE computation...")
        combined_features = np.vstack([memory_bank_features, new_samples])
        n_memory_bank = memory_bank_features.shape[0]
    else:
        combined_features = memory_bank_features
        n_memory_bank = memory_bank_features.shape[0]
    
    # Apply t-SNE on combined features
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, verbose=1)
    tsne_features = tsne.fit_transform(combined_features)
    
    # Split back into memory bank and new samples
    memory_bank_tsne = tsne_features[:n_memory_bank]
    new_samples_tsne = tsne_features[n_memory_bank:] if new_samples is not None else None
    
    # Create the plot
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        # Plot memory bank features
        plt.scatter(memory_bank_tsne[:, 0], memory_bank_tsne[:, 1], alpha=0.6, s=50, c='blue', label='Memory Bank')
        # Plot new samples if provided
        if new_samples_tsne is not None:
            plt.scatter(new_samples_tsne[:, 0], new_samples_tsne[:, 1], alpha=0.8, s=80, c='red', label='New Samples', marker='^')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        # Plot memory bank features
        ax.scatter(memory_bank_tsne[:, 0], memory_bank_tsne[:, 1], memory_bank_tsne[:, 2], alpha=0.6, s=50, c='blue', label='Memory Bank')
        # Plot new samples if provided
        if new_samples_tsne is not None:
            ax.scatter(new_samples_tsne[:, 0], new_samples_tsne[:, 1], new_samples_tsne[:, 2], alpha=0.8, s=80, c='red', label='New Samples', marker='^')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    
    plt.close()
    
    return memory_bank_tsne, new_samples_tsne

def plot_memory_bank_umap(memory_bank_features, new_samples=None, save_path=None, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, title="Memory Bank UMAP Visualization"):
    """
    Create a UMAP plot of memory bank features with optional new samples overlay.
    
    Args:
        memory_bank_features (np.ndarray): Memory bank features of shape (N, D) where N is number of samples and D is feature dimension
        new_samples (np.ndarray, optional): New sample features of shape (M, D) to overlay on the plot
        save_path (str, optional): Path to save the plot. If None, displays the plot
        n_neighbors (int): Number of neighbors to consider for each point (UMAP parameter)
        min_dist (float): Minimum distance between points in the low-dimensional representation
        n_components (int): Number of components for UMAP (2 or 3)
        random_state (int): Random state for reproducibility
        title (str): Title for the plot
        
    Returns:
        tuple: (memory_bank_umap, new_samples_umap) where new_samples_umap is None if new_samples not provided
    """
    print(f"Computing UMAP for {memory_bank_features.shape[0]} memory bank features with {memory_bank_features.shape[1]} dimensions...")
    
    # Combine features for fitting UMAP if new samples provided
    if new_samples is not None:
        print(f"Including {new_samples.shape[0]} new samples in UMAP computation...")
        combined_features = np.vstack([memory_bank_features, new_samples])
        n_memory_bank = memory_bank_features.shape[0]
    else:
        combined_features = memory_bank_features
        n_memory_bank = memory_bank_features.shape[0]
    
    # Apply UMAP on combined features
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state, verbose=True)
    umap_features = reducer.fit_transform(combined_features)
    
    # Split back into memory bank and new samples
    memory_bank_umap = umap_features[:n_memory_bank]
    new_samples_umap = umap_features[n_memory_bank:] if new_samples is not None else None
    
    # Create the plot
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        # Plot memory bank features
        plt.scatter(memory_bank_umap[:, 0], memory_bank_umap[:, 1], alpha=0.6, s=50, c='blue', label='Memory Bank')
        # Plot new samples if provided
        if new_samples_umap is not None:
            plt.scatter(new_samples_umap[:, 0], new_samples_umap[:, 1], alpha=0.8, s=80, c='red', label='New Samples', marker='^')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        # Plot memory bank features
        ax.scatter(memory_bank_umap[:, 0], memory_bank_umap[:, 1], memory_bank_umap[:, 2], alpha=0.6, s=50, c='blue', label='Memory Bank')
        # Plot new samples if provided
        if new_samples_umap is not None:
            ax.scatter(new_samples_umap[:, 0], new_samples_umap[:, 1], new_samples_umap[:, 2], alpha=0.8, s=80, c='red', label='New Samples', marker='^')
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f"UMAP plot saved to {save_path}")
    
    plt.close()
    
    return memory_bank_umap, new_samples_umap
