import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedCutLossChatGPT(torch.nn.Module):
    def __init__(self, sigma_spatial=16.0, sigma_color=0.1, eps=1e-5, patch_size=16):
        """
        Parameters:
            sigma_spatial: Controls spatial influence in the affinity calculation.
            sigma_color: Controls feature similarity influence in the affinity calculation.
            eps: Small value to avoid division by zero.
            patch_size: Size of local regions for computing affinity, reducing memory usage.
        """
        super().__init__()
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.eps = eps
        self.patch_size = patch_size

    def compute_local_affinity(self, features, spatial_coords):
        """
        Computes local affinity matrices for patches to reduce memory usage.
        Parameters:
            features: Tensor of shape (N, C, H, W).
            spatial_coords: Tensor of shape (N, H, W, 2).
        Returns:
            affinity_matrices: Tensor of shape (N, P, P), where P = patch_size^2.
        """
        N, C, H, W = features.shape
        patches = F.unfold(features, kernel_size=self.patch_size, stride=self.patch_size)  # (N, C*P, L)
        patches = patches.transpose(1, 2)  # (N, L, C*P)

        # Compute pairwise distances in feature space
        feature_diffs = patches.unsqueeze(2) - patches.unsqueeze(3)  # (N, L, P, P, C)
        feature_affinity = torch.exp(-torch.norm(feature_diffs, dim=-1) ** 2 / (2 * self.sigma_color ** 2))  # (N, L, P, P)

        # Compute pairwise distances in spatial space
        spatial_patches = F.unfold(spatial_coords.permute(0, 3, 1, 2), kernel_size=self.patch_size, stride=self.patch_size)  # (N, 2*P, L)
        spatial_patches = spatial_patches.transpose(1, 2).view(N, -1, self.patch_size, self.patch_size, 2)  # (N, L, P, P, 2)
        spatial_diffs = spatial_patches.unsqueeze(3) - spatial_patches.unsqueeze(2)  # (N, L, P, P, 2)
        spatial_affinity = torch.exp(-torch.norm(spatial_diffs, dim=-1) ** 2 / (2 * self.sigma_spatial ** 2))  # (N, L, P, P)

        # Combine feature and spatial affinities
        affinity_matrix = feature_affinity * spatial_affinity
        return affinity_matrix

    def forward(self, predictions, features):
        """
        Computes the normalized cut loss.
        Parameters:
            predictions: Tensor of shape (N, 1, H, W), with sigmoid activation applied.
            features: Tensor of shape (N, C, H, W), feature representation of the input.
        Returns:
            loss: Scalar tensor representing the normalized cut loss.
        """
        N, _, H, W = predictions.shape
        predictions = predictions.view(N, 1, -1)  # (N, 1, H*W)
        spatial_coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, H, device=predictions.device),
            torch.linspace(0, 1, W, device=predictions.device)
        ), dim=-1).repeat(N, 1, 1, 1)  # (N, H, W, 2)

        # Compute local affinity matrices
        affinity_matrices = self.compute_local_affinity(features, spatial_coords)  # (N, L, P, P)

        # Reshape predictions for patches
        prediction_patches = F.unfold(predictions.view(N, 1, H, W), kernel_size=self.patch_size, stride=self.patch_size)  # (N, P, L)
        prediction_patches = prediction_patches.transpose(1, 2).unsqueeze(2)  # (N, L, 1, P)

        # Compute cut and association
        cut = (affinity_matrices * prediction_patches * (1 - prediction_patches.transpose(2, 3))).sum(dim=(2, 3))  # (N, L)
        association = (affinity_matrices * prediction_patches).sum(dim=(2, 3))  # (N, L)

        # Compute normalized cut loss
        loss = cut / (association + self.eps)  # (N, L)
        return loss.mean()  # Scalar loss

class NormalizedCutLossClaude(torch.nn.Module):
    """
    Implementation of the Normalized Cut Loss for weakly-supervised semantic segmentation.
    This loss encourages balanced segmentation by considering both the association within segments
    and the disassociation between different segments.
    
    The loss is computed as: sum_k(cut(Ak,Ak̄)/vol(Ak))
    where Ak is the k-th segment and Ak̄ is its complement.
    """
    def __init__(self, radius=8, sigma_x=8, sigma_i=0.1):
        """
        Initialize the Normalized Cut Loss module.
        
        Args:
            radius (int): Radius for computing the spatial affinity matrix
            sigma_x (float): Sigma for spatial gaussian kernel
            sigma_i (float): Sigma for intensity gaussian kernel
        """
        super().__init__()
        self.radius = radius
        self.sigma_x = sigma_x
        self.sigma_i = sigma_i

    def compute_spatial_kernel(self, height, width, device):
        """
        Compute the spatial affinity kernel based on pixel coordinates.
        
        Args:
            height (int): Height of the feature map
            width (int): Width of the feature map
            device: Device to create tensors on
            
        Returns:
            torch.Tensor: Spatial affinity kernel
        """
        x_grid, y_grid = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Compute pairwise distances
        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)
        
        dist_x = x_grid.unsqueeze(0) - x_grid.unsqueeze(1)
        dist_y = y_grid.unsqueeze(0) - y_grid.unsqueeze(1)
        
        spatial_dist = dist_x.pow(2) + dist_y.pow(2)
        kernel_spatial = torch.exp(-spatial_dist / (2 * self.sigma_x ** 2))
        
        # Apply radius constraint
        mask = spatial_dist > self.radius ** 2
        kernel_spatial[mask] = 0
        
        return kernel_spatial

    def compute_feature_kernel(self, features):
        """
        Compute the feature affinity kernel based on feature similarities.
        
        Args:
            features (torch.Tensor): Input features of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Feature affinity kernel
        """
        b, c, h, w = features.size()
        features = features.view(b, c, -1)  # (B, C, H*W)
        
        # Compute pairwise feature differences
        feature_diff = features.unsqueeze(-1) - features.unsqueeze(-2)  # (B, C, H*W, H*W)
        feature_dist = feature_diff.pow(2).sum(dim=1)  # (B, H*W, H*W)
        
        # Apply Gaussian kernel
        kernel_feature = torch.exp(-feature_dist / (2 * self.sigma_i ** 2))
        
        return kernel_feature

    def forward(self, features, seg_pred):
        """
        Compute the Normalized Cut Loss.
        
        Args:
            features (torch.Tensor): Input features of shape (B, C, H, W)
            seg_pred (torch.Tensor): Predicted segmentation probabilities of shape (B, K, H, W)
                                   where K is the number of classes
        
        Returns:
            torch.Tensor: Normalized Cut Loss value
        """
        b, c, h, w = features.size()
        n_classes = seg_pred.size(1)
        
        # Compute affinity matrices
        kernel_spatial = self.compute_spatial_kernel(h, w, features.device)
        kernel_feature = self.compute_feature_kernel(features)
        
        # Combine spatial and feature kernels
        affinity = kernel_spatial.unsqueeze(0) * kernel_feature  # (B, H*W, H*W)
        
        # Reshape segmentation predictions
        seg_pred = seg_pred.view(b, n_classes, -1)  # (B, K, H*W)
        
        # Initialize loss
        loss = torch.tensor(0., device=features.device)
        
        # Compute normalized cut loss for each class
        for k in range(n_classes):
            seg_class = seg_pred[:, k]  # (B, H*W)
            
            # Compute association (within segments)
            association = torch.bmm(
                seg_class.unsqueeze(1),  # (B, 1, H*W)
                affinity * seg_class.unsqueeze(2)  # (B, H*W, H*W)
            ).squeeze(1)  # (B, H*W)
            
            # Compute volume (total connectivity)
            volume = torch.bmm(
                seg_class.unsqueeze(1),  # (B, 1, H*W)
                affinity.sum(dim=2, keepdim=True)  # (B, H*W, 1)
            ).squeeze(1)  # (B, H*W)
            
            # Compute normalized cut
            ncut = association.sum(dim=1) / (volume.sum(dim=1) + 1e-8)
            loss = loss - ncut.mean()
        
        return loss / n_classes

# Example Usage
if __name__ == "__main__":
    batch_size, channels, height, width = 2, 64, 32, 32
    predictions = torch.sigmoid(torch.randn(batch_size, 1, height, width))  # Simulated prediction
    features = torch.randn(batch_size, channels, height, width)  # Feature representations

    loss_fn = NormalizedCutLossChatGPT(patch_size=8)
    loss = loss_fn(predictions, features)
    print(f"Normalized Cut Loss: {loss.item()}")
