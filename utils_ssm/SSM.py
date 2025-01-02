import numpy as np
import torch
from sklearn.decomposition import PCA
from warnings import warn
from typing import Any, Tuple
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
def check_data_scale(dataset: np.ndarray) -> None:
    """Check that data is appropriate for PCA, meaning that each sample has 0
    mean and 1 std.

    Parameters
    ----------
    dataset : array_like
        N-dimension array of data to model, where each row on the first axis is one sample
        and each column on the second axis is a landmark value

    Returns
    -------
    None

    Raises
    ------
    Warning
        If mean of each sample in dataset not equal to 0
    Warning
        If standard deviation of each sample in dataset not equal to 1
    """
    if np.allclose(dataset.mean(axis=1), 0):
        pass
    else:
        warn("Dataset mean should be 0, " f"is equal to {dataset.mean(axis=1)}")

    if np.allclose(dataset.std(axis=1), 1):
        pass
    else:
        warn(
            "Dataset standard deviation should be 1, "
            f"is equal to {dataset.std(axis=1)}"
        )


class SSM:
    def __init__(self, correspondences: np.ndarray, max_mode=None) -> None:
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes
        """

        # check dataset
        # check_data_scale(correspondences)

        self.mean = np.mean(correspondences, 0)
        print("see self.mean shape:",self.mean.shape)
        self.max_mode = 128



        data_centered = correspondences - self.mean
        # print("data_centered shape", data_centered.shape)
        # print("correspondences shape", correspondences.shape)
        # print("mean shape ", self.mean.shape)

        # data_centered = data_centered.transpose()
        cov_dual = np.matmul(data_centered.transpose(), data_centered) / (
            data_centered.shape[0] - 1
        )
        # print("cov_dual shape", cov_dual.shape)
        evals, evectors = np.linalg.eigh(cov_dual)
        # print("evals shape", evals.shape) # evals shape (3000,)
        # print("evecs shape", evectors.shape) # (3072, 3072)
        # evecs = np.matmul(data_centered, evectors)
        # exit()
        evecs = evectors
        # Normalize the col-vectors
        evecs /= np.sqrt(np.sum(np.square(evecs), 0))

        # Sort
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]

        # new_envals = []
        # for idx, eigenvector in enumerate(evecs.T):
        #     print(np.dot(eigenvector.T, np.dot(cov_dual, eigenvector)))
        #     print("evals[idx]", evals[idx])
        #     exit()

        # Remove the last eigen pair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.variances[self.variances < 0] = 0
        self.modes_norm = evecs[:, :-1]
        # print("modes_norm shape", self.modes_norm.shape)
        # modes_norm shape (3072, 3071)
        # exit()
        # Compute the modes scaled by corresp. std. dev.
        # print("self.variances vals:", self.variances)
        # plot self.variances vals
        # Plotting

        variances = np.array(self.variances)# TODO: !!!!!!!!!!!!!!!!!!use it in diffusion model!!!!!!!!!!!!!!!!!!

        self.eigenvals = variances

        # Calculate the total sum of variances
        total_variance = np.sum(variances)
        # print("total_variance", total_variance)
        # Compute the cumulative sum of variances
        cumulative_variance = np.cumsum(variances)

        # Find the number of components for 99% of total variance
        required_mode_number = np.where(cumulative_variance >= 0.999999 * total_variance)[0][0] + 1

        self.required_components_number = required_mode_number
        print("Required_mode_number for saving 99.9999% info:", required_mode_number)

        variance_64_modes = (np.sum(variances[:128]) / total_variance) * 100
        print(f"Variance explained by first 128 modes: {variance_64_modes:.4f}%")

        variance_64_modes = (np.sum(variances[:64]) / total_variance) * 100
        print(f"Variance explained by first 64 modes: {variance_64_modes:.4f}%")

        variance_32_modes = (np.sum(variances[:32]) / total_variance) * 100
        print(f"Variance explained by first 32 modes: {variance_32_modes:.4f}%")

        variance_16_modes = (np.sum(variances[:16]) / total_variance) * 100
        print(f"Variance explained by first 16 modes: {variance_16_modes:.4f}%")


        # self.modes_norm = self.modes_norm[:, :required_mode_number]
        self.variances_mode_number = self.variances[:required_mode_number]
        # self.variances[required_mode_number-1] = 0
        # # self.variances = self.variances[:required_mode_number]
        # print("len(self.variances)", len(self.variances))
        # print("required_mode_number", required_mode_number)

        # plt.figure(figsize=(10, 6))
        # plt.plot(variances, marker='o', linestyle='-', markersize=5)
        # plt.title('Variances by Principal Component')
        # plt.xlabel('Principal Component Index')
        # plt.ylabel('Variance Explained')
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.xticks(range(len(variances)))  # Setting x-tick labels to match the number of components
        # plt.tight_layout()
        # # Show the plot
        # plt.show()

        # print("self.variances shape:", self.variances.shape)
        # print("val of self.variances:", self.variances[:10])



        # print("self.modes_norm shape:", self.modes_norm.shape)
        # print("self.variances shape:", self.variances.shape)
        # # self.modes_norm:shape: (3000, 2999)
        # # self.variances: shape: (2999,)
        self.modes_scaled = np.multiply(self.modes_norm, np.sqrt(self.variances))
        # print("------------>self.modes_scaled shape", self.modes_scaled.shape)
        # self.modes_norm: shape: (3000, 2999)
        # # perform principal component analysis to train shape model
        # self.pca_object, self.required_mode_number = self.do_pca(
        #     data_centered.transpose(), desired_variance
        # )
        #
        # print("explained_variance_ndarray", self.pca_object.explained_variance_.shape)
        #
        # # get principal components (eigenvectors) and variances (eigenvalues)
        # self.pca_model_components = self.pca_object.components_
        # print("pca_model_components shape", self.pca_model_components.shape)
        # self.variance = self.pca_object.explained_variance_
        # print("variance shape", self.variance.shape)
        # # get standard deviation of each component,
        # # and initialise model parameters as zero
        # self.std = np.sqrt(self.variance)
        # print("std shape", self.std.shape)
        # self.model_parameters = np.zeros(len(self.std))
        # print("model_parameters shape", self.model_parameters.shape)
        # # save desired_variance as global variable to help post-processing and debugging
        # self.desired_variance = desired_variance
        # print("desired_variance shape", self.desired_variance)

    def save_ssm(self, path: str) -> None:
        """
        Save the SSM to a file
        Args: path
        """
        np.savez(
            path,
            mean=self.mean,                  # mean shape
            modes_norm=self.modes_norm[:, :self.max_mode],      # eigenvectors
            # modes_scaled=self.modes_scaled,
            required_components_number = self.required_components_number,
            variances=self.variances[:self.max_mode],        # eigenvalues
            variance_mode_number=self.variances_mode_number,
        )
    def get_variance_num_modes(self, num_modes: int) -> np.ndarray:
        """
        Get the variance of the first num_modes modes
        Args:
            num_modes:  number of modes to consider
        Returns:
            variance:   variance of the first num_modes modes
        """
        return self.variances[:num_modes]

    def get_theta(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        # print("weights shape:", weights.shape)
        return weights

    def get_scaled_theta(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        print("shape of shape:", shape.shape)
        print("shape of mean:", self.mean.shape)
        # exit(0)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            print("if n_modes: self.modes_scaled shape:", self.modes_scaled.shape)
            evecs = self.modes_scaled[:, :n_modes]
        else:
            print("else: self.modes_scaled shape:", self.modes_scaled.shape)
            evecs = self.modes_scaled
        data_proj_re = data_proj.reshape(-1, 1)

        print("shape of evecs.transpose(1, 0)", evecs.transpose(1, 0).shape)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        print("ORI SSM weights shape:", weights.shape)
        return weights

    def theta_to_shape(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Reconstruct shape from theta
        Args:
            weights:      weights of modes
        Returns:
            shape:      reconstructed shape
        """

        if n_modes:
            evecs = self.modes_scaled[:, :n_modes]
        else:
            evecs = self.modes_scaled

        print("ORI weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
        print("ORI evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)
        test_val = np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        print("np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0)):", test_val.shape)
        # self.mean = self.mean.reshape(1, -1)
        print("self.mean shape:", self.mean.shape)
        # print("self.mean val:", self.mean)

        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

    def theta_to_shape_norm(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Reconstruct shape from theta
        Args:
            weights:      weights of modes
        Returns:
            shape:      reconstructed shape
        """

        if n_modes:
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        # print("weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
        # print("evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)
        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        # data_proj = self.mean + torch.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

    def generate_random_samples(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
            print("self.modes_scaled", self.modes_scaled.shape)
        weights = np.random.standard_normal([n_samples, n_modes])
        # print("weights shape", weights.shape)
        samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())
        return np.squeeze(samples)

    def generate_samples_fromSSM(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
            print("self.modes_scaled", self.modes_scaled.shape)

        eigenvals = self.variances[:n_modes]
        mean_vector = np.zeros(n_modes)

        covariance_matrix = np.diag(eigenvals)

        # weights = np.random.standard_normal([n_samples, n_modes])

        weights = np.random.multivariate_normal(mean_vector, covariance_matrix, size=n_samples)

        # print("weights shape", weights.shape)
        samples = self.mean + np.matmul(weights, self.modes_scaled[:,:n_modes].transpose())
        return weights, eigenvals,  np.squeeze(samples)

    def generate_similar_samples(self, n_samples: int = 1, n_modes=None,  ref_shape_weight=None) -> np.ndarray:
        """
        Generate similar samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """

        variation_coeff = 0.1
        if n_modes:
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm


        added_noise = np.random.normal(0, variation_coeff, (ref_shape_weight.shape[0], ref_shape_weight.shape[1], n_samples))
        # print("added_noise shape:", added_noise) #added_noise shape: (6, 1, 10)
        print("added_noise shape:", added_noise.shape) #added_noise shape: (6, 1)
        added_noise = (ref_shape_weight + np.transpose(added_noise, (2, 0, 1))).squeeze(2)
        print("added_noise shape:", added_noise.shape) #added_noise shape: (10, 6)

        print("self.modes_scaled shape:", evecs.shape) #self.modes_scaled shape: (6, 3)

        samples = self.mean + np.matmul(added_noise, evecs.transpose())



        # samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())

        return np.squeeze(samples)
    def get_reconstruction(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)

        # print("weights shape:", weights.shape)
        # print("weights:", weights)

        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)
        return data_proj

    def do_pca(
            self, dataset: np.ndarray, desired_variance: float = 0.9) -> Tuple[Any, int]:
        """Fit principal component analysis to given dataset.

        Parameters
        ----------
        dataset : array_like
            2D array of data to model, where each row on the first axis is one sample
            and each column on the second axis is e.g. shape or appearance for a landmark
        desired_variance : float
            Fraction of total variance to be described by the reduced-dimension model

        Returns
        -------
        pca : sklearn.decomposition._pca.PCA
            Object containing fitted PCA information e.g. components, explained variance
        required_mode_number : int
            Number of principal components needed to produce desired_variance

        Raises
        ------
        Warning
            If mean of each sample in dataset not equal to 0
        Warning
            If standard deviation of each sample in dataset not equal to 1
        """
        # self._check_data_scale(dataset)

        pca = PCA(svd_solver="auto")
        pca.fit(dataset)
        required_mode_number = np.where(
            np.cumsum(pca.explained_variance_ratio_) > desired_variance
        )[0][0]

        return pca, required_mode_number


class HierarchicalSSM:
    def __init__(self, correspondences: np.ndarray) -> None:
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes
        """

        # check dataset
        # check_data_scale(correspondences)

        self.mean = np.mean(correspondences, 0)
        print("see self.mean shape:",self.mean.shape)
        self.max_mode = 128

        data_centered = correspondences - self.mean

        # print("data_centered shape", self.data_centered.shape) # data_centered shape (3014, 3072)
        # # perform clustering
        # labels, n_clusters = self.dbscan_clustering(eps=0.5, min_samples=5)
        # # visualize clustering results
        # self.visualize_clusters(labels)
        # exit(0)

        # data_centered = data_centered.transpose()
        cov_dual = np.matmul(data_centered.transpose(), data_centered) / (
            data_centered.shape[0] - 1
        )
        # print("cov_dual shape", cov_dual.shape)
        evals, evectors = np.linalg.eigh(cov_dual)
        # print("evals shape", evals.shape) # evals shape (3000,)
        # print("evecs shape", evectors.shape) # (3072, 3072)
        # evecs = np.matmul(data_centered, evectors)
        # exit()
        evecs = evectors
        # Normalize the col-vectors
        evecs /= np.sqrt(np.sum(np.square(evecs), 0))

        # Sort
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]

        # new_envals = []
        # for idx, eigenvector in enumerate(evecs.T):
        #     print(np.dot(eigenvector.T, np.dot(cov_dual, eigenvector)))
        #     print("evals[idx]", evals[idx])
        #     exit()

        # Remove the last eigen pair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.variances[self.variances < 0] = 0
        self.modes_norm = evecs[:, :-1]
        # print("modes_norm shape", self.modes_norm.shape)
        # modes_norm shape (3072, 3071)
        # exit()
        # Compute the modes scaled by corresp. std. dev.
        # print("self.variances vals:", self.variances)
        # plot self.variances vals
        # Plotting

        variances = np.array(self.variances)# TODO: !!!!!!!!!!!!!!!!!!use it in diffusion model!!!!!!!!!!!!!!!!!!

        self.eigenvals = variances

        # Calculate the total sum of variances
        total_variance = np.sum(variances)
        # print("total_variance", total_variance)
        # Compute the cumulative sum of variances
        cumulative_variance = np.cumsum(variances)

        # Find the number of components for 99% of total variance
        required_mode_number = np.where(cumulative_variance >= 0.999999 * total_variance)[0][0] + 1

        self.required_components_number = required_mode_number
        print("Required_mode_number for saving 99.9999% info:", required_mode_number)

        variance_64_modes = (np.sum(variances[:128]) / total_variance) * 100
        print(f"Variance explained by first 128 modes: {variance_64_modes:.4f}%")

        variance_64_modes = (np.sum(variances[:64]) / total_variance) * 100
        print(f"Variance explained by first 64 modes: {variance_64_modes:.4f}%")

        variance_32_modes = (np.sum(variances[:32]) / total_variance) * 100
        print(f"Variance explained by first 32 modes: {variance_32_modes:.4f}%")

        variance_16_modes = (np.sum(variances[:16]) / total_variance) * 100
        print(f"Variance explained by first 16 modes: {variance_16_modes:.4f}%")


        # self.modes_norm = self.modes_norm[:, :required_mode_number]
        self.variances_mode_number = self.variances[:required_mode_number]
        # self.variances[required_mode_number-1] = 0
        # # self.variances = self.variances[:required_mode_number]
        # print("len(self.variances)", len(self.variances))
        # print("required_mode_number", required_mode_number)

        # plt.figure(figsize=(10, 6))
        # plt.plot(variances, marker='o', linestyle='-', markersize=5)
        # plt.title('Variances by Principal Component')
        # plt.xlabel('Principal Component Index')
        # plt.ylabel('Variance Explained')
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.xticks(range(len(variances)))  # Setting x-tick labels to match the number of components
        # plt.tight_layout()
        # # Show the plot
        # plt.show()

        # print("self.variances shape:", self.variances.shape)
        # print("val of self.variances:", self.variances[:10])



        # print("self.modes_norm shape:", self.modes_norm.shape)
        # print("self.variances shape:", self.variances.shape)
        # # self.modes_norm:shape: (3000, 2999)
        # # self.variances: shape: (2999,)
        self.modes_scaled = np.multiply(self.modes_norm, np.sqrt(self.variances))

    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on the centered data.
        """
        # Normalize data for better clustering results
        data_norm = self.data_centered / np.linalg.norm(self.data_centered, axis=1)[:, np.newaxis]

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        labels = clustering.fit_predict(data_norm)

        # Calculate number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        print(f"Found {n_clusters} clusters")
        return labels, n_clusters

    def visualize_clusters(self, labels):
        """
        Visualize clustering results using t-SNE for dimensionality reduction.
        """
        # Reduce dimensionality for visualization
        print("Performing t-SNE dimensionality reduction for visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        data_2d = tsne.fit_transform(self.data_centered)

        # Create visualization
        plt.figure(figsize=(12, 8))

        # Plot points colored by cluster
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1],
                              c=labels,
                              cmap='viridis',
                              alpha=0.6)

        # Add colorbar
        plt.colorbar(scatter, label='Cluster')

        # Add labels and title
        plt.title(f'Shape Deformation Clusters (DBSCAN)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')

        # Add text box with clustering info
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        info_text = f'Clusters: {n_clusters}\nNoise points: {n_noise}'
        plt.text(0.02, 0.98, info_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def save_ssm(self, path: str) -> None:
        """
        Save the SSM to a file
        Args: path
        """
        np.savez(
            path,
            mean=self.mean,                  # mean shape
            modes_norm=self.modes_norm[:, :self.max_mode],      # eigenvectors
            # modes_scaled=self.modes_scaled,
            required_components_number = self.required_components_number,
            variances=self.variances[:self.max_mode],        # eigenvalues
            variance_mode_number=self.variances_mode_number,
        )
    def get_variance_num_modes(self, num_modes: int) -> np.ndarray:
        """
        Get the variance of the first num_modes modes
        Args:
            num_modes:  number of modes to consider
        Returns:
            variance:   variance of the first num_modes modes
        """
        return self.variances[:num_modes]

    def get_theta(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        # print("weights shape:", weights.shape)
        return weights

    def get_scaled_theta(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        print("shape of shape:", shape.shape)
        print("shape of mean:", self.mean.shape)
        # exit(0)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            print("if n_modes: self.modes_scaled shape:", self.modes_scaled.shape)
            evecs = self.modes_scaled[:, :n_modes]
        else:
            print("else: self.modes_scaled shape:", self.modes_scaled.shape)
            evecs = self.modes_scaled
        data_proj_re = data_proj.reshape(-1, 1)

        print("shape of evecs.transpose(1, 0)", evecs.transpose(1, 0).shape)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)
        print("ORI SSM weights shape:", weights.shape)
        return weights

    def theta_to_shape(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Reconstruct shape from theta
        Args:
            weights:      weights of modes
        Returns:
            shape:      reconstructed shape
        """

        if n_modes:
            evecs = self.modes_scaled[:, :n_modes]
        else:
            evecs = self.modes_scaled

        print("ORI weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
        print("ORI evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)
        test_val = np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        print("np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0)):", test_val.shape)
        # self.mean = self.mean.reshape(1, -1)
        print("self.mean shape:", self.mean.shape)
        # print("self.mean val:", self.mean)

        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

    def theta_to_shape_norm(self, weights: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Reconstruct shape from theta
        Args:
            weights:      weights of modes
        Returns:
            shape:      reconstructed shape
        """

        if n_modes:
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        # print("weights.transpose(1, 0) shape:", weights.transpose(1, 0).shape)
        # print("evecs.transpose(1, 0) shape:", evecs.transpose(1, 0).shape)
        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        # data_proj = self.mean + torch.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)

        return data_proj

    def generate_random_samples(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
            print("self.modes_scaled", self.modes_scaled.shape)
        weights = np.random.standard_normal([n_samples, n_modes])
        # print("weights shape", weights.shape)
        samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())
        return np.squeeze(samples)

    def generate_samples_fromSSM(self, n_samples: int = 1, n_modes=None) -> np.ndarray:
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
            print("self.modes_scaled", self.modes_scaled.shape)

        eigenvals = self.variances[:n_modes]
        mean_vector = np.zeros(n_modes)

        covariance_matrix = np.diag(eigenvals)

        # weights = np.random.standard_normal([n_samples, n_modes])

        weights = np.random.multivariate_normal(mean_vector, covariance_matrix, size=n_samples)

        # print("weights shape", weights.shape)
        samples = self.mean + np.matmul(weights, self.modes_scaled[:,:n_modes].transpose())
        return weights, eigenvals,  np.squeeze(samples)

    def generate_similar_samples(self, n_samples: int = 1, n_modes=None,  ref_shape_weight=None) -> np.ndarray:
        """
        Generate similar samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples
        """

        variation_coeff = 0.1
        if n_modes:
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm


        added_noise = np.random.normal(0, variation_coeff, (ref_shape_weight.shape[0], ref_shape_weight.shape[1], n_samples))
        # print("added_noise shape:", added_noise) #added_noise shape: (6, 1, 10)
        print("added_noise shape:", added_noise.shape) #added_noise shape: (6, 1)
        added_noise = (ref_shape_weight + np.transpose(added_noise, (2, 0, 1))).squeeze(2)
        print("added_noise shape:", added_noise.shape) #added_noise shape: (10, 6)

        print("self.modes_scaled shape:", evecs.shape) #self.modes_scaled shape: (6, 3)

        samples = self.mean + np.matmul(added_noise, evecs.transpose())



        # samples = self.mean + np.matmul(weights, self.modes_scaled.transpose())

        return np.squeeze(samples)
    def get_reconstruction(self, shape: np.ndarray, n_modes: int = None) -> np.ndarray:
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction
        """
        shape = shape.reshape(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            # if n_modes > self.length:
            #     n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        data_proj_re = data_proj.reshape(-1, 1)
        weights = np.matmul(evecs.transpose(1, 0), data_proj_re)

        # print("weights shape:", weights.shape)
        # print("weights:", weights)

        data_proj = self.mean + np.matmul(weights.transpose(1, 0), evecs.transpose(1, 0))
        data_proj = data_proj.reshape(-1, 3)
        return data_proj

    def do_pca(
            self, dataset: np.ndarray, desired_variance: float = 0.9) -> Tuple[Any, int]:
        """Fit principal component analysis to given dataset.

        Parameters
        ----------
        dataset : array_like
            2D array of data to model, where each row on the first axis is one sample
            and each column on the second axis is e.g. shape or appearance for a landmark
        desired_variance : float
            Fraction of total variance to be described by the reduced-dimension model

        Returns
        -------
        pca : sklearn.decomposition._pca.PCA
            Object containing fitted PCA information e.g. components, explained variance
        required_mode_number : int
            Number of principal components needed to produce desired_variance

        Raises
        ------
        Warning
            If mean of each sample in dataset not equal to 0
        Warning
            If standard deviation of each sample in dataset not equal to 1
        """
        # self._check_data_scale(dataset)

        pca = PCA(svd_solver="auto")
        pca.fit(dataset)
        required_mode_number = np.where(
            np.cumsum(pca.explained_variance_ratio_) > desired_variance
        )[0][0]

        return pca, required_mode_number



class notProbabilisticSSM:
    def __init__(self,correspondences: np.ndarray)-> None:
        """
                Compute the Probabilistic SSM based on eigendecomposition.
                Args:
                    correspondences:    Corresponded shapes
        """

        self.mean = np.mean(correspondences, 0)
        print("see self.mean shape:", self.mean.shape)

        data_centered = correspondences - self.mean

        # data_centered = data_centered.transpose()
        cov_dual = np.matmul(data_centered.transpose(), data_centered) / (
                data_centered.shape[0] - 1
        )
        # print("cov_dual shape", cov_dual.shape)
        evals, evectors = np.linalg.eigh(cov_dual)
        print("evals shape", evals.shape) # evals shape (3072,)
        print("evecs shape", evectors.shape) # (3072, 3072)

        # exit()
        evecs = evectors
        # Normalize the col-vectors
        evecs /= np.sqrt(np.sum(np.square(evecs), 0))

        # Sort
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]


        self.variances = evals[:-1]
        self.variances[self.variances < 0] = 0
        self.modes_norm = evecs[:, :-1]


class PPCA:
    def __init__(self, n_components: int, method: str = 'direct') -> None:
        """
        Initialize PPCA model.
        Args:
            n_components: Number of components to keep
            method: 'direct' or 'em' for optimization method
        """
        self.n_components = n_components
        self.method = method
        self.W = None  # transformation matrix
        self.sigma_squared = None  # noise variance
        self.mean = None  # data mean
        self.components_ = None  # principal components
        self.explained_variance_ = None  # explained variance by each component

    def fit(self, X: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> 'PPCA':
        """
        Fit the PPCA model.
        Args:
            X: Data matrix of shape (n_samples, n_features)
            max_iter: Maximum number of iterations for EM
            tol: Tolerance for convergence
        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        if self.method == 'direct':
            # Direct solution using eigendecomposition
            S = np.dot(X_centered.T, X_centered) / n_samples
            eigenvals, eigenvecs = np.linalg.eigh(S)

            # Sort in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Keep only top components
            eigenvals = eigenvals[:self.n_components]
            eigenvecs = eigenvecs[:, :self.n_components]

            # Compute noise variance (average of discarded eigenvalues)
            self.sigma_squared = np.mean(eigenvals[self.n_components:]) if self.n_components < n_features else 0.0

            # Compute W
            self.W = eigenvecs * np.sqrt(eigenvals - self.sigma_squared)[np.newaxis, :]
            self.components_ = eigenvecs
            self.explained_variance_ = eigenvals

        else:  # EM algorithm
            # Initialize W randomly
            self.W = np.random.randn(n_features, self.n_components)
            self.sigma_squared = 1.0

            old_likelihood = -np.inf

            for i in range(max_iter):
                # E-step: Compute sufficient statistics
                M = np.dot(self.W.T, self.W) + self.sigma_squared * np.eye(self.n_components)
                M_inv = np.linalg.inv(M)
                exp_z = np.dot(np.dot(M_inv, self.W.T), X_centered.T).T
                exp_zz = self.sigma_squared * M_inv + np.dot(exp_z.T, exp_z)

                # M-step: Update parameters
                self.W = np.dot(np.dot(X_centered.T, exp_z), np.linalg.inv(exp_zz))
                self.sigma_squared = np.mean(
                    np.sum(np.square(X_centered), axis=1) -
                    2 * np.sum(np.dot(exp_z, self.W.T) * X_centered, axis=1) +
                    np.trace(np.dot(np.dot(exp_zz, self.W.T), self.W))
                ) / n_features

                # Check convergence
                likelihood = self._compute_likelihood(X_centered)
                if np.abs(likelihood - old_likelihood) < tol:
                    break
                old_likelihood = likelihood

            # Compute components and explained variance
            U, S, _ = np.linalg.svd(self.W, full_matrices=False)
            self.components_ = U
            self.explained_variance_ = np.square(S) + self.sigma_squared

        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to latent space.
        Args:
            X: Data matrix of shape (n_samples, n_features)
        Returns:
            Tuple of (posterior_mean, posterior_covariance)
        """
        X_centered = X - self.mean

        # Compute posterior distribution parameters
        M = np.dot(self.W.T, self.W) + self.sigma_squared * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)

        # Posterior mean
        posterior_mean = np.dot(np.dot(M_inv, self.W.T), X_centered.T).T

        # Posterior covariance (same for all points)
        posterior_covariance = self.sigma_squared * M_inv

        return posterior_mean, posterior_covariance

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Transform from latent space back to data space.
        Args:
            Z: Latent variables of shape (n_samples, n_components)
        Returns:
            Reconstructed data matrix
        """
        return np.dot(Z, self.W.T) + self.mean

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate random samples from the model.
        Args:
            n_samples: Number of samples to generate
        Returns:
            Generated samples
        """
        # Sample from standard normal in latent space
        Z = np.random.normal(0, 1, (n_samples, self.n_components))

        # Transform to data space
        X = np.dot(Z, self.W.T)

        # Add noise
        X += np.random.normal(0, np.sqrt(self.sigma_squared), X.shape)

        # Add mean
        X += self.mean

        return X

    def _compute_likelihood(self, X_centered: np.ndarray) -> float:
        """
        Compute the log likelihood of the data.
        Args:
            X_centered: Centered data matrix
        Returns:
            Log likelihood
        """
        n_samples, n_features = X_centered.shape

        # Compute covariance matrix
        C = np.dot(self.W, self.W.T) + self.sigma_squared * np.eye(n_features)

        # Compute log likelihood
        _, logdet = np.linalg.slogdet(C)
        inv_C = np.linalg.inv(C)

        log_likelihood = -0.5 * (
                n_samples * (n_features * np.log(2 * np.pi) + logdet) +
                np.sum(np.dot(X_centered, inv_C) * X_centered)
        )

        return log_likelihood


import numpy as np
from scipy import linalg
from typing import Tuple, Optional


class ProbabilisticSSM:
    def __init__(self, correspondences: np.ndarray, n_components: Optional[int] = None) -> None:
        """
        Compute the Probabilistic SSM (Statistical Shape Model) using PPCA.
        Args:
            correspondences: Corresponded shapes (n_samples, n_features).
            n_components: Number of principal components to retain. If None, all are used.
        """
        self.mean = np.mean(correspondences, axis=0)
        data_centered = correspondences - self.mean

        if n_components is None:
            n_components = min(data_centered.shape)  # Use min of samples/features, not all features

        self.W, self.sigma2 = self._ppca(data_centered, n_components)
        self.modes_norm = self.W  # Modes are directly the PPCA projection matrix


    def _ppca(self, data: np.ndarray, n_components: int, max_iter: int = 1000,
              tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
        """
        Internal PPCA implementation using EM algorithm.

        Args:
            data: Centered data matrix (n_samples, n_features)
            n_components: Number of components to retain
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple[np.ndarray, float]: (W matrix, noise variance)
        """
        n_samples, n_features = data.shape

        # Initialize parameters
        W = np.random.randn(n_features, n_components) * 0.1  # Smaller initialization
        sigma2 = np.var(data.flatten())  # Better initialization

        for _ in range(max_iter):
            # E-step: Compute expected latent variables
            M = W.T @ W + sigma2 * np.eye(n_components)
            M_inv = linalg.inv(M)

            # M-step: Update parameters
            # Simplified and corrected update equations
            exp_z = data @ W @ M_inv
            exp_zz = sigma2 * M_inv + exp_z.T @ exp_z

            W_new = data.T @ exp_z @ linalg.inv(exp_zz)

            # Corrected sigma2 update
            sigma2_new = np.mean(
                np.sum(np.square(data), axis=1) -
                2 * np.sum(exp_z @ W.T * data, axis=1) +
                np.trace(exp_zz @ W.T @ W)
            ) / n_features

            # Check convergence
            if (np.abs(sigma2_new - sigma2) < tolerance and
                    np.linalg.norm(W_new - W, ord='fro') < tolerance):
                break

            W = W_new
            sigma2 = sigma2_new

        return W, sigma2

    def project(self, data: np.ndarray) -> np.ndarray:
        """
        Projects new data into the learned latent space.

        Args:
            data: Data matrix to project (n_samples, n_features)

        Returns:
            np.ndarray: Latent representations
        """
        data_centered = data - self.mean
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.W.shape[1])
        M_inv = linalg.inv(M)

        # Corrected projection formula
        latents = data_centered @ self.W @ M_inv
        return latents

    def reconstruct(self, latents: np.ndarray) -> np.ndarray:
        """
        Reconstructs data from the latent space.

        Args:
            latents: Latent variables (n_samples, n_components)

        Returns:
            np.ndarray: Reconstructed data
        """
        return latents @ self.W.T + self.mean

    def get_reconstruction_error(self, data: np.ndarray) -> float:
        """
        Compute reconstruction error for given data.

        Args:
            data: Data matrix (n_samples, n_features)

        Returns:
            float: Mean squared reconstruction error
        """
        latents = self.project(data)
        reconstructed = self.reconstruct(latents)
        return np.mean(np.square(data - reconstructed))