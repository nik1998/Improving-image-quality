import numpy as np
import tensorflow as tf
from time import time


def batch_pairwise_distances(u, v):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = tf.reduce_sum(tf.square(u), 1)
    norm_v = tf.reduce_sum(tf.square(v), 1)

    # norm_u as a column and norm_v as a row vectors.
    norm_u = tf.reshape(norm_u, [-1, 1])
    norm_v = tf.reshape(norm_v, [1, -1])

    # Pairwise squared Euclidean distances.
    d = tf.maximum(norm_u - 2 * tf.matmul(u, v, False, True) + norm_v, 0.0)
    return d


class ManifoldEstimator:
    """Estimates the manifold of given feature vectors."""

    def __init__(self, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=None, clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.

            Args:
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        if nhood_sizes is None:
            nhood_sizes = [3]
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1 - begin1, begin2:end2] = batch_pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1 - begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images, ], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images, ], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1 - begin1, begin2:end2] = batch_pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1 - begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps),
                                                    axis=1)
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


# ----------------------------------------------------------------------------
# https://arxiv.org/pdf/1904.06991.pdf
def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=None,
                                  row_batch_size=10000, col_batch_size=50000):
    """Calculates k-NN precision and recall for two sets of feature vectors.

        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    if nhood_sizes is None:
        nhood_sizes = [3]
    state = dict()
    ref_features = np.reshape(ref_features, (ref_features.shape[0], -1))
    eval_features = np.reshape(eval_features, (eval_features.shape[0], -1))
    num_images = ref_features.shape[0]

    # Initialize DistanceBlock and ManifoldEstimators.
    ref_manifold = ManifoldEstimator(ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    print(state)
    return state
