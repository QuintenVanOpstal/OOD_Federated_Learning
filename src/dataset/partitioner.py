from torch.utils.data import Subset
import numpy as np


class Partition:
    def partitionByClass(self, data):
        lstParts = []
        classes = np.unique(data.targets)
        for y in classes:
            idx = np.arange(len(data))
            X = Subset(data, idx[data.targets == y])
            lstParts.append(X)

        return lstParts

    def iidParts(self, data, numParts):
        N = len(data)  # data.data.shape[0]
        idxs = np.random.permutation(N)
        partIdxs = np.array_split(idxs, numParts)
        lstParts = [Subset(data, partIdx) for partIdx in partIdxs]
        return lstParts

    def hetero_dir_partition(self, targets, labels, num_clients, num_classes, dir_alpha, min_require_size=None):
        """

        Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
        `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
        and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

        This method simulates heterogeneous partition for which number of data points and class
        proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
        :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
        samples of class :math:`k` to local client :math:`j`.

        Sample number for each client is decided in this function.

        Args:
            targets (TensorDataset): Sample targets. Unshuffled preferred.
            num_clients (int): Number of clients for partition.
            num_classes (int): Number of classes in samples.
            dir_alpha (float): Parameter alpha for Dirichlet distribution.
            min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

        Returns:
            dict: ``{ client_id: indices}``.
        """
        if min_require_size is None:
            min_require_size = num_classes

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        num_samples = labels.shape[0]

        print(('number of samples {}').format(num_samples))

        min_size = 0
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(labels - 1 == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(dir_alpha, num_clients))
                # print(proportions)
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                     zip(proportions, idx_batch)])
                # print(proportions)
                proportions = proportions / proportions.sum()

                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # print(np.split(idx_k, proportions))
                # print(idx_batch)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        client_dict = dict()
        for cid in range(num_clients):
            np.random.shuffle(idx_batch[cid])
            client_dict[cid] = Subset(targets, idx_batch[cid])

        return client_dict