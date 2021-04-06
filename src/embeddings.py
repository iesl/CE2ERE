import torch
from torch.nn import Module
from torch.nn.functional import softplus


class BoxEmbedding:
    def __init__(self, volume_temp: float = 1.0, threshold: float = 20):
        super().__init__()
        self.volume_temp = volume_temp
        self.threshold = threshold

    def get_box_embeddings(self, vector: torch.Tensor):
        """
        create box embedding from vector
        shape: [batch_size, box_min/box_max, hidden_dim]
        box_min: bottom-left corner (=center-offset), box_max: top-right corner (= center+offset)
        center: (box_max+box_min)/2
        offset: (box_max-box_min)/2
        """
        len_dim = vector.shape[-1]
        dim = -1

        if len_dim % 2 != 0:
            raise ValueError(f"The last dimension of vector should be even but is {vector.shape[-1]}")

        split_point = int(len_dim/2)
        # box_min: [batch_size, vector_dim/2]; [64, 256]
        box_min = vector.index_select(dim, torch.tensor(list(range(split_point)), dtype=torch.int64, device=vector.device))
        delta = vector.index_select(dim, torch.tensor(list(range(split_point, len_dim)), dtype=torch.int64, device=vector.device))
        box_max = box_min + softplus(delta, beta=1 / self.volume_temp, threshold=self.threshold)

        assert box_min.shape == box_max.shape
        assert (box_max >= box_min).all()

        return torch.stack((box_min, box_max), dim=-2) # [batch_size, 2, vector_dim/2]; [64, 2, 256]