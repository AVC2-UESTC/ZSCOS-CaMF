
import torch

def one_hot_encode(labels, num_classes):
        # Create a one-hot encoded tensor
        max_index = torch.max(labels)
        if max_index >= num_classes:
            raise ValueError(f"Label values should be less than {num_classes}")
        # elif max_index < num_classes:
        #     pass
        one_hot = torch.zeros(num_classes, labels.size(0), labels.size(1), dtype=torch.float32)
        one_hot.scatter_(0, labels.unsqueeze(0), 1)
        return one_hot