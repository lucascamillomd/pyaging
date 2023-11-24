import torch


def scale(data_tensor, scaler):
    """
    Scales the input tensor using the provided scaler.
    """
    # Store the device (CPU/GPU) of the input tensor to restore it later
    device = data_tensor.device

    # Convert the tensor to a NumPy array. This requires moving the tensor to the CPU, as NumPy does not support GPU tensors
    data_np = data_tensor.cpu().numpy()

    # Apply the scaling transformation to the NumPy array
    data_scaled_np = scaler.transform(data_np)

    # Convert the scaled data back to a PyTorch tensor, ensuring it's on the same device and of the same type as the original tensor
    data_scaled = torch.tensor(data_scaled_np, dtype=data_tensor.dtype, device=device)

    return data_scaled


def binarize(data_tensor):
    """
    Binarizes a torch tensor based on the median of each row, excluding zeros.
    """
    # Store the device (CPU/GPU) of the input tensor to restore it later
    device = data_tensor.device

    # Create a mask for non-zero elements
    non_zero_mask = data_tensor != 0

    # Apply mask and calculate median for each row
    medians = torch.zeros(data_tensor.shape[0])
    for i, row in enumerate(data_tensor):
        non_zero_elements = row[non_zero_mask[i]]
        medians[i] = non_zero_elements.median()

    # Expand the medians tensor to match the shape of data_tensor for element-wise comparison
    medians_expanded = medians.unsqueeze(1).expand_as(data_tensor)

    # Binarize the tensor: 1 if the element is greater than the median, 0 otherwise
    binarized_tensor = (data_tensor > medians_expanded).float()

    binarized_tensor.to(device)

    return binarized_tensor
