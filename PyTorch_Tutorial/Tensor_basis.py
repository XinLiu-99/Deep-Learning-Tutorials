# %%
import torch
import numpy as np

# %%
#Initialize a tensor directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
# %%
#Transfer a tensor or a ndarray to a tensor, avoiding unnecessary data copies
a = np.array([1, 2, 3])
t = torch.as_tensor(a)

# %%
#Initialize a tensor from a Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(np_array)
print(x_np)
# %%
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")



##
# Tensor Attributes


# %%
print(f"Shape of x_ones: {x_ones.shape}")
print(f"Datatype of x_ones: {x_ones.dtype}")
print(f"Device x_ones is stored on: {x_ones.device}")




##
## Tensor Operations

# %%
# Transposed
tensor = torch.tensor([[1, 2], [3, 4]])
transposed = tensor.T
# %%
# Indexing
element = tensor[0, 1]
# %%
# Slicing
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
sliced = tensor[0:2, 1:3]
# %%
# Mathematical Operation
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
sum_tensor = tensor1 + tensor2
# %%
# Linear Algebra(matrix inverse)
tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
inv_tensor = torch.linalg.inv(tensor)
# %%
# random sampling
random_tensor = torch.rand(3, 2)



# %%
# storage
storage = x_data.storage()
print(torch.is_storage(storage))

# %%
# Bridge with Numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# %%
# The change in the tensor reflects in the numpy array
t.add_(5)
