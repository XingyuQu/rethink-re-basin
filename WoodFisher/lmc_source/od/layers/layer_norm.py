import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch
# import flax
# import jax.numpy as jnp
# import jax
# import numpy as np


class LayerNorm(nn.LayerNorm):

    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 3, 1)
        num_channels = input.size(-1)
        out = F.layer_norm(input, [num_channels], self.weight[:num_channels],
                           self.bias[:num_channels], self.eps)
        # numerator = input - input.mean(-1, keepdim=True)
        # denominator = torch.sqrt(input.var(-1, keepdim=True, correction=0) + self.eps)
        # out = (numerator / denominator) * self.weight + self.bias
        return out.permute(0, 3, 1, 2)

# use flax to implement layer norm
# class LayerNorm(nn.LayerNorm):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         key = jax.random.PRNGKey(0)
#         tmp = list(self.normalized_shape)
#         noramalized_shape_list = [tmp[1], tmp[2], tmp[0]]
#         input_shape = [1] + noramalized_shape_list

#         x = jax.random.normal(key, input_shape)
#         self.layer_norm_flax = flax.linen.LayerNorm()
#         self.layer_norm_flax.init(key, x)

#     def forward(self, input: Tensor) -> Tensor:
#         # only support image input
#         input = input.permute(0, 2, 3, 1)
#         input = jnp.array(input.detach().cpu().numpy())
#         params = {'scale': jnp.array(self.weight.data.detach().cpu().numpy()).mean((1, 2)),
#                   'bias': jnp.array(self.bias.data.detach().cpu().numpy().mean((1, 2)))}
#         out = self.layer_norm_flax.apply({'params': params}, input)
#         out = torch.tensor(np.array(out)).permute(0, 3, 1, 2)
#         return out