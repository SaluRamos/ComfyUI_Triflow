"""Convert original TorchSparse VAE kernels to spconv KRSC layout."""
def convert_vae_weights(state):
    converted = {}
    for key, value in state.items():
        if key.endswith('.conv.kernel'):
            if value.ndim == 2:
                value = value.T.reshape(value.shape[1], 1, 1, 1, value.shape[0])
            else:
                size = round(value.shape[0] ** (1 / 3))
                # TorchSparse odd kernels iterate z,y,x; spconv indices are x,y,z.
                value = value.reshape(size, size, size, *value.shape[1:]).permute(4, 2, 1, 0, 3)
            key = key.removesuffix('kernel') + 'weight'
            value = value.contiguous()
        converted[key] = value
    return converted
