import torch


class YCbCrTensor:
    def __init__(self, y, cb, cr):
        self._tensors = {'y': y, 'cb': cb, 'cr': cr}
        self.device = y.device
        self.dtype = torch.float32  # Default dtype for math ops

    @property
    def shape(self):
        """Reflects (B..., 3, H, W) based on Y's shape."""
        y_shape = list(self._tensors['y'].shape)
        y_shape[-3] = 3  # Change the channel dimension (3rd from end) to 3
        return torch.Size(y_shape)

    @property
    def cb_cr_shape(self):
        """Returns actual underlying chroma resolution."""
        return self._tensors['cb'].shape

    # --- Component Accessors (Auto-casting to float) ---
    @property
    def y(self):
        return self._tensors['y'].to(self.dtype)

    @property
    def cb(self):
        return self._tensors['cb'].to(self.dtype)

    @property
    def cr(self):
        return self._tensors['cr'].to(self.dtype)

    def get_components(self, normalize=False):
        """Returns (Y, Cb, Cr) ready for NN input with arbitrary leading dims."""
        y, cb, cr = self.y, self.cb, self.cr
        if normalize:
            return y / 255.0, cb / 255.0, cr / 255.0
        return y, cb, cr

    # --- Conversions ---
    @classmethod
    def from_rgb(cls, rgb_tensor, ratio=(2, 2), as_uint8=True):
        """Converts RGB (B..., 3, H, W) to YCbCr [0, 255]."""
        img = rgb_tensor.float()
        matrix = torch.tensor(
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
            dtype=img.dtype,
            device=img.device,
        )
        ycbcr = torch.einsum("... c h w, o c -> ... o h w", img, matrix)
        ycbcr[..., 1:3, :, :] += 128.0
        y = ycbcr[..., 0:1, :, :]
        cb = ycbcr[..., 1:2, :, :]
        cr = ycbcr[..., 2:3, :, :]

        # Subsample using ellipsis and steps
        cb_sub = cb[..., ::ratio[0], ::ratio[1]]
        cr_sub = cr[..., ::ratio[0], ::ratio[1]]

        if as_uint8:
            y, cb_sub, cr_sub = [torch.clamp(t, 0, 255).round().to(torch.uint8)
                                 for t in (y, cb_sub, cr_sub)]

        return cls(y, cb_sub, cr_sub)

    def to_rgb(self):
        """Converts to RGB (B..., 3, H, W) [0, 255]."""
        ycbcr = self.get_virtual_tensor().float()
        ycbcr[..., 1:3, :, :] -= 128.0
        matrix = torch.tensor(
            [
                [1.0, 0.0, 1.402],
                [1.0, -0.344136, -0.714136],
                [1.0, 1.772, 0.0],
            ],
            dtype=ycbcr.dtype,
            device=ycbcr.device,
        )
        rgb = torch.einsum("... c h w, o c -> ... o h w", ycbcr, matrix)
        return torch.clamp(rgb, 0, 255)

    def crop(self, x, y, w, h):
        """Spatial crop applying to all batch dimensions."""
        if any(v % 2 != 0 for v in (x, y, w, h)):
            raise ValueError("Crop values must be even for 4:2:0 alignment.")

        ry = self._tensors['y'].shape[-2] // self._tensors['cb'].shape[-2]
        rx = self._tensors['y'].shape[-1] // self._tensors['cb'].shape[-1]

        new_y = self._tensors['y'][..., y: y + h, x: x + w]
        new_cb = self._tensors['cb'][..., y // ry: (y + h) // ry, x // rx: (x + w) // rx]
        new_cr = self._tensors['cr'][..., y // ry: (y + h) // ry, x // rx: (x + w) // rx]

        return YCbCrTensor(new_y, new_cb, new_cr)

    # --- Internal Helpers ---
    def get_virtual_tensor(self):
        y, cb, cr = self.y, self.cb, self.cr
        H, W = y.shape[-2:]
        # repeat_interleave works on specific dims from the end
        cb_up = cb.repeat_interleave(H // cb.shape[-2], dim=-2).repeat_interleave(W // cb.shape[-1], dim=-1)
        cr_up = cr.repeat_interleave(H // cr.shape[-2], dim=-2).repeat_interleave(W // cr.shape[-1], dim=-1)
        return torch.cat([y, cb_up, cr_up], dim=-3)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        args = [a.get_virtual_tensor() if isinstance(a, cls) else a for a in args]
        return func(*args, **kwargs)

    def __repr__(self):
        return f"YCbCrTensor(virtual_shape={list(self.shape)}, storage={self._tensors['y'].dtype})"

    def save(self, path):
        """
        Saves the underlying uint8 components to a single file.
        """
        # We save as a dictionary to keep Y, Cb, Cr separate and compressed
        data_to_save = {
            'y': self._tensors['y'],
            'cb': self._tensors['cb'],
            'cr': self._tensors['cr'],
            'ratio': (
                self._tensors['y'].shape[-2] // self._tensors['cb'].shape[-2],
                self._tensors['y'].shape[-1] // self._tensors['cb'].shape[-1]
            )
        }
        torch.save(data_to_save, path)
        print(f"Saved YCbCrTensor components to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        """
        Loads components and reconstructs the YCbCrTensor.
        """
        data = torch.load(path, map_location=device)
        return cls(data['y'], data['cb'], data['cr'])

    def show(self, batch_idx=0):
        """
        Visualizes the Y, Cb, and Cr planes side-by-side.
        Works for any number of leading batch dimensions.
        """
        # Extract components for a single sample in the batch
        # We flatten leading dims to (Samples, C, H, W) for easy indexing
        y_comp = self._tensors['y'].reshape(-1, 1, *self._tensors['y'].shape[-2:])
        cb_comp = self._tensors['cb'].reshape(-1, 1, *self._tensors['cb'].shape[-2:])
        cr_comp = self._tensors['cr'].reshape(-1, 1, *self._tensors['cr'].shape[-2:])

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Helper to plot each plane
        planes = [
            (y_comp[batch_idx, 0], 'Luminance (Y)', 'gray'),
            (cb_comp[batch_idx, 0], 'Chroma Blue (Cb)', 'viridis'),
            (cr_comp[batch_idx, 0], 'Chroma Red (Cr)', 'plasma')
        ]

        for ax, (data, title, cmap) in zip(axes, planes):
            # Move to CPU for plotting
            img_data = data.detach().cpu().numpy()
            im = ax.imshow(img_data, cmap=cmap)
            ax.set_title(f"{title}\n{img_data.shape}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
