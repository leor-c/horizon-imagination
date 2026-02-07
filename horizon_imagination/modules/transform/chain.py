from horizon_imagination.modules.transform.base import BaseTransform


class ChainTransform(BaseTransform):

    def __init__(self, transforms: list[BaseTransform]):
        super().__init__()
        self.transforms = transforms

    def transform(self, x, *args, **kwargs):
        for t in self.transforms:
            x = t.transform(x, *args, **kwargs)
        return x
    
    def inverse(self, z, *args, **kwargs):
        for t in self.transforms:
            z = t.inverse(z, *args, **kwargs)
        return z
