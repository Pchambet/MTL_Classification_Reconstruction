from torchvision import datasets, transforms
from torch.utils.data import Dataset


def make_transform(size=(64, 64)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class RelabeledDataset(Dataset):
    """Wrapper around torchvision ImageFolder that relabels classes and optionally filters by class names.

    This implementation is importable from a module so multiprocessing DataLoader workers can pickle it.
    """
    def __init__(self, base: datasets.ImageFolder, mapping: dict, keep_names=None):
        self.base = base
        if keep_names is None:
            self.indices = list(range(len(base.samples)))
        else:
            keep = set(keep_names)
            self.indices = [i for i, (_, y) in enumerate(base.samples) if base.classes[y] in keep]
        self.mapping = mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        return x, self.mapping[y]


def build_binary_dataset(root, keep=("Forest", "Residential"), size=(64, 64)):
    base = datasets.ImageFolder(root, transform=make_transform(size))
    mapping = {base.class_to_idx[keep[0]]: 0, base.class_to_idx[keep[1]]: 1}
    ds = RelabeledDataset(base, mapping, keep_names=keep)
    return ds, base, mapping


def relabel_imagefolder(root, mapping, size=(64, 64)):
    base = datasets.ImageFolder(root, transform=make_transform(size))
    ds = RelabeledDataset(base, mapping, keep_names=None)
    return ds, base


def indices_by_class(base, name):
    c = base.class_to_idx[name]
    return [i for i, (_, y) in enumerate(base.samples) if y == c]
