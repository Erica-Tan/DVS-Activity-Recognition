import torch

class Loader:
    def __init__(self, dataset, flags, device):
        self.device = "cuda:0"
        split_indices = list(range(len(dataset)))
        # print(split_indices)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        # print(sampler)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, sampler=sampler,
                                             num_workers=flags.num_workers, pin_memory=flags.pin_memory)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)
        