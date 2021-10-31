from torch.utils.data import sampler
from torch.utils.data.sampler import Sampler
import numpy as np
from random import shuffle

""" 
https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c

PyTorch has pack_padded_sequence this doesn’t work with dense layers. For sequence data with high variance in its length 
the best way to minimize padding and masking within a batch is by feeding in data that is already grouped by sequence length 
(while still shuffling it somewhat). Here is my current solution in numpy. 
I will need to convert every function over to torch to allow it to run on the GPU and am sure there are many other 
ways to optimize it further. Hope this helps others and that maybe it can become a new PyTorch Batch Sampler someday.
General approach to how it works:
Decide what your bucket boundaries for the data are.
1. Iterate through your data (provided in an array) and for each element its index and length is recorded
2. Given these indices and lengths, each index is assigned to a bucket ID (I took this whole function from the tensorflow batch_by_sequence_length linked to above)
3. Shuffle the data in these buckets
4. Split the data in each bucket into approximately the batch size (may be slightly larger)
5. Shuffle all of the batches made
6. yield a batch (which contains index references to your data)
Some code and inspiration taken from: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length
"""


class BySequenceLengthSampler(Sampler):
    def __init__(self, data_source, bucket_boundaries, batch_size, len_fn=len):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append((i, len_fn(p)))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.data_source = data_source

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            if data_buckets[k].shape[0] < self.batch_size:
                iter_list.append(data_buckets[k])
            else:
                iter_list += np.array_split(
                    data_buckets[k], int(data_buckets[k].shape[0] / self.batch_size)
                )
        shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return len(self.data_source)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
            np.less_equal(buckets_min, seq_length), np.less(seq_length, buckets_max)
        )
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id


if __name__ == "__main__":

    # below is a simple dataset to demonstration

    from torch.utils.data import Dataset, DataLoader
    import torch

    class TestDataSet(Dataset):
        def __init__(self):
            self.lengths = torch.randint(low=0, high=100, size=(1000,))
            self.data = [torch.LongTensor([0] * l) for l in self.lengths]
            self.labels = torch.randint(low=0, high=2, size=(1000,))

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

    def collate_batch(batch):

        # break a list of tuple to a tuple of lists
        token_lists, _ = list(zip(*batch))
        length_list = torch.LongTensor([len(token_list) for token_list in token_lists])

        token_list_pad = torch.nn.utils.rnn.pad_sequence(
            token_lists, batch_first=True, padding_value=1
        )

        return token_list_pad, length_list

    dataset = TestDataSet()
    sequence_sampler = BySequenceLengthSampler(
        dataset, [25, 50, 75, 100], batch_size=20, len_fn=lambda x: len(x[0])
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=sequence_sampler,
        drop_last=False,
        collate_fn=collate_batch,
    )

    for batch in data_loader:
        print(batch[0].shape, batch[1])

