# Dataset configurations

A dataset is expected to be a pytorch lightning `LightningDataModule`, where
the constructor at least accepts the following parameters which are used in the
experiment config tests.

 - `batch_size: int`
 - `num_workers: int`
 - `shuffle_buffer_size: int`

Check out the [datasets section][ocl.datasets] of the api for some examples on
how a dataset can be implemented.
