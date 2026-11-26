# Dataset files

MoReVec is not redistributed in this repository. Download it with `python benchmark.py download-morevec`,
then inspect the actual HDF5 paths with `python benchmark.py inspect-data --path <download-directory>`.

Configure `source.data_file`, `source.vectors_key`, and `source.attributes.avgrating` in
`configs/pilot.yaml` using the inspected paths. The HDF5 files are read-only; the benchmark writes
prepared queries, ground truth and CSV output under this directory.
