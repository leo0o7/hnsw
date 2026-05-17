# hnsw

> [!NOTE]
> This is mostly a learning project: the goal was to build the data structure by hand and get something that can be benchmarked on real ANN datasets

**Hierarchical Navigable Small World** (HNSW) is an approximate nearest-neighbor index.
Instead of comparing a query vector with every vector in the dataset, it builds a layered graph and uses that graph to quickly explore close candidates.

The tradeoff is the usual one for approximate search: searches are much faster than brute force, but recall depends on the parameters used to build and query the index.

## What this includes

- [x] insertion into the graph
- [x] search for the `k` closest vectors
- [x] save/load support
- [x] the usual `M`, `M0`, `ef_construction`, and `ef_search` parameters
- [x] seeded construction for reproducible indexes
- [x] a small benchmark runner for HDF5 ANN datasets
- [x] reusable search/insert contexts to avoid allocating on every operation
- [x] squared L2 distance
- [x] product quantization
- [ ] multiple distance metrics
- [ ] parallel construction

### Non goals

- [ ] deletion
- [ ] filtering
- [ ] metadata storage

The goal is the HNSW data structure itself, not a full vector database.
Also, this is not intended to replace production ANN libraries.

## Benchmark summary

Measured on an Apple M3 Pro with saved indexes loaded from disk.

| Dataset         |   Best high-throughput result |      Best high-recall result |
| --------------- | ----------------------------: | ---------------------------: |
| SIFT-1M, 128d   | 13.4k QPS at 0.8897 recall@10 | 3.2k QPS at 0.9948 recall@10 |
| MNIST-60k, 784d |  8.0k QPS at 0.9866 recall@10 | 2.4k QPS at 0.9998 recall@10 |

## Usage

```rust
use hnsw::Hnsw;

let mut index = Hnsw::<2>::new(
    16,  // M: max links on upper layers
    32,  // M0: max links on layer 0
    128, // ef_construction: candidate list size during insertion
    32,  // ef_search: candidate list size during search
);

index.insert([0.0, 0.0]);
index.insert([3.0, 3.0]);
index.insert([4.0, 4.0]);

let results = index.search(&[1.0, 1.0], 2);

for (id, dist) in results {
    println!("id: {id}, dist²: {dist:.3}");
}
```

For a quick default setup:

```rust
let mut index = Hnsw::<128>::new_default(16);
```

`new_default(M)` uses:

- `M0 = 2 * M`
- `ef_construction = 128`
- `ef_search = 32`

## Benchmarking

There is a benchmark binary that reads a TOML config, loads an HDF5 dataset, then reports recall, QPS, and latency percentiles.

```sh
cargo run --release --bin bench
```

By default it reads:

```sh
bench-config.toml
```

You can also pass a config path:

```sh
cargo run --release --bin bench -- path/to/config.toml
```

The benchmark runner can either build an index from the dataset or load an existing saved index, depending on whether `load_index_prefix` is set in the config.

### Results

These results were measured with saved indexes loaded from disk.

Build/run command:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench -- <config>.toml
```

Environment:

- CPU: Apple M3 Pro
- Memory: 18 GB
- OS: macOS 15.6.1
- Rust: rustc 1.95.0
- Distance: squared L2
- Metric: recall@10
- Warmup queries: 100
- Query cycles: 100
- Measured queries: 90,000

#### SIFT-1M

Config:

- Dataset: `data/sift-128-euclidean.hdf5`
- Base vectors: `train`, 1,000,000 vectors
- Query vectors: `test`, 1,000 vectors
- Dimension: 128
- Ground truth: `neighbors`

|   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |     QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| --: | --: | --------------: | --------: | -----: | ---------: | --------: | ------: | -----: | -----: | -----: | -----: | -----: |
|  16 |  32 |             128 |        32 |  0.528 |     872.46 |    0.8897 | 13444.5 |  0.074 |  0.075 |  0.090 |  0.109 |  0.361 |
|  16 |  32 |             128 |        64 |  0.560 |     872.46 |    0.9564 |  7784.5 |  0.128 |  0.132 |  0.153 |  0.178 |  0.585 |
|  32 |  64 |             200 |        64 |  0.638 |     999.03 |    0.9783 |  5743.7 |  0.174 |  0.180 |  0.217 |  0.252 |  0.849 |
|  32 |  64 |             200 |       128 |  0.609 |     999.03 |    0.9948 |  3226.1 |  0.310 |  0.323 |  0.391 |  0.443 |  0.996 |

#### MNIST-60k

Config:

- Dataset: `data/mnist-784-euclidean.hdf5`
- Base vectors: `train`, 60,000 vectors
- Query vectors: `test`, 1,000 vectors
- Dimension: 784
- Ground truth: `neighbors`

|   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |    QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| --: | --: | --------------: | --------: | -----: | ---------: | --------: | -----: | -----: | -----: | -----: | -----: | -----: |
|  16 |  32 |             128 |        32 |  0.089 |     199.02 |    0.9866 | 8017.7 |  0.125 |  0.126 |  0.156 |  0.185 |  0.497 |
|  16 |  32 |             128 |        64 |  0.089 |     199.02 |    0.9981 | 4849.5 |  0.206 |  0.209 |  0.261 |  0.311 |  0.756 |
|  32 |  64 |             200 |        64 |  0.111 |     202.54 |    0.9986 | 3901.2 |  0.256 |  0.261 |  0.335 |  0.400 |  0.744 |
|  32 |  64 |             200 |       128 |  0.107 |     202.54 |    0.9998 | 2413.7 |  0.414 |  0.421 |  0.552 |  0.657 |  1.425 |

#### Product quantization

These results use frozen PQ indexes over the same saved HNSW graphs.
PQ fit and encode are one-time preprocessing costs; search uses ADC distances over compressed vectors.

##### SIFT-1M PQ

Config:

- Dataset: `data/sift-128-euclidean.hdf5`
- Base vectors: `train`, 1,000,000 vectors
- Query vectors: `test`, 1,000 vectors
- Dimension: 128
- Ground truth: `neighbors`
- PQ centroids per quantizer: 256

| quantizers | pq fit s | pq encode s |   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |     QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| ---------: | -------: | ----------: | --: | --: | --------------: | --------: | -----: | ---------: | --------: | ------: | -----: | -----: | -----: | -----: | -----: |
|         32 |  471.168 |       9.423 |  16 |  32 |             128 |        32 |  1.069 |     414.82 |    0.6781 | 11192.8 |  0.089 |  0.087 |  0.111 |  0.173 |  8.544 |
|         32 |  471.168 |       9.423 |  16 |  32 |             128 |        64 |  0.615 |     414.82 |    0.6998 |  6855.5 |  0.146 |  0.144 |  0.180 |  0.276 |  4.243 |
|         32 |  471.168 |       9.423 |  32 |  64 |             200 |        64 |  0.931 |     541.39 |    0.7044 |  5381.7 |  0.186 |  0.186 |  0.237 |  0.356 |  2.527 |
|         32 |  471.168 |       9.423 |  32 |  64 |             200 |       128 |  0.855 |     541.39 |    0.7077 |  3094.8 |  0.323 |  0.324 |  0.420 |  0.616 |  2.117 |
|         64 |  779.599 |      15.349 |  16 |  32 |             128 |        32 |  1.107 |     445.34 |    0.8069 |  8598.7 |  0.116 |  0.117 |  0.142 |  0.168 |  0.821 |
|         64 |  779.599 |      15.349 |  16 |  32 |             128 |        64 |  0.593 |     445.34 |    0.8474 |  5346.6 |  0.187 |  0.191 |  0.224 |  0.267 |  1.113 |
|         64 |  779.599 |      15.349 |  32 |  64 |             200 |        64 |  0.755 |     571.91 |    0.8603 |  3969.4 |  0.252 |  0.258 |  0.315 |  0.391 |  2.311 |
|         64 |  779.599 |      15.349 |  32 |  64 |             200 |       128 |  1.185 |     571.91 |    0.8657 |  2226.4 |  0.449 |  0.458 |  0.573 |  0.791 | 16.574 |
|        128 |  290.315 |      28.699 |  16 |  32 |             128 |        32 |  1.148 |     506.37 |    0.8748 |  5788.0 |  0.173 |  0.174 |  0.207 |  0.248 |  4.636 |
|        128 |  290.315 |      28.699 |  16 |  32 |             128 |        64 |  0.874 |     506.37 |    0.9333 |  3638.3 |  0.275 |  0.282 |  0.327 |  0.386 |  4.838 |
|        128 |  290.315 |      28.699 |  32 |  64 |             200 |        64 |  1.080 |     632.94 |    0.9512 |  2695.1 |  0.371 |  0.380 |  0.460 |  0.570 |  6.420 |
|        128 |  290.315 |      28.699 |  32 |  64 |             200 |       128 |  0.911 |     632.94 |    0.9646 |  1569.1 |  0.637 |  0.657 |  0.809 |  1.041 | 14.773 |

##### MNIST-60k PQ

Config:

- Dataset: `data/mnist-784-euclidean.hdf5`
- Base vectors: `train`, 60,000 vectors
- Query vectors: `test`, 1,000 vectors
- Dimension: 784
- Ground truth: `neighbors`
- PQ centroids per quantizer: 256

| quantizers | pq fit s | pq encode s |   M |  M0 | ef_construction | ef_search | load s | memory MiB | recall@10 |    QPS | avg ms | p50 ms | p90 ms | p99 ms | max ms |
| ---------: | -------: | ----------: | --: | --: | --------------: | --------: | -----: | ---------: | --------: | -----: | -----: | -----: | -----: | -----: | -----: |
|        196 |  150.985 |       3.412 |  16 |  32 |             128 |        32 |  0.116 |      31.56 |    0.9149 | 7484.3 |  0.134 |  0.133 |  0.151 |  0.182 |  2.430 |
|        196 |  150.985 |       3.412 |  16 |  32 |             128 |        64 |  0.125 |      31.56 |    0.9221 | 5629.8 |  0.178 |  0.177 |  0.208 |  0.261 |  1.966 |
|        196 |  150.985 |       3.412 |  32 |  64 |             200 |        64 |  0.134 |      35.08 |    0.9226 | 4962.1 |  0.202 |  0.201 |  0.243 |  0.320 |  2.982 |
|        196 |  150.985 |       3.412 |  32 |  64 |             200 |       128 |  0.144 |      35.08 |    0.9239 | 3233.8 |  0.309 |  0.306 |  0.397 |  0.527 |  1.147 |

### Config file

The benchmark is configured by `bench-config.toml`.

The fields describe the dataset and how the benchmark should run:

- `dataset_path`: HDF5 file to read
- `dimension`: vector dimension, currently matched in `src/bin/bench.rs`
- `top_k = 10`
- `warmup_queries = 100`
- `query_limit = 1000`
- `query_cycles = 100`
- `load_index_prefix = "data/index/sift-128-euclidean-1MLN"`

There are optional dataset-name fields too.
If they are not set, the runner tries common names like `train`/`base` for vectors, `test`/`query`/`queries` for queries, and `neighbors`/`knns`/`groundtruth` for the expected nearest neighbors.

The `[[configs]]` entries are the HNSW parameter sets to run.
Each one produces or loads a separate index using the same filename pattern:

```toml
[[configs]]
m = 16
m0 = 32
ef_construction = 128
ef_search = 32
```

If `load_index_prefix` is set, the benchmark loads matching index files from disk.
If `save_index_prefix` is set instead, it builds the index from the base dataset and writes it to disk after construction.

## How it works

Each inserted vector becomes a node in a graph.
Most nodes live only on layer 0, while a few are randomly promoted to higher layers.
The higher layers are sparse and act like long-range shortcuts.

Search starts at the current entry point on the top layer, greedily moves closer to the query, and then repeats this while descending layer by layer.
On layer 0, the search becomes wider. It keeps a frontier of candidates to visit and a bounded set of the best candidates found so far.
Once the closest item in the frontier is already worse than the worst item in the result set, the search can stop.

Insertion uses the same idea. It searches the existing graph to find candidate neighbors for the new node, prunes that candidate set, links the new node, and adds backlinks from the selected neighbors.
If an existing node gets too many links, its neighbor list is pruned again.

## Some implementation choices

### Const generic dimensions

Vectors are stored as `[f32; D]` instead of `Vec<f32>`.
That makes dimensions a compile-time part of the index type.

Every vector in one index has the same size, so representing that in the type avoids checking the dimension on every insert/search.
It also keeps vector storage flat and makes the distance function work on fixed-size arrays instead of slices whose length has to be trusted at runtime.

### Reusable contexts

There are `insert_context` and `search_context` helpers.
They hold the heaps and scratch buffers used by insertion/search, so repeated calls do not need to keep allocating the same temporary structures.

The simple methods still exist:

```rust
index.insert(vector);
let results = index.search(&query, 10);
```

But benchmark code uses the context versions:

```rust
let mut search_ctx = index.search_context();
let results = index.search_with_context(&query, 10, &mut search_ctx);
```

### Epoch markers instead of clearing visited arrays

Search and neighbor selection need to know which nodes have already been seen.
Instead of clearing a visited array for every search, each node stores a small epoch marker.

During a query, insertion search, or neighbor selection, only a small part of the graph is usually touched.
Clearing a full `visited` array every time would make each operation pay a cost proportional to the whole index size, even when the graph walk itself only visited a small number of nodes.

With epoch markers, each operation increments the current epoch and marks the nodes it touches with that value.
Checking whether a node was already seen is then just comparing its stored epoch with the current one.

### Save/load

Indexes can be saved to disk and loaded again:

```rust
index.save("index.bin")?;
let loaded = Hnsw::<128>::load("index.bin")?;
```

The random seed is stored too.
When an index is loaded, the RNG is advanced by the number of already-inserted vectors so that continuing insertion behaves the same as it would have before saving.

## Tests

```sh
cargo test
```

The tests cover small exact examples, empty indexes, duplicate vectors, save/load roundtrips, max connection limits, duplicate neighbor checks, and a random recall check against brute force.

## References

This is based on the HNSW paper:

- Yu. A. Malkov and D. A. Yashunin, _Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs_, 2018.

I also looked at Redis' HNSW/vector set implementation while working through some of the practical details around graph construction and neighbor pruning.
