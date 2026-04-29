#[path = "bench/dataset.rs"]
mod dataset;
#[path = "bench/helpers.rs"]
mod helpers;

use dataset::{load_ground_truth, load_vectors, open_dataset, open_optional_dataset};
use hdf5::File;
use helpers::{compute_ground_truth, duration_average, ms, percentile, recall_at_k};
use hnsw::Hnsw;
use std::{
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

// mnist-784-euclidean.hdf5
const DIM: usize = 128;
const DATASET_PATH: &str = "data/sift-128-euclidean.hdf5";
const TOP_K: usize = 10;
const WARMUP_QUERIES: usize = 100;
const QUERY_LIMIT: Option<usize> = Some(1_000);
const BASE_LIMIT: Option<usize> = None;
const LOAD_INDEX_PREFIX: Option<&str> = Some("data/index/sift-128-euclidean-1MLN");
const SAVE_INDEX_PREFIX: Option<&str> = None;
const SEED: u64 = 42;

const BASE_DATASETS: &[&str] = &["train", "base"];
const QUERY_DATASETS: &[&str] = &["test", "query", "queries"];
const GROUND_TRUTH_DATASETS: &[&str] = &["neighbors", "knns", "groundtruth"];

#[derive(Clone, Copy, Debug)]
struct BenchConfig {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
}

impl BenchConfig {
    fn index_path(self, prefix: &str) -> String {
        format!(
            "{prefix}-dim{DIM}-m{}-m0{}-efc{}-efs{}.bin",
            self.m, self.m0, self.ef_construction, self.ef_search
        )
    }
}

#[allow(non_upper_case_globals)]
const config: [BenchConfig; 4] = [
    BenchConfig {
        m: 16,
        m0: 32,
        ef_construction: 128,
        ef_search: 32,
    },
    BenchConfig {
        m: 16,
        m0: 32,
        ef_construction: 128,
        ef_search: 64,
    },
    BenchConfig {
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: 64,
    },
    BenchConfig {
        m: 32,
        m0: 64,
        ef_construction: 200,
        ef_search: 128,
    },
];

#[derive(Debug)]
struct Metrics {
    build_time: Option<Duration>,
    insert_qps: Option<f64>,
    load_time: Option<Duration>,
    load_path: Option<String>,
    save_time: Option<Duration>,
    save_path: Option<String>,
    query_count: usize,
    qps: f64,
    recall: f64,
    avg_latency: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max_latency: Duration,
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(DATASET_PATH)?;
    let (base_name, base_dataset) = open_dataset(&file, BASE_DATASETS)?;
    let (query_name, query_dataset) = open_dataset(&file, QUERY_DATASETS)?;

    let base = load_vectors::<DIM>(&base_name, &base_dataset, BASE_LIMIT)?;
    let queries = load_vectors::<DIM>(&query_name, &query_dataset, QUERY_LIMIT)?;

    if base.is_empty() {
        return Err("base dataset is empty".into());
    }
    if queries.is_empty() {
        return Err("query dataset is empty".into());
    }

    let k = TOP_K.min(base.len());
    let ground_truth = match open_optional_dataset(&file, GROUND_TRUTH_DATASETS) {
        Some((name, dataset)) if BASE_LIMIT.is_none() => {
            println!("using ground truth dataset '{name}'");
            load_ground_truth(&name, &dataset, queries.len(), k)?
        }
        Some((name, _)) => {
            println!(
                "ignoring ground truth dataset '{name}' because BASE_LIMIT is set; computing exact recall for the truncated base set"
            );
            compute_ground_truth(&base, &queries, k)
        }
        None => {
            println!("no ground truth dataset found; computing exact recall with brute force");
            compute_ground_truth(&base, &queries, k)
        }
    };

    println!("dataset: {DATASET_PATH}");
    println!("base: {base_name} ({} vectors)", base.len());
    println!("queries: {query_name} ({} vectors)", queries.len());
    println!("dimension: {DIM}");
    println!("recall metric: recall@{k}");
    println!(
        "warmup queries: {}",
        WARMUP_QUERIES.min(queries.len().saturating_sub(1))
    );
    println!();

    for params in config {
        let metrics = run_benchmark(&base, &queries, &ground_truth, params, k)?;
        print_metrics(params, k, &metrics);
    }

    Ok(())
}

fn run_benchmark(
    base: &[[f32; DIM]],
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    params: BenchConfig,
    k: usize,
) -> Result<Metrics, Box<dyn Error>> {
    let load_path = LOAD_INDEX_PREFIX.map(|prefix| params.index_path(prefix));
    let (index, build_time, insert_qps, load_time) = match &load_path {
        Some(path) => {
            let load_start = Instant::now();
            let index = Hnsw::<DIM>::load(path)?;
            let load_time = load_start.elapsed();
            (index, None, None, Some(load_time))
        }
        None => {
            let mut index = Hnsw::<DIM>::new_seeded(
                params.m,
                params.m0,
                params.ef_construction,
                params.ef_search,
                SEED,
            );

            let build_start = Instant::now();
            let mut insert_ctx = index.insert_context();
            for &vector in base {
                index.insert_with_context(vector, &mut insert_ctx);
            }
            let build_time = build_start.elapsed();
            let insert_qps = base.len() as f64 / build_time.as_secs_f64();
            (index, Some(build_time), Some(insert_qps), None)
        }
    };

    let save_path = SAVE_INDEX_PREFIX.map(|prefix| params.index_path(prefix));
    let save_time = if let Some(path) = &save_path {
        let save_start = Instant::now();
        index.save(path)?;
        Some(save_start.elapsed())
    } else {
        None
    };

    let warmup = WARMUP_QUERIES.min(queries.len().saturating_sub(1));
    let mut search_ctx = index.search_context();
    for query in queries.iter().take(warmup) {
        let _ = black_box(index.search_with_context(black_box(query), k, &mut search_ctx));
    }

    let measured_queries = queries.len() - warmup;
    let mut total_search_time = Duration::ZERO;
    let mut recall_sum = 0.0;
    let mut latencies = Vec::with_capacity(measured_queries);

    for (query, expected) in queries.iter().zip(ground_truth).skip(warmup) {
        let start = Instant::now();
        let result = index.search_with_context(black_box(query), k, &mut search_ctx);
        let latency = start.elapsed();

        total_search_time += latency;
        recall_sum += recall_at_k(expected, &result);
        latencies.push(latency);
        black_box(result);
    }

    latencies.sort_unstable();
    let avg_latency = duration_average(total_search_time, measured_queries);

    Ok(Metrics {
        build_time,
        insert_qps,
        load_time,
        load_path,
        save_time,
        save_path,
        query_count: measured_queries,
        qps: measured_queries as f64 / total_search_time.as_secs_f64(),
        recall: recall_sum / measured_queries as f64,
        avg_latency,
        p50: percentile(&latencies, 0.50),
        p90: percentile(&latencies, 0.90),
        p99: percentile(&latencies, 0.99),
        max_latency: *latencies.last().unwrap(),
    })
}

fn print_metrics(params: BenchConfig, k: usize, metrics: &Metrics) {
    println!(
        "M={} M0={} ef_construction={} ef_search={}",
        params.m, params.m0, params.ef_construction, params.ef_search
    );
    if let Some(load_time) = metrics.load_time {
        if let Some(path) = &metrics.load_path {
            println!("  load: {:.3}s ({path})", load_time.as_secs_f64());
        } else {
            println!("  load: {:.3}s", load_time.as_secs_f64());
        }
    }
    if let (Some(build_time), Some(insert_qps)) = (metrics.build_time, metrics.insert_qps) {
        println!(
            "  build: {:.3}s ({:.0} inserts/s)",
            build_time.as_secs_f64(),
            insert_qps,
        );
    }
    if let Some(save_time) = metrics.save_time {
        if let Some(path) = &metrics.save_path {
            println!("  save: {:.3}s ({path})", save_time.as_secs_f64());
        } else {
            println!("  save: {:.3}s", save_time.as_secs_f64());
        }
    }
    println!(
        "  search: recall@{k} {:.4}, {:.1} QPS over {} measured queries",
        metrics.recall, metrics.qps, metrics.query_count,
    );
    println!(
        "  latency: avg {:.3} ms, p50 {:.3} ms, p90 {:.3} ms, p99 {:.3} ms, max {:.3} ms",
        ms(metrics.avg_latency),
        ms(metrics.p50),
        ms(metrics.p90),
        ms(metrics.p99),
        ms(metrics.max_latency),
    );
    println!();
}
