#[path = "bench/dataset.rs"]
mod dataset;
#[path = "bench/helpers.rs"]
mod helpers;

use dataset::{load_ground_truth, load_vectors, open_dataset, open_optional_dataset};
use hdf5::File;
use helpers::{compute_ground_truth, duration_average, ms, percentile, recall_at_k};
use hnsw::Hnsw;
use serde::Deserialize;
use std::{
    env,
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

const DEFAULT_CONFIG_PATH: &str = "bench-config.toml";
const DEFAULT_BASE_DATASETS: &[&str] = &["train", "base"];
const DEFAULT_QUERY_DATASETS: &[&str] = &["test", "query", "queries"];
const DEFAULT_GROUND_TRUTH_DATASETS: &[&str] = &["neighbors", "knns", "groundtruth"];

#[derive(Debug, Deserialize)]
struct BenchFile {
    dataset_path: String,
    dimension: usize,
    top_k: usize,
    warmup_queries: usize,
    query_cycles: Option<usize>,
    query_limit: Option<usize>,
    base_limit: Option<usize>,
    load_index_prefix: Option<String>,
    save_index_prefix: Option<String>,
    seed: Option<u64>,
    base_datasets: Option<Vec<String>>,
    query_datasets: Option<Vec<String>>,
    ground_truth_datasets: Option<Vec<String>>,
    configs: Vec<BenchConfig>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct BenchConfig {
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
}

impl BenchConfig {
    fn index_path(self, prefix: &str, dimension: usize) -> String {
        format!(
            "{prefix}-dim{dimension}-m{}-m0{}-efc{}-efs{}.bin",
            self.m, self.m0, self.ef_construction, self.ef_search
        )
    }
}

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
    let config_path = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_owned());
    let config: BenchFile = toml::from_str(&std::fs::read_to_string(&config_path)?)?;

    match config.dimension {
        128 => run::<128>(&config),
        784 => run::<784>(&config),
        other => Err(
            format!("unsupported dimension {other}; add a match arm in src/bin/bench.rs").into(),
        ),
    }
}

fn run<const DIM: usize>(config: &BenchFile) -> Result<(), Box<dyn Error>> {
    let file = File::open(&config.dataset_path)?;
    let base_dataset_names = dataset_names(config.base_datasets.as_deref(), DEFAULT_BASE_DATASETS);
    let query_dataset_names =
        dataset_names(config.query_datasets.as_deref(), DEFAULT_QUERY_DATASETS);
    let ground_truth_dataset_names = dataset_names(
        config.ground_truth_datasets.as_deref(),
        DEFAULT_GROUND_TRUTH_DATASETS,
    );
    let (base_name, base_dataset) = open_dataset(&file, &base_dataset_names)?;
    let (query_name, query_dataset) = open_dataset(&file, &query_dataset_names)?;

    let base = load_vectors::<DIM>(&base_name, &base_dataset, config.base_limit)?;
    let queries = load_vectors::<DIM>(&query_name, &query_dataset, config.query_limit)?;

    if base.is_empty() {
        return Err("base dataset is empty".into());
    }
    if queries.is_empty() {
        return Err("query dataset is empty".into());
    }

    let k = config.top_k.min(base.len());
    let ground_truth = match open_optional_dataset(&file, &ground_truth_dataset_names) {
        Some((name, dataset)) if config.base_limit.is_none() => {
            println!("using ground truth dataset '{name}'");
            load_ground_truth(&name, &dataset, queries.len(), k)?
        }
        Some((name, _)) => {
            println!(
                "ignoring ground truth dataset '{name}' because base_limit is set; computing exact recall for the truncated base set"
            );
            compute_ground_truth(&base, &queries, k)
        }
        None => {
            println!("no ground truth dataset found; computing exact recall with brute force");
            compute_ground_truth(&base, &queries, k)
        }
    };

    println!("dataset: {}", config.dataset_path);
    println!("base: {base_name} ({} vectors)", base.len());
    println!("queries: {query_name} ({} vectors)", queries.len());
    println!("dimension: {DIM}");
    println!("recall metric: recall@{k}");
    println!(
        "warmup queries: {}",
        config.warmup_queries.min(queries.len().saturating_sub(1))
    );
    println!("query cycles: {}", config.query_cycles());
    println!();

    for params in config.configs.iter().copied() {
        let metrics = run_benchmark(&base, &queries, &ground_truth, params, k, config)?;
        print_metrics(params, k, &metrics);
    }

    Ok(())
}

fn dataset_names<'a>(configured: Option<&'a [String]>, defaults: &[&'a str]) -> Vec<&'a str> {
    configured
        .map(|names| names.iter().map(String::as_str).collect())
        .unwrap_or_else(|| defaults.to_vec())
}

fn run_benchmark<const DIM: usize>(
    base: &[[f32; DIM]],
    queries: &[[f32; DIM]],
    ground_truth: &[Vec<usize>],
    params: BenchConfig,
    k: usize,
    config: &BenchFile,
) -> Result<Metrics, Box<dyn Error>> {
    let load_path = config
        .load_index_prefix
        .as_deref()
        .map(|prefix| params.index_path(prefix, DIM));
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
                config.seed.unwrap_or(42),
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

    let save_path = config
        .save_index_prefix
        .as_deref()
        .map(|prefix| params.index_path(prefix, DIM));
    let save_time = if let Some(path) = &save_path {
        let save_start = Instant::now();
        index.save(path)?;
        Some(save_start.elapsed())
    } else {
        None
    };

    let warmup = config.warmup_queries.min(queries.len().saturating_sub(1));
    let mut search_ctx = index.search_context();
    for query in queries.iter().take(warmup) {
        let _ = black_box(index.search_with_context(black_box(query), k, &mut search_ctx));
    }

    let measured_queries = queries.len() - warmup;
    let query_cycles = config.query_cycles();
    let total_measured_queries = measured_queries * query_cycles;
    let mut total_search_time = Duration::ZERO;
    let mut recall_sum = 0.0;
    let mut latencies = Vec::with_capacity(total_measured_queries);

    for _ in 0..query_cycles {
        for (query, expected) in queries.iter().zip(ground_truth).skip(warmup) {
            let start = Instant::now();
            let result = index.search_with_context(black_box(query), k, &mut search_ctx);
            let latency = start.elapsed();

            total_search_time += latency;
            recall_sum += recall_at_k(expected, &result);
            latencies.push(latency);
            black_box(result);
        }
    }

    latencies.sort_unstable();
    let avg_latency = duration_average(total_search_time, total_measured_queries);

    Ok(Metrics {
        build_time,
        insert_qps,
        load_time,
        load_path,
        save_time,
        save_path,
        query_count: total_measured_queries,
        qps: total_measured_queries as f64 / total_search_time.as_secs_f64(),
        recall: recall_sum / total_measured_queries as f64,
        avg_latency,
        p50: percentile(&latencies, 0.50),
        p90: percentile(&latencies, 0.90),
        p99: percentile(&latencies, 0.99),
        max_latency: *latencies.last().unwrap(),
    })
}

impl BenchFile {
    fn query_cycles(&self) -> usize {
        self.query_cycles.unwrap_or(1).max(1)
    }
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
