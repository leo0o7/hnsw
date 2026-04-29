use hdf5::{Dataset, File};
use std::error::Error;

pub(crate) fn open_dataset(
    file: &File,
    candidates: &[&str],
) -> Result<(String, Dataset), Box<dyn Error>> {
    open_optional_dataset(file, candidates).ok_or_else(|| {
        format!(
            "could not find any dataset named one of: {}",
            candidates.join(", ")
        )
        .into()
    })
}

pub(crate) fn open_optional_dataset(file: &File, candidates: &[&str]) -> Option<(String, Dataset)> {
    candidates.iter().find_map(|name| {
        file.dataset(name)
            .ok()
            .map(|dataset| ((*name).to_owned(), dataset))
    })
}

pub(crate) fn load_vectors<const D: usize>(
    dataset_name: &str,
    dataset: &Dataset,
    limit: Option<usize>,
) -> Result<Vec<[f32; D]>, Box<dyn Error>> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(format!("dataset '{dataset_name}' must be 2-D, got shape {shape:?}").into());
    }

    let rows = limit.map_or(shape[0], |value| value.min(shape[0]));
    let dims = shape[1];
    if dims != D {
        return Err(format!(
            "dataset '{dataset_name}' has dimension {dims}, but DIM is set to {D}"
        )
        .into());
    }

    let raw = read_f32_values(dataset, dataset_name)?;
    let expected_len = shape[0] * dims;
    if raw.len() != expected_len {
        return Err(format!("dataset '{dataset_name}' has {expected_len} values from shape {shape:?}, but read {} values", raw.len()).into());
    }

    let mut vectors = Vec::with_capacity(rows);
    for row in raw.chunks_exact(dims).take(rows) {
        vectors.push(row.try_into().unwrap());
    }
    Ok(vectors)
}

pub(crate) fn load_ground_truth(
    dataset_name: &str,
    dataset: &Dataset,
    query_count: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(format!("dataset '{dataset_name}' must be 2-D, got shape {shape:?}").into());
    }
    if shape[1] < k {
        return Err(format!("dataset '{dataset_name}' has only {} ground-truth neighbors per query, but recall@{k} was requested", shape[1]).into());
    }

    let rows = query_count.min(shape[0]);
    if rows != query_count {
        return Err(format!("dataset '{dataset_name}' contains only {rows} rows, but {query_count} queries were loaded").into());
    }

    if let Ok(raw) = dataset.read_raw::<u64>() {
        return build_ground_truth_from_u64(raw, dataset_name, shape[1], rows, k);
    }
    if let Ok(raw) = dataset.read_raw::<u32>() {
        return Ok(build_ground_truth_from_usize(
            raw.into_iter().map(|value| value as usize).collect(),
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }
    if let Ok(raw) = dataset.read_raw::<i64>() {
        let mut ids = Vec::with_capacity(raw.len());
        for value in raw {
            ids.push(usize::try_from(value).map_err(|_| {
                format!("dataset '{dataset_name}' contains a negative ground-truth id: {value}")
            })?);
        }
        return Ok(build_ground_truth_from_usize(
            ids,
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }
    if let Ok(raw) = dataset.read_raw::<i32>() {
        let mut ids = Vec::with_capacity(raw.len());
        for value in raw {
            ids.push(usize::try_from(value).map_err(|_| {
                format!("dataset '{dataset_name}' contains a negative ground-truth id: {value}")
            })?);
        }
        return Ok(build_ground_truth_from_usize(
            ids,
            dataset_name,
            shape[1],
            rows,
            k,
        )?);
    }

    Err(format!(
        "dataset '{dataset_name}' could not be read as u64, u32, i64, or i32 ground-truth ids"
    )
    .into())
}

fn build_ground_truth_from_u64(
    raw: Vec<u64>,
    dataset_name: &str,
    width: usize,
    rows: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut ids = Vec::with_capacity(raw.len());
    for value in raw {
        ids.push(usize::try_from(value).map_err(|_| {
            format!("dataset '{dataset_name}' contains a ground-truth id that does not fit in usize: {value}")
        })?);
    }
    Ok(build_ground_truth_from_usize(
        ids,
        dataset_name,
        width,
        rows,
        k,
    )?)
}

fn build_ground_truth_from_usize(
    raw: Vec<usize>,
    dataset_name: &str,
    width: usize,
    rows: usize,
    k: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let expected_len = rows * width;
    if raw.len() < expected_len {
        return Err(format!(
            "dataset '{dataset_name}' ended early: expected at least {expected_len} ids, got {}",
            raw.len()
        ));
    }

    let mut ground_truth = Vec::with_capacity(rows);
    for row in raw.chunks_exact(width).take(rows) {
        ground_truth.push(row[..k].to_vec());
    }
    Ok(ground_truth)
}

fn read_f32_values(dataset: &Dataset, dataset_name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    if let Ok(values) = dataset.read_raw::<f32>() {
        return Ok(values);
    }
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values.into_iter().map(|value| value as f32).collect());
    }

    Err(format!("dataset '{dataset_name}' could not be read as f32 or f64").into())
}
