#![allow(unused)]
use crate::node::Node;
use std::cell::Cell;

mod dist;
mod link;
mod node;

#[allow(non_snake_case)]
struct Hnsw<const D: usize> {
    M: usize,
    M0: usize,
    ef_construction: usize,
    ef_search: usize,
    entry_point: usize,
    data: Vec<[f32; D]>,
    nodes: Vec<Node>,
}

impl<const D: usize> Hnsw<D> {}
