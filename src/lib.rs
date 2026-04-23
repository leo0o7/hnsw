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

impl<const D: usize> Hnsw<D> {
    #[allow(non_snake_case)]
    pub fn new(M: usize, M0: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            M,
            M0,
            ef_construction,
            ef_search,
            entry_point: 0,
            data: Vec::new(),
            nodes: Vec::new(),
        }
    }
    #[allow(non_snake_case)]
    pub fn new_default(M: usize) -> Self {
        Self::new(M, 2 * M, 128, 32)
    }
}

