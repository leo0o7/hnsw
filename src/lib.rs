#![allow(unused)]
use crate::{dist::l2_squared, link::Link, node::Node};
use std::{cell::Cell, cmp::Reverse, collections::BinaryHeap};

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
    max_layer: usize,
    epoch: Cell<usize>,
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
            max_layer: 0,
            epoch: Cell::new(0),
        }
    }
    #[allow(non_snake_case)]
    pub fn new_default(M: usize) -> Self {
        Self::new(M, 2 * M, 128, 32)
    }

    pub fn search(&self, q: &[f32; D]) {
        // TODO: assertions
        // ...
        todo!()
    }

    fn search_layer(&self, q: &[f32; D], ep: usize, lyr: usize, ef: usize) -> Vec<Link> {
        self.epoch.set(self.epoch.get() + 1);
        // TODO: assertions
        // ...
        let mut frontier = BinaryHeap::<Reverse<Link>>::new();
        let mut best = BinaryHeap::<Link>::with_capacity(ef);

        let ep_link = Link {
            node_index: ep,
            distance: l2_squared(q, &self.data[ep]),
        };
        frontier.push(Reverse(ep_link));
        best.push(ep_link);
        self.nodes[ep].epoch.set(self.epoch.get());

        while let Some(Reverse(candidate)) = frontier.pop() {
            let furthest_dist = best.peek().map_or(f32::INFINITY, |l| l.distance);
            if candidate.distance > furthest_dist {
                break;
            }
            for neigh in self.nodes[candidate.node_index].layers[lyr].iter() {
                if self.nodes[neigh.node_index].epoch == self.epoch {
                    continue;
                }
                self.nodes[neigh.node_index].epoch.set(self.epoch.get());
                let dist = l2_squared(q, &self.data[neigh.node_index]);
                if best.len() == ef && best.peek().is_some_and(|furthest| furthest.distance > dist)
                {
                    best.pop();
                }
                if best.len() < ef {
                    let link = Link {
                        node_index: neigh.node_index,
                        distance: dist,
                    };
                    best.push(link);
                    frontier.push(Reverse(link));
                }
            }
        }

        self.avoid_epoch_overflow();
        best.into_sorted_vec()
    }

    #[inline(always)]
    /// very unlikely to ever be required
    /// this can happen only after 2^64 - 1 epochs
    fn avoid_epoch_overflow(&self) {
        if self.epoch.get() == usize::MAX {
            self.epoch.set(0);
            for node in self.nodes.iter() {
                node.epoch.set(0);
            }
        }
    }
}

