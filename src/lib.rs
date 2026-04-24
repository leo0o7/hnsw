#![allow(unused)]

use rand::prelude::*;

use crate::{dist::l2_squared, link::Link, node::Node};
use std::{
    cell::Cell,
    cmp::{Reverse, max},
    collections::BinaryHeap,
};

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
    ml: f64,
    rng: StdRng,
}

impl<const D: usize> Hnsw<D> {
    #[allow(non_snake_case)]
    pub fn new(M: usize, M0: usize, ef_construction: usize, ef_search: usize) -> Self {
        let seed = rand::rng().next_u64();
        Self::new_seeded(M, M0, ef_construction, ef_search, seed)
    }

    #[allow(non_snake_case)]
    pub fn new_seeded(
        M: usize,
        M0: usize,
        ef_construction: usize,
        ef_search: usize,
        seed: u64,
    ) -> Self {
        assert!(M > 1, "M must be > 1");
        assert!(M0 > 0, "M0 must be > 0");
        assert!(ef_construction > 0, "ef_construction must be > 0");
        assert!(ef_search > 0, "ef_search must be > 0");

        let ml = 1.0 / (M as f64).ln();
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
            ml,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    #[allow(non_snake_case)]
    pub fn new_default(M: usize) -> Self {
        Self::new(M, 2 * M, 128, 32)
    }

    pub fn insert(&mut self, vec: [f32; D]) {
        let insert_idx = self.data.len();
        let insert_lyr = self.random_layer();
        self.data.push(vec);
        self.nodes.push(Node {
            layers: Vec::with_capacity(insert_lyr + 1),
            epoch: Cell::new(0),
        });
        for lyr in 0..=insert_lyr {
            let max_connections = self.max_connections(lyr);
            self.nodes[insert_idx]
                .layers
                .push(Vec::with_capacity(max_connections));
        }

        if insert_idx == 0 {
            self.entry_point = 0;
            self.max_layer = insert_lyr;
            return;
        }

        let mut ep = self.entry_point;
        for lyr in ((insert_lyr + 1)..=self.max_layer).rev() {
            ep = self
                .search_layer(&vec, ep, lyr, 1)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (insert)")
                })
                .node_index;
        }

        for lyr in (0..=insert_lyr.min(self.max_layer)).rev() {
            let candidates = self.search_layer(&vec, ep, lyr, self.ef_construction);
            let neighs = self.select_neighbors(&vec, lyr, candidates, true, true);
            self.nodes[insert_idx].layers[lyr] = neighs;

            // can't use .iter() here because it would keep an immutable borrow of
            // the list for the whole loop, which wouldn't allow the mutable
            // borrow of `self` in `add_backlink`
            let len = self.nodes[insert_idx].layers[lyr].len();
            for i in 0..len {
                // only borrow here, copying the value and ending the borrow before `add_backlink`
                let fw_link = self.nodes[insert_idx].layers[lyr][i];
                let backlink = Link {
                    node_index: insert_idx,
                    distance: fw_link.distance,
                };
                self.add_backlink(fw_link.node_index, backlink, lyr);
            }

            ep = self.nodes[insert_idx].layers[lyr]
                .iter()
                .min()
                .unwrap_or_else(|| {
                    panic!("ERROR: no neighbours found while inserting vec at layer={lyr}, an empty array was returned by select_neighbors (insert)")
                })
                .node_index;
        }

        if insert_lyr > self.max_layer {
            self.max_layer = insert_lyr;
            self.entry_point = insert_idx;
        }
    }

    pub fn search(&self, q: &[f32; D], k: usize) -> Vec<(usize, f32)> {
        if self.data.is_empty() {
            return Vec::new();
        }

        let mut ep = self.entry_point;
        for lyr in (1..=self.max_layer).rev() {
            ep = self
                .search_layer(q, ep, lyr, 1)
                .first()
                .unwrap_or_else(|| {
                    panic!("ERROR: search_layer@{lyr} returned an empty array (search)")
                })
                .node_index;
        }

        let mut results = self.search_layer(q, ep, 0, self.ef_search.max(k));
        // take k best from final layer search
        results
            .into_iter()
            .take(k)
            .map(|l| (l.node_index, l.distance))
            .collect()
    }

    fn search_layer(&self, q: &[f32; D], ep: usize, lyr: usize, ef: usize) -> Vec<Link> {
        assert!(ef > 0, "ef must be > 0");
        assert!(lyr <= self.max_layer, "layer not initialized",);
        assert!(ep < self.data.len(), "entry point out of bounds",);
        assert!(
            lyr < self.nodes[ep].layers.len(),
            "entry point does not exist in this layer"
        );

        self.epoch.set(self.epoch.get() + 1);
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

    fn select_neighbors(
        &self,
        qv: &[f32; D],
        lyr: usize,
        candidates: Vec<Link>,
        extend: bool,
        keep_pruned: bool,
    ) -> Vec<Link> {
        assert!(lyr <= self.max_layer, "layer not initialized",);

        self.epoch.set(self.epoch.get() + 1);
        let mut pq = BinaryHeap::<Reverse<Link>>::new();
        let mut discarded = BinaryHeap::<Reverse<Link>>::new();
        let mut best = Vec::<Link>::new();
        let max_connections = self.max_connections(lyr);

        for (node, vec, idx) in candidates.iter().map(|link| {
            (
                &self.nodes[link.node_index],
                &self.data[link.node_index],
                link.node_index,
            )
        }) {
            if node.epoch == self.epoch {
                continue;
            }
            node.epoch.set(self.epoch.get());
            pq.push(Reverse(Link {
                node_index: idx,
                distance: l2_squared(qv, vec),
            }));

            if extend {
                let neighs = &node.layers[lyr];
                for (neigh, vec, idx) in neighs.iter().map(|link| {
                    (
                        &self.nodes[link.node_index],
                        &self.data[link.node_index],
                        link.node_index,
                    )
                }) {
                    if neigh.epoch == self.epoch {
                        continue;
                    }
                    neigh.epoch.set(self.epoch.get());
                    pq.push(Reverse(Link {
                        node_index: idx,
                        distance: l2_squared(qv, vec),
                    }));
                }
            }
        }

        // no pruning required
        if (pq.len() <= max_connections) {
            return pq.into_sorted_vec().into_iter().map(|r| r.0).collect();
        }

        while let Some((node, vec, idx)) = pq.pop().map(|c| {
            (
                &self.nodes[c.0.node_index],
                &self.data[c.0.node_index],
                c.0.node_index,
            )
        }) && best.len() < max_connections
        {
            let mut diverse = true;
            let c_to_q = l2_squared(qv, vec);
            for other in best.iter().map(|link| &self.data[link.node_index]) {
                let c_to_other = l2_squared(vec, other);
                if c_to_q >= c_to_other {
                    diverse = false;
                    break;
                }
            }

            if diverse {
                best.push(Link {
                    node_index: idx,
                    distance: c_to_q,
                });
            } else if keep_pruned {
                discarded.push(Reverse(Link {
                    node_index: idx,
                    distance: c_to_q,
                }));
            }
        }

        if keep_pruned {
            while let Some(Reverse(link)) = discarded.pop()
                && best.len() < max_connections
            {
                best.push(link);
            }
        }

        best
    }

    fn add_backlink(&mut self, at: usize, link: Link, lyr: usize) {
        assert!(lyr <= self.max_layer, "layer not initialized",);
        assert!(at < self.data.len(), "backlink base index out of bounds",);
        assert!(
            link.node_index < self.data.len(),
            "backlink connection index out of bounds"
        );
        assert!(
            lyr < self.nodes[at].layers.len(),
            "node does not exist in this layer"
        );
        assert!(link.node_index != at, "can't link node to itself");

        let max_connections = self.max_connections(lyr);
        let links = &mut self.nodes[at].layers[lyr];

        links.push(link);
        if links.len() >= max_connections {
            let candidates = std::mem::take(links);
            let new_links = self.select_neighbors(&self.data[at], lyr, candidates, true, true);
            self.nodes[at].layers[lyr] = new_links;
        }
    }

    #[inline(always)]
    fn max_connections(&self, lyr: usize) -> usize {
        if lyr == 0 { self.M0 } else { self.M }
    }

    #[inline(always)]
    fn random_layer(&mut self) -> usize {
        (-self.rng.random::<f64>().ln() * self.ml).floor() as usize
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

