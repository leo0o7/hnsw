use std::cell::Cell;

use serde::{Deserialize, Serialize};

use crate::{
    disk::{deserialize_epoch_as_zero, serialize_epoch_as_zero},
    link::Link,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub layers: Vec<Vec<Link>>,
    #[serde(
        serialize_with = "serialize_epoch_as_zero",
        deserialize_with = "deserialize_epoch_as_zero"
    )]
    pub epoch: Cell<usize>,
}

pub(crate) fn nodes_heap_usage_bytes(nodes: &Vec<Node>) -> usize {
    let mut bytes = nodes.capacity() * size_of::<Node>();
    for node in nodes {
        bytes += node.layers.capacity() * size_of::<Vec<Link>>();
        for layer in &node.layers {
            bytes += layer.capacity() * size_of::<Link>();
        }
    }
    bytes
}
