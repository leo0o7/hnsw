use std::cell::Cell;

use crate::link::Link;

#[derive(Debug, Clone)]
pub struct Node {
    pub layers: Vec<Vec<Link>>,
    pub epoch: Cell<usize>,
}
