use std::cell::Cell;

use crate::link::Link;

pub struct Node {
    pub layers: Vec<Vec<Link>>,
    pub epoch: Cell<usize>,
}
