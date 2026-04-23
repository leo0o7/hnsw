#[inline(always)]
pub fn l2_squared<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b).map(|(x1, x2)| (x1 - x2).powi(2)).sum()
}
