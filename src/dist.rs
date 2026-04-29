#[inline(always)]
pub fn l2_squared<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut total = 0.0;
    for i in 0..D {
        let diff = a[i] - b[i];
        total += diff * diff;
    }
    total
}
