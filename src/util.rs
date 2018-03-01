use std::slice;
use std::mem::size_of;

/// Reinterpret a given slice as a slice of another type. This function checks that the resulting
/// slice is appropriately sized.
pub fn reinterpret_mut_slice<T, S>(slice: &mut [T]) -> &mut [S] {
    // We must be able to split the given slice into appropriately sized chunks.
    assert_eq!((slice.len() * size_of::<T>()) % size_of::<S>(), 0);
    let nu_len = (slice.len() * size_of::<T>()) / size_of::<S>();
    unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut S, nu_len) }
}

/// Reinterpret a given slice as a slice of another type. This function checks that the resulting
/// slice is appropriately sized.
pub fn reinterpret_slice<T, S>(slice: &[T]) -> &[S] {
    // We must be able to split the given slice into appropriately sized chunks.
    assert_eq!((slice.len() * size_of::<T>()) % size_of::<S>(), 0);
    let nu_len = (slice.len() * size_of::<T>()) / size_of::<S>();
    unsafe { slice::from_raw_parts(slice.as_ptr() as *const S, nu_len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::math::Vector3;

    /// Check that we can reinterpret a slice of `Vector3<f64>`s as a slice of `f64`s.
    #[test]
    fn reinterpret_slice_test() {
        let slice = &[
            Vector3([0.1, 1.0, 2.0]),
            Vector3([1.2, 1.4, 2.1]),
            Vector3([0.5, 3.2, 4.0]),
        ];
        let flat = &[0.1, 1.0, 2.0, 1.2, 1.4, 2.1, 0.5, 3.2, 4.0];

        let nu_flat: &[f64] = reinterpret_slice(slice);
        assert_eq!(*nu_flat, *flat);

        let nu_slice: &[Vector3<f64>] = reinterpret_slice(flat);
        assert_eq!(*nu_slice, *slice);

        let flat_mut = &mut [0.1, 1.0, 2.0, 1.2, 1.4, 2.1, 0.5, 3.2, 4.0];

        let nu_mut_slice: &mut [Vector3<f64>] = reinterpret_mut_slice(flat_mut);
        for v in nu_mut_slice.iter_mut() {
            *v += Vector3([1.0, 2.0, 3.0]);
        }

        assert_eq!(
            *nu_mut_slice,
            [
                Vector3([1.1, 3.0, 5.0]),
                Vector3([2.2, 3.4, 5.1]),
                Vector3([1.5, 5.2, 7.0]),
            ]
        );
    }
}
