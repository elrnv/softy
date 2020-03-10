use benchmark::*;
use tensr::*;

fn main() {
    let n = 2000;
    let m = ChunkedN::from_flat_with_stride(random_vec(n * n), n);
    let v = random_vec(n);
    //// symmetrize matrix to make consistent results
    //for i in 0..n {
    //    for j in 0..n {
    //        if j > i {
    //            let other = *m.view().at(j).at(i);
    //            let val = m.view_mut().isolate(i).isolate(j);
    //            *val = other;
    //        }
    //    }
    //}
    let mut size = 0;
    for _ in 0..1000 {
        size += inner(m.view(), v.view()).len();
        //size += inner_simd_unchecked(m.view(), v.view()).len();
        //size += lazy_expr(m.view(), v.view()).len();
        //size += outer_read_local(m.view(), v.view()).len();
        eprint!(".");
    }
    eprintln!("{}", size);
}
