/// Zip multiple iterators
/// 
/// # Examples
/// 
/// Zipping multiple iterators can significantly reduce the boulerplate ".zip" calls and extra
/// parentheses.
///
/// ```rust
/// # fn main() {
///     use utils::zip;
///
///     let av = vec![1,2,3,4];
///     let bv = vec![5,6,7,8];
///     let cv = vec![9,10,11,12];
///
///     for (a, b, c) in zip!(av.into_iter(), bv.into_iter(), cv.into_iter()) {
///         println!("({}, {}, {})", a, b, c);
///     }
/// # }
/// ```
///
/// This macro can be used in other contexts where iterators are useful. The trailing
/// comma is optional for convenience.
///
/// ```rust
/// # fn main() {
///     use utils::zip;
///
///     let a_vec = vec![1,2,3,4];
///     let b_vec = vec![5,6,7,8];
///     let c_vec = vec![9,10,11,12];
///
///     let zipped: Vec<(usize, usize, usize)> = zip!(
///         a_vec.into_iter(),
///         b_vec.into_iter(),
///         c_vec.into_iter(), // with trailing comma
///     ).collect();
///
///     assert_eq!(zipped, vec![(1,5,9), (2,6,10), (3,7,11), (4,8,12)])
/// # }
/// ```
#[macro_export]
macro_rules! zip {
    // Implementation calls
    (@flatten |$in:pat| $out:expr ) => { // base case for flatten
        |$in| $out
    };
    (@flatten |$in:pat| ($($out:tt)*), $_:expr $(,$rest:expr)*) => { // flatten the tuple
        zip!(@flatten |($in, x)| ( $($out)*, x ) $(,$rest)*)
    };
    // Main entry point
    ($iter:expr $(, $rest:expr)*) => {
        $iter $(.zip($rest))*.map(zip!(@flatten |x| (x) $(,$rest)*))
    };
    // Same as the main entry point, but with a trailing comma
    ($iter:expr $(, $rest:expr)* ,) => {
        zip!($iter $(,$rest)*)
    }
}
