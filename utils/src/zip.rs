/// Zip multiple iterators
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
    }
}
