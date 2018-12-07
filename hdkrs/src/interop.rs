//!
//! This module defines utility functions for translating C types to rust types.
//!

use crate::cffi;
pub use libc::c_void;
use std::ffi::CString;
use std::ptr::NonNull;

//
// Translate interrupt callback
//

/// Utility to cast the void pointer to the interrupt checker function a valid Rust type.
pub unsafe fn interrupt_callback(
    checker: *mut c_void,
    check_interrupt: Option<extern "C" fn(*const c_void) -> bool>,
) -> impl Fn() -> bool {
    let interrupt_ref = &*checker; // conversion needed since *mut c_void is not Send
    move || match check_interrupt {
        Some(cb) => cb(interrupt_ref as *const c_void),
        None => true,
    }
}

//
// Translate result type
//

/// The Rust version of the cook result enum.
pub enum CookResult {
    Success(String),
    Warning(String),
    Error(String),
}

impl From<CookResult> for cffi::CookResult {
    fn from(res: CookResult) -> cffi::CookResult {
        match res {
            CookResult::Success(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Success,
            },
            CookResult::Warning(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Warning,
            },
            CookResult::Error(msg) => cffi::CookResult {
                message: CString::new(msg.as_str()).unwrap().into_raw(),
                tag: cffi::CookResultTag::Error,
            },
        }
    }
}

//
// Translate pointers
//

/// A convenience utility to convert a mutable pointer to an optional mutable reference.
pub unsafe fn as_mut<'a, T: 'a>(ptr: *mut T) -> Option<&'a mut T> {
    NonNull::new(ptr).map(|x| &mut *x.as_ptr())
}

/// A convenience utility to convert a mutable pointer to an optional owning box.
pub unsafe fn into_box<'a, T: 'a>(ptr: *mut T) -> Option<Box<T>> {
    NonNull::new(ptr).map(|x| Box::from_raw(x.as_ptr()))
}
