use std::ops::*;
use std::cmp::*;

/// unsigned integer
pub trait Uint: Clone + Copy + std::fmt::Debug + PartialEq + Eq + PartialOrd + Ord + Sub<Output=Self> + SubAssign + Add<Output=Self> + AddAssign {
    /// return the largest value that can be represented by this interger
    fn max_value() -> Self;
    /// return the zero value
    fn zero() -> Self;
    /// decrement one
    fn dec(&mut self);
    /// increment one
    fn inc(&mut self);
    /// add one
    fn add_one(self) -> Self;
    /// substrate one
    fn sub_one(self) -> Self;
    /// range started at 0
    fn range(self) -> Range<Self>;
    // convert to usize
    fn usize(self) -> usize;
    // convert from usize
    fn from_usize(v: usize) -> Self;
}

macro_rules! Uintify {
    ($uint:ty) => {
        impl Uint for $uint {
            #[inline]
            fn max_value() -> Self { <$uint>::MAX }
            #[inline]
            fn zero() -> Self { <$uint>::MIN }
            #[inline]
            fn dec(&mut self) { *self -= 1; }
            #[inline]
            fn inc(&mut self) { *self += 1; }
            #[inline]
            fn add_one(self) -> Self { self + 1 }
            #[inline]
            fn sub_one(self) -> Self { self - 1 }
            #[inline]
            fn range(self) -> Range<Self> {
                0..self
            }
            #[inline]
            fn usize(self) -> usize {
                self as usize
            }
            #[inline]
            fn from_usize(v: usize) -> Self {
                v as Self
            }
        }
    };
}

Uintify!(u8);
Uintify!(u16);
Uintify!(u32);
Uintify!(u64);
Uintify!(u128);
Uintify!(usize);