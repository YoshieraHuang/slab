#![cfg_attr(not(feature = "std"), no_std)]
#![warn(
    missing_debug_implementations,
    missing_docs,
    rust_2018_idioms,
    unreachable_pub
)]
#![doc(test(
    no_crate_inject,
    attr(deny(warnings, rust_2018_idioms), allow(dead_code, unused_variables))
))]

//! Pre-allocated storage for a uniform data type.
//!
//! `Slab` provides pre-allocated storage for a single data type. If many values
//! of a single type are being allocated, it can be more efficient to
//! pre-allocate the necessary storage. Since the size of the type is uniform,
//! memory fragmentation can be avoided. Storing, clearing, and lookup
//! operations become very cheap.
//!
//! While `Slab` may look like other Rust collections, it is not intended to be
//! used as a general purpose collection. The primary difference between `Slab`
//! and `Vec` is that `Slab` returns the key when storing the value.
//!
//! It is important to note that keys may be reused. In other words, once a
//! value associated with a given key is removed from a slab, that key may be
//! returned from future calls to `insert`.
//!
//! # Examples
//!
//! Basic storing and retrieval.
//!
//! ```
//! # use slab::*;
//! let mut slab = Slab::<_, u8>::new();
//!
//! let hello = slab.insert("hello");
//! let world = slab.insert("world");
//!
//! assert_eq!(slab[hello], "hello");
//! assert_eq!(slab[world], "world");
//!
//! slab[world] = "earth";
//! assert_eq!(slab[world], "earth");
//! ```
//!
//! Sometimes it is useful to be able to associate the key with the value being
//! inserted in the slab. This can be done with the `vacant_entry` API as such:
//!
//! ```
//! # use slab::*;
//! let mut slab = Slab::<_, u8>::new();
//!
//! let hello = {
//!     let entry = slab.vacant_entry();
//!     let key = entry.key();
//!
//!     entry.insert((key, "hello"));
//!     key
//! };
//!
//! assert_eq!(hello, slab[hello].0);
//! assert_eq!("hello", slab[hello].1);
//! ```
//!
//! It is generally a good idea to specify the desired capacity of a slab at
//! creation time. Note that `Slab` will grow the internal capacity when
//! attempting to insert a new value once the existing capacity has been reached.
//! To avoid this, add a check.
//!
//! ```
//! # use slab::*;
//! let mut slab = Slab::with_capacity(1024u16);
//!
//! // ... use the slab
//!
//! if slab.len() == slab.capacity() {
//!     panic!("slab full");
//! }
//!
//! slab.insert("the slab is not at capacity yet");
//! ```
//!
//! # Capacity and reallocation
//!
//! The capacity of a slab is the amount of space allocated for any future
//! values that will be inserted in the slab. This is not to be confused with
//! the *length* of the slab, which specifies the number of actual values
//! currently being inserted. If a slab's length is equal to its capacity, the
//! next value inserted into the slab will require growing the slab by
//! reallocating.
//!
//! For example, a slab with capacity 10 and length 0 would be an empty slab
//! with space for 10 more stored values. Storing 10 or fewer elements into the
//! slab will not change its capacity or cause reallocation to occur. However,
//! if the slab length is increased to 11 (due to another `insert`), it will
//! have to reallocate, which can be slow. For this reason, it is recommended to
//! use [`Slab::with_capacity`] whenever possible to specify how many values the
//! slab is expected to store.
//!
//! # Implementation
//!
//! `Slab` is backed by a `Vec` of slots. Each slot is either occupied or
//! vacant. `Slab` maintains a stack of vacant slots using a linked list. To
//! find a vacant slot, the stack is popped. When a slot is released, it is
//! pushed onto the stack.
//!
//! If there are no more available slots in the stack, then `Vec::reserve(1)` is
//! called and a new slot is created.
//!
//! [`Slab::with_capacity`]: struct.Slab.html#with_capacity

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std as alloc;

#[cfg(feature = "serde")]
mod serde;

mod uint;

use alloc::vec::{self, Vec};
use core::iter::{self, FromIterator, FusedIterator};
use core::{fmt, mem, ops, slice};
use uint::Uint;

/// Pre-allocated storage for a uniform data type
///
/// See the [module documentation] for more details.
///
/// [module documentation]: index.html
#[derive(Clone)]
pub struct Slab<T, U: Uint = usize> {
    // Chunk of memory
    entries: Vec<Entry<T, U>>,

    // Number of Filled elements currently in the slab
    len: U,

    // Offset of the next available slot in the slab. Set to the slab's
    // capacity when the slab is full.
    next: U,

    // Max capacity of filled elements
    max_capacity: U,
}

impl<T, U: Uint> Default for Slab<T, U> {
    fn default() -> Self {
        Slab::new()
    }
}

/// A handle to a vacant entry in a `Slab`.
///
/// `VacantEntry` allows constructing values with the key that they will be
/// assigned to.
///
/// # Examples
///
/// ```
/// # use slab::*;
/// let mut slab = Slab::<_, usize>::new();
///
/// let hello = {
///     let entry = slab.vacant_entry();
///     let key = entry.key();
///
///     entry.insert((key, "hello"));
///     key
/// };
///
/// assert_eq!(hello, slab[hello].0);
/// assert_eq!("hello", slab[hello].1);
/// ```
#[derive(Debug)]
pub struct VacantEntry<'a, T, U: Uint = usize> {
    slab: &'a mut Slab<T, U>,
    key: U,
}

/// A consuming iterator over the values stored in a `Slab`
pub struct IntoIter<T, U: Uint = usize> {
    entries: iter::Enumerate<vec::IntoIter<Entry<T, U>>>,
    len: U,
}

/// An iterator over the values stored in the `Slab`
pub struct Iter<'a, T, U: Uint = usize> {
    entries: iter::Enumerate<slice::Iter<'a, Entry<T, U>>>,
    len: U,
}

/// A mutable iterator over the values stored in the `Slab`
pub struct IterMut<'a, T, U: Uint = usize> {
    entries: iter::Enumerate<slice::IterMut<'a, Entry<T, U>>>,
    len: U,
}

/// A draining iterator for `Slab`
pub struct Drain<'a, T, U: Uint = usize> {
    inner: vec::Drain<'a, Entry<T, U>>,
    len: U,
}

#[derive(Clone)]
enum Entry<T, U: Uint = usize> {
    Vacant(U),
    Occupied(T),
}

impl<T, U: Uint> Slab<T, U> {
    /// Construct a new, empty `Slab`.
    ///
    /// The function does not allocate and the returned slab will have no
    /// capacity until `insert` is called or capacity is explicitly reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let slab: Slab<i32, u8> = Slab::new();
    /// ```
    pub fn new() -> Slab<T, U> {
        Slab::with_capacity(U::zero())
    }

    /// Construct a new, empty `Slab` with the specified capacity.
    ///
    /// The returned slab will be able to store exactly `capacity` without
    /// reallocating. If `capacity` is 0, the slab will not allocate.
    ///
    /// It is important to note that this function does not specify the *length*
    /// of the returned slab, but only the capacity. For an explanation of the
    /// difference between length and capacity, see [Capacity and
    /// reallocation](index.html#capacity-and-reallocation).
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::with_capacity(10usize);
    ///
    /// // The slab contains no values, even though it has capacity for more
    /// assert_eq!(slab.len(), 0);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     slab.insert(i);
    /// }
    ///
    /// // ...but this may make the slab reallocate
    /// slab.insert(11);
    /// ```
    pub fn with_capacity(capacity: U) -> Slab<T, U> {
        Slab {
            entries: Vec::with_capacity(capacity.usize()),
            next: U::zero(),
            len: U::zero(),
            max_capacity: U::max_value(),
        }
    }

    /// Set the max capacity of this Slab
    /// 
    /// # Examples
    /// 
    /// ```
    /// # use slab::*;
    /// let mut slab: Slab<i32, usize> = Slab::new();
    /// slab.set_max_capacity(5);
    /// assert_eq!(slab.max_capacity(), 5);
    /// ```
    pub fn set_max_capacity(&mut self, max: U) {
        self.max_capacity = max;
    }

    /// Return the max capacity of this Slab
    pub fn max_capacity(&self) -> U {
        self.max_capacity
    }

    /// Return the number of values the slab can store without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let slab: Slab<i32, usize> = Slab::with_capacity(10);
    /// assert_eq!(slab.capacity(), 10);
    /// ```
    pub fn capacity(&self) -> U {
        U::from_usize(self.entries.capacity())
    }

    #[inline]
    fn entries_len(&self) -> U {
        U::from_usize(self.entries.len())
    }

    /// Reserve capacity for at least `additional` more values to be stored
    /// without allocating.
    ///
    /// `reserve` does nothing if the slab already has sufficient capacity for
    /// `additional` more values. If more capacity is required, a new segment of
    /// memory will be allocated and all existing values will be copied into it.
    /// As such, if the slab is already very large, a call to `reserve` can end
    /// up being expensive.
    ///
    /// The slab may reserve more than `additional` extra space in order to
    /// avoid frequent reallocations. Use `reserve_exact` instead to guarantee
    /// that only the requested space is allocated.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    /// slab.insert("hello");
    /// slab.reserve(10);
    /// assert!(slab.capacity() >= 11);
    /// ```
    pub fn reserve(&mut self, additional: U) {
        if self.capacity() - self.len >= additional {
            return;
        }
        if additional + self.entries_len() >= self.max_capacity() {
            panic!("capacity overflow")
        }
        let need_add = additional - (self.entries_len() - self.len);
        self.entries.reserve(need_add.usize());
    }

    /// Reserve the minimum capacity required to store exactly `additional`
    /// more values.
    ///
    /// `reserve_exact` does nothing if the slab already has sufficient capacity
    /// for `additional` more valus. If more capacity is required, a new segment
    /// of memory will be allocated and all existing values will be copied into
    /// it.  As such, if the slab is already very large, a call to `reserve` can
    /// end up being expensive.
    ///
    /// Note that the allocator may give the slab more space than it requests.
    /// Therefore capacity can not be relied upon to be precisely minimal.
    /// Prefer `reserve` if future insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    /// slab.insert("hello");
    /// slab.reserve_exact(10);
    /// assert!(slab.capacity() >= 11);
    /// ```
    pub fn reserve_exact(&mut self, additional: U) {
        if self.capacity() - self.len >= additional {
            return;
        }
        if additional + self.entries_len() >= self.max_capacity() {
            panic!("capacity overflow")
        }
        let need_add = additional - (self.entries_len() - self.len);
        self.entries.reserve_exact(need_add.usize());
    }

    /// Shrink the capacity of the slab as much as possible without invalidating keys.
    ///
    /// Because values cannot be moved to a different index, the slab cannot
    /// shrink past any stored values.
    /// It will drop down as close as possible to the length but the allocator may
    /// still inform the underlying vector that there is space for a few more elements.
    ///
    /// This function can take O(n) time even when the capacity cannot be reduced
    /// or the allocation is shrunk in place. Repeated calls run in O(1) though.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::with_capacity(10usize);
    ///
    /// for i in 0..3 {
    ///     slab.insert(i);
    /// }
    ///
    /// slab.shrink_to_fit();
    /// assert!(slab.capacity() >= 3 && slab.capacity() < 10);
    /// ```
    ///
    /// The slab cannot shrink past the last present value even if previous
    /// values are removed:
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::with_capacity(10usize);
    ///
    /// for i in 0..4 {
    ///     slab.insert(i);
    /// }
    ///
    /// slab.remove(0);
    /// slab.remove(3);
    ///
    /// slab.shrink_to_fit();
    /// assert!(slab.capacity() >= 3 && slab.capacity() < 10);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        // Remove all vacant entries after the last occupied one, so that
        // the capacity can be reduced to what is actually needed.
        // If the slab is empty the vector can simply be cleared, but that
        // optimization would not affect time complexity when T: Drop.
        let len_before = self.entries.len();
        while let Some(&Entry::Vacant(_)) = self.entries.last() {
            self.entries.pop();
        }

        // Removing entries breaks the list of vacant entries,
        // so it must be repaired
        if self.entries.len() != len_before {
            // Some vacant entries were removed, so the list now likely¹
            // either contains references to the removed entries, or has an
            // invalid end marker. Fix this by recreating the list.
            self.recreate_vacant_list();
            // ¹: If the removed entries formed the tail of the list, with the
            // most recently popped entry being the head of them, (so that its
            // index is now the end marker) the list is still valid.
            // Checking for that unlikely scenario of this infrequently called
            // is not worth the code complexity.
        }

        self.entries.shrink_to_fit();
    }

    /// Iterate through all entries to recreate and repair the vacant list.
    /// self.len must be correct and is not modified.
    fn recreate_vacant_list(&mut self) {
        self.next = self.entries_len();
        // We can stop once we've found all vacant entries
        let mut remaining_vacant = self.entries_len() - self.len;
        // Iterate in reverse order so that lower keys are at the start of
        // the vacant list. This way future shrinks are more likely to be
        // able to remove vacant entries.
        for (i, entry) in self.entries.iter_mut().enumerate().rev() {
            if remaining_vacant == U::zero() {
                break;
            }
            if let Entry::Vacant(ref mut next) = *entry {
                *next = self.next;
                self.next = U::from_usize(i);
                remaining_vacant.dec();
            }
        }
    }

    /// Reduce the capacity as much as possible, changing the key for elements when necessary.
    ///
    /// To allow updating references to the elements which must be moved to a new key,
    /// this function takes a closure which is called before moving each element.
    /// The second and third parameters to the closure are the current key and
    /// new key respectively.
    /// In case changing the key for one element turns out not to be possible,
    /// the move can be cancelled by returning `false` from the closure.
    /// In that case no further attempts at relocating elements is made.
    /// If the closure unwinds, the slab will be left in a consistent state,
    /// but the value that the closure panicked on might be removed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    ///
    /// let mut slab = Slab::with_capacity(10usize);
    /// let a = slab.insert('a');
    /// slab.insert('b');
    /// slab.insert('c');
    /// slab.remove(a);
    /// slab.compact(|&mut value, from, to| {
    ///     assert_eq!((value, from, to), ('c', 2, 0));
    ///     true
    /// });
    /// assert!(slab.capacity() >= 2 && slab.capacity() < 10);
    /// ```
    ///
    /// The value is not moved when the closure returns `Err`:
    ///
    /// ```
    /// # use slab::*;
    ///
    /// let mut slab = Slab::with_capacity(100usize);
    /// let a = slab.insert('a');
    /// let b = slab.insert('b');
    /// slab.remove(a);
    /// slab.compact(|&mut value, from, to| false);
    /// assert_eq!(slab.iter().next(), Some((b, &'b')));
    /// ```
    pub fn compact<F>(&mut self, mut rekey: F)
    where
        F: FnMut(&mut T, U, U) -> bool,
    {
        // If the closure unwinds, we need to restore a valid list of vacant entries
        struct CleanupGuard<'a, T, U: Uint> {
            slab: &'a mut Slab<T, U>,
            decrement: bool,
        }
        impl<T, U: Uint> Drop for CleanupGuard<'_, T, U> {
            fn drop(&mut self) {
                if self.decrement {
                    // Value was popped and not pushed back on
                    self.slab.len.dec();
                }
                self.slab.recreate_vacant_list();
            }
        }
        let mut guard = CleanupGuard {
            slab: self,
            decrement: true,
        };

        let mut occupied_until = 0;
        // While there are vacant entries
        while guard.slab.entries_len() > guard.slab.len {
            // Find a value that needs to be moved,
            // by popping entries until we find an occopied one.
            // (entries cannot be empty because 0 is not greater than anything)
            if let Some(Entry::Occupied(mut value)) = guard.slab.entries.pop() {
                // Found one, now find a vacant entry to move it to
                while let Some(&Entry::Occupied(_)) = guard.slab.entries.get(occupied_until) {
                    occupied_until += 1;
                }
                // Let the caller try to update references to the key
                if !rekey(&mut value, guard.slab.entries_len(), U::from_usize(occupied_until)) {
                    // Changing the key failed, so push the entry back on at its old index.
                    guard.slab.entries.push(Entry::Occupied(value));
                    guard.decrement = false;
                    guard.slab.entries.shrink_to_fit();
                    return;
                    // Guard drop handles cleanup
                }
                // Put the value in its new spot
                guard.slab.entries[occupied_until] = Entry::Occupied(value);
                // ... and mark it as occupied (this is optional)
                occupied_until += 1;
            }
        }
        guard.slab.next = guard.slab.len;
        guard.slab.entries.shrink_to_fit();
        // Normal cleanup is not necessary
        mem::forget(guard);
    }

    /// Clear the slab of all values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// for i in 0..3 {
    ///     slab.insert(i);
    /// }
    ///
    /// slab.clear();
    /// assert!(slab.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.entries.clear();
        self.len = U::zero();
        self.next = U::zero();
    }

    /// Return the number of stored values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::new();
    ///
    /// for i in 0..3 {
    ///     slab.insert(i);
    /// }
    ///
    /// assert_eq!(3usize, slab.len());
    /// ```
    pub fn len(&self) -> U {
        self.len
    }

    /// Return `true` if there are no values stored in the slab.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    /// assert!(slab.is_empty());
    ///
    /// slab.insert(1);
    /// assert!(!slab.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == U::zero()
    }

    /// Return an iterator over the slab.
    ///
    /// This function should generally be **avoided** as it is not efficient.
    /// Iterators must iterate over every slot in the slab even if it is
    /// vacant. As such, a slab with a capacity of 1 million but only one
    /// stored value must still iterate the million slots.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    ///
    /// for i in 0..3 {
    ///     slab.insert(i);
    /// }
    ///
    /// let mut iterator = slab.iter();
    ///
    /// assert_eq!(iterator.next(), Some((0, &0)));
    /// assert_eq!(iterator.next(), Some((1, &1)));
    /// assert_eq!(iterator.next(), Some((2, &2)));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T, U> {
        Iter {
            entries: self.entries.iter().enumerate(),
            len: self.len,
        }
    }

    /// Return an iterator that allows modifying each value.
    ///
    /// This function should generally be **avoided** as it is not efficient.
    /// Iterators must iterate over every slot in the slab even if it is
    /// vacant. As such, a slab with a capacity of 1 million but only one
    /// stored value must still iterate the million slots.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// let key1 = slab.insert(0);
    /// let key2 = slab.insert(1);
    ///
    /// for (key, val) in slab.iter_mut() {
    ///     if key == key1 {
    ///         *val += 2;
    ///     }
    /// }
    ///
    /// assert_eq!(slab[key1], 2);
    /// assert_eq!(slab[key2], 1);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<'_, T, U> {
        IterMut {
            entries: self.entries.iter_mut().enumerate(),
            len: self.len,
        }
    }

    #[inline]
    fn entries_get(&self, index: U) -> Option<&Entry<T, U>> {
        self.entries.get(index.usize())
    }

    #[inline]
    fn entries_get_mut(&mut self, index: U) -> Option<&mut Entry<T, U>> {
        self.entries.get_mut(index.usize())
    }

    /// Return a reference to the value associated with the given key.
    ///
    /// If the given key is not associated with a value, then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    /// let key = slab.insert("hello");
    ///
    /// assert_eq!(slab.get(key), Some(&"hello"));
    /// assert_eq!(slab.get(123), None);
    /// ```
    pub fn get(&self, key: U) -> Option<&T> {
        match self.entries_get(key) {
            Some(&Entry::Occupied(ref val)) => Some(val),
            _ => None,
        }
    }

    /// Return a mutable reference to the value associated with the given key.
    ///
    /// If the given key is not associated with a value, then `None` is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    /// let key = slab.insert("hello");
    ///
    /// *slab.get_mut(key).unwrap() = "world";
    ///
    /// assert_eq!(slab[key], "world");
    /// assert_eq!(slab.get_mut(123), None);
    /// ```
    pub fn get_mut(&mut self, key: U) -> Option<&mut T> {
        match self.entries_get_mut(key) {
            Some(&mut Entry::Occupied(ref mut val)) => Some(val),
            _ => None,
        }
    }

    /// Return two mutable references to the values associated with the two
    /// given keys simultaneously.
    ///
    /// If any one of the given keys is not associated with a value, then `None`
    /// is returned.
    ///
    /// This function can be used to get two mutable references out of one slab,
    /// so that you can manipulate both of them at the same time, eg. swap them.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// use std::mem;
    ///
    /// let mut slab = Slab::<_, usize>::new();
    /// let key1 = slab.insert(1);
    /// let key2 = slab.insert(2);
    /// let (value1, value2) = slab.get2_mut(key1, key2).unwrap();
    /// mem::swap(value1, value2);
    /// assert_eq!(slab[key1], 2);
    /// assert_eq!(slab[key2], 1);
    /// ```
    pub fn get2_mut(&mut self, key1: U, key2: U) -> Option<(&mut T, &mut T)> {
        assert!(key1 != key2);

        let (entry1, entry2);

        if key1 > key2 {
            let (slice1, slice2) = self.entries.split_at_mut(key1.usize());
            entry1 = slice2.get_mut(0);
            entry2 = slice1.get_mut(key2.usize());
        } else {
            let (slice1, slice2) = self.entries.split_at_mut(key2.usize());
            entry1 = slice1.get_mut(key1.usize());
            entry2 = slice2.get_mut(0);
        }

        match (entry1, entry2) {
            (
                Some(&mut Entry::Occupied(ref mut val1)),
                Some(&mut Entry::Occupied(ref mut val2)),
            ) => Some((val1, val2)),
            _ => None,
        }
    }

    /// Return a reference to the value associated with the given key without
    /// performing bounds checking.
    ///
    /// For a safe alternative see [`get`](Slab::get).
    ///
    /// This function should be used with care.
    ///
    /// # Safety
    ///
    /// The key must be within bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    /// let key = slab.insert(2);
    ///
    /// unsafe {
    ///     assert_eq!(slab.get_unchecked(key), &2);
    /// }
    /// ```
    pub unsafe fn get_unchecked(&self, key: U) -> &T {
        match *self.entries.get_unchecked(key.usize()) {
            Entry::Occupied(ref val) => val,
            _ => unreachable!(),
        }
    }

    /// Return a mutable reference to the value associated with the given key
    /// without performing bounds checking.
    ///
    /// For a safe alternative see [`get_mut`](Slab::get_mut).
    ///
    /// This function should be used with care.
    ///
    /// # Safety
    ///
    /// The key must be within bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    /// let key = slab.insert(2);
    ///
    /// unsafe {
    ///     let val = slab.get_unchecked_mut(key);
    ///     *val = 13;
    /// }
    ///
    /// assert_eq!(slab[key], 13);
    /// ```
    pub unsafe fn get_unchecked_mut(&mut self, key: U) -> &mut T {
        match *self.entries.get_unchecked_mut(key.usize()) {
            Entry::Occupied(ref mut val) => val,
            _ => unreachable!(),
        }
    }

    /// Return two mutable references to the values associated with the two
    /// given keys simultaneously without performing bounds checking and safety
    /// condition checking.
    ///
    /// For a safe alternative see [`get2_mut`](Slab::get2_mut).
    ///
    /// This function should be used with care.
    ///
    /// # Safety
    ///
    /// - Both keys must be within bounds.
    /// - The condition `key1 != key2` must hold.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// use std::mem;
    ///
    /// let mut slab = Slab::<_, usize>::new();
    /// let key1 = slab.insert(1);
    /// let key2 = slab.insert(2);
    /// let (value1, value2) = unsafe { slab.get2_unchecked_mut(key1, key2) };
    /// mem::swap(value1, value2);
    /// assert_eq!(slab[key1], 2);
    /// assert_eq!(slab[key2], 1);
    /// ```
    pub unsafe fn get2_unchecked_mut(&mut self, key1: U, key2: U) -> (&mut T, &mut T) {
        let ptr1 = self.entries.get_unchecked_mut(key1.usize()) as *mut Entry<T, U>;
        let ptr2 = self.entries.get_unchecked_mut(key2.usize()) as *mut Entry<T, U>;
        match (&mut *ptr1, &mut *ptr2) {
            (&mut Entry::Occupied(ref mut val1), &mut Entry::Occupied(ref mut val2)) => {
                (val1, val2)
            }
            _ => unreachable!(),
        }
    }

    /// Get the key for an element in the slab.
    ///
    /// The reference must point to an element owned by the slab.
    /// Otherwise this function will panic.
    /// This is a constant-time operation because the key can be calculated
    /// from the reference with pointer arithmetic.
    ///
    /// # Panics
    ///
    /// This function will panic if the reference does not point to an element
    /// of the slab.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    ///
    /// let mut slab = Slab::<_, u8>::new();
    /// let key = slab.insert(String::from("foo"));
    /// let value = &slab[key];
    /// assert_eq!(slab.key_of(value), key);
    /// ```
    ///
    /// Values are not compared, so passing a reference to a different locaton
    /// will result in a panic:
    ///
    /// ```should_panic
    /// # use slab::*;
    ///
    /// let mut slab = Slab::<_, u8>::new();
    /// let key = slab.insert(0);
    /// let bad = &0;
    /// slab.key_of(bad); // this will panic
    /// unreachable!();
    /// ```
    pub fn key_of(&self, present_element: &T) -> U {
        let element_ptr = present_element as *const T as usize;
        let base_ptr = self.entries.as_ptr() as usize;
        // Use wrapping subtraction in case the reference is bad
        let byte_offset = element_ptr.wrapping_sub(base_ptr);
        // The division rounds away any offset of T inside Entry
        // The size of Entry<T, U> is never zero even if T is due to Vacant(usize)
        let key = U::from_usize(byte_offset / mem::size_of::<Entry<T, U>>());
        // Prevent returning unspecified (but out of bounds) values
        if key >= self.entries_len() {
            panic!("The reference points to a value outside this slab");
        }
        // The reference cannot point to a vacant entry, because then it would not be valid
        key
    }

    /// Insert a value in the slab, returning key assigned to the value.
    ///
    /// The returned key can later be used to retrieve or remove the value using indexed
    /// lookup and `remove`. Additional capacity is allocated if needed. See
    /// [Capacity and reallocation](index.html#capacity-and-reallocation).
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows `U`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    /// let key = slab.insert("hello");
    /// assert_eq!(slab[key], "hello");
    /// ```
    pub fn insert(&mut self, val: T) -> U {
        let key = self.next;

        self.insert_at(key, val);

        key
    }

    /// Insert a value in the slab, returning key assigned to the value.
    /// If the number of stored values exceeds max capacity after this insertion, this value will not be inserted and
    /// this function returns None.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in the vector overflows U.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::new();
    /// slab.set_max_capacity(5usize);
    /// for i in 0..5 {
    ///     assert!(slab.insert_check(i).is_some());
    /// }
    /// assert_eq!(slab.insert_check(5), None);
    /// ```
    pub fn insert_check(&mut self, val: T) -> Option<U> {
        if self.len() >= self.max_capacity {
            return None
        }

        let key = self.next;

        self.insert_at(key, val);

        Some(key)
    }

    /// Return a handle to a vacant entry allowing for further manipulation.
    ///
    /// This function is useful when creating values that must contain their
    /// slab key. The returned `VacantEntry` reserves a slot in the slab and is
    /// able to query the associated key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab[hello].0);
    /// assert_eq!("hello", slab[hello].1);
    /// ```
    pub fn vacant_entry(&mut self) -> VacantEntry<'_, T, U> {
        VacantEntry {
            key: self.next,
            slab: self,
        }
    }

    fn insert_at(&mut self, key: U, val: T) {
        self.len.inc();

        if key == self.entries_len() {
            self.entries.push(Entry::Occupied(val));
            self.next = key.add_one();
        } else {
            self.next = match self.entries_get(key) {
                Some(&Entry::Vacant(next)) => next,
                _ => unreachable!(),
            };
            self.entries[key.usize()] = Entry::Occupied(val);
        }
    }

    /// Remove and return the value associated with the given key.
    ///
    /// The key is then released and may be associated with future stored
    /// values.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not associated with a value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// let hello = slab.insert("hello");
    ///
    /// assert_eq!(slab.remove(hello), "hello");
    /// assert!(!slab.contains(hello));
    /// ```
    pub fn remove(&mut self, key: U) -> T {
        let next = self.next;
        if let Some(entry) = self.entries_get_mut(key) {
            // Swap the entry at the provided value
            let prev = mem::replace(entry, Entry::Vacant(next));

            match prev {
                Entry::Occupied(val) => {
                    self.len.dec();
                    self.next = key;
                    return val;
                }
                _ => {
                    // Woops, the entry is actually vacant, restore the state
                    *entry = prev;
                }
            }
        }
        panic!("invalid key");
    }

    /// Return `true` if a value is associated with the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// let hello = slab.insert("hello");
    /// assert!(slab.contains(hello));
    ///
    /// slab.remove(hello);
    ///
    /// assert!(!slab.contains(hello));
    /// ```
    pub fn contains(&self, key: U) -> bool {
        match self.entries_get(key) {
            Some(&Entry::Occupied(_)) => true,
            _ => false,
        }
    }

    /// Retain only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(usize, &mut e)`
    /// returns false. This method operates in place and preserves the key
    /// associated with the retained values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    ///
    /// let k1 = slab.insert(0);
    /// let k2 = slab.insert(1);
    /// let k3 = slab.insert(2);
    ///
    /// slab.retain(|key, val| key == k1 || *val == 1);
    ///
    /// assert!(slab.contains(k1));
    /// assert!(slab.contains(k2));
    /// assert!(!slab.contains(k3));
    ///
    /// assert_eq!(2, slab.len());
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(U, &mut T) -> bool,
    {
        for i in 0..self.entries.len() {
            let keep = match self.entries[i] {
                Entry::Occupied(ref mut v) => f(U::from_usize(i), v),
                _ => true,
            };

            if !keep {
                self.remove(U::from_usize(i));
            }
        }
    }

    /// Return a draining iterator that removes all elements from the slab and
    /// yields the removed items.
    ///
    /// Note: Elements are removed even if the iterator is only partially
    /// consumed or not consumed at all.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, u8>::new();
    ///
    /// let _ = slab.insert(0);
    /// let _ = slab.insert(1);
    /// let _ = slab.insert(2);
    ///
    /// {
    ///     let mut drain = slab.drain();
    ///
    ///     assert_eq!(Some(0), drain.next());
    ///     assert_eq!(Some(1), drain.next());
    ///     assert_eq!(Some(2), drain.next());
    ///     assert_eq!(None, drain.next());
    /// }
    ///
    /// assert!(slab.is_empty());
    /// ```
    pub fn drain(&mut self) -> Drain<'_, T, U> {
        let old_len = self.len;
        self.len = U::zero();
        self.next = U::zero();
        Drain {
            inner: self.entries.drain(..),
            len: old_len,
        }
    }
}

impl<T, U: Uint> ops::Index<U> for Slab<T, U> {
    type Output = T;

    fn index(&self, key: U) -> &T {
        match self.entries_get(key.into()) {
            Some(&Entry::Occupied(ref v)) => v,
            _ => panic!("invalid key"),
        }
    }
}

impl<T, U: Uint> ops::IndexMut<U> for Slab<T, U> {
    fn index_mut(&mut self, key: U) -> &mut T {
        match self.entries_get_mut(key) {
            Some(&mut Entry::Occupied(ref mut v)) => v,
            _ => panic!("invalid key"),
        }
    }
}

impl<T, U: Uint> IntoIterator for Slab<T, U> {
    type Item = (U, T);
    type IntoIter = IntoIter<T, U>;

    fn into_iter(self) -> IntoIter<T, U> {
        IntoIter {
            entries: self.entries.into_iter().enumerate(),
            len: self.len,
        }
    }
}

impl<'a, T, U: Uint> IntoIterator for &'a Slab<T, U> {
    type Item = (U, &'a T);
    type IntoIter = Iter<'a, T, U>;

    fn into_iter(self) -> Iter<'a, T, U> {
        self.iter()
    }
}

impl<'a, T, U: Uint> IntoIterator for &'a mut Slab<T, U> {
    type Item = (U, &'a mut T);
    type IntoIter = IterMut<'a, T, U>;

    fn into_iter(self) -> IterMut<'a, T, U> {
        self.iter_mut()
    }
}

/// Create a slab from an iterator of key-value pairs.
///
/// If the iterator produces duplicate keys, the previous value is replaced with the later one.
/// The keys does not need to be sorted beforehand, and this function always
/// takes O(n) time.
/// Note that the returned slab will use space proportional to the largest key,
/// so don't use `Slab` with untrusted keys.
///
/// # Examples
///
/// ```
/// # use slab::*;
///
/// let vec = vec![(2,'a'), (6,'b'), (7,'c')];
/// let slab = vec.into_iter().collect::<Slab<char, usize>>();
/// assert_eq!(slab.len(), 3);
/// assert!(slab.capacity() >= 8);
/// assert_eq!(slab[2], 'a');
/// ```
///
/// With duplicate and unsorted keys:
///
/// ```
/// # use slab::*;
///
/// let vec = vec![(20,'a'), (10,'b'), (11,'c'), (10,'d')];
/// let slab = vec.into_iter().collect::<Slab<char, usize>>();
/// assert_eq!(slab.len(), 3);
/// assert_eq!(slab[10], 'd');
/// ```
impl<T, U: Uint> FromIterator<(U, T)> for Slab<T, U> {
    fn from_iter<I>(iterable: I) -> Self
    where
        I: IntoIterator<Item = (U, T)>,
    {
        let iterator = iterable.into_iter();
        let mut slab = Self::with_capacity(U::from_usize(iterator.size_hint().0));

        let mut vacant_list_broken = false;
        for (key, value) in iterator {
            if key < slab.entries_len() {
                // iterator is not sorted, might need to recreate vacant list
                if let Entry::Vacant(_) = slab.entries[key.usize()] {
                    vacant_list_broken = true;
                    slab.len.inc();
                }
                // if an element with this key already exists, replace it.
                // This is consisent with HashMap and BtreeMap
                slab.entries[key.usize()] = Entry::Occupied(value);
            } else {
                // insert holes as necessary
                while slab.entries_len() < key {
                    // add the entry to the start of the vacant list
                    let next = slab.next;
                    slab.next = slab.entries_len();
                    slab.entries.push(Entry::Vacant(next));
                }
                slab.entries.push(Entry::Occupied(value));
                slab.len.inc();
            }
        }
        if slab.len == slab.entries_len() {
            // no vacant enries, so next might not have been updated
            slab.next = slab.entries_len();
        } else if vacant_list_broken {
            slab.recreate_vacant_list();
        }
        slab
    }
}

impl<T, U: Uint> fmt::Debug for Slab<T, U>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Slab")
            .field("len", &self.len)
            .field("cap", &self.capacity())
            .finish()
    }
}

impl<T, U: Uint> fmt::Debug for IntoIter<T, U>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Iter")
            .field("remaining", &self.len)
            .finish()
    }
}

impl<T, U: Uint> fmt::Debug for Iter<'_, T, U>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Iter")
            .field("remaining", &self.len)
            .finish()
    }
}

impl<T, U: Uint> fmt::Debug for IterMut<'_, T, U>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("IterMut")
            .field("remaining", &self.len)
            .finish()
    }
}

impl<T, U: Uint> fmt::Debug for Drain<'_, T, U> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Drain").finish()
    }
}

// ===== VacantEntry =====

impl<'a, T, U: Uint> VacantEntry<'a, T, U> {
    /// Insert a value in the entry, returning a mutable reference to the value.
    ///
    /// To get the key associated with the value, use `key` prior to calling
    /// `insert`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab[hello].0);
    /// assert_eq!("hello", slab[hello].1);
    /// ```
    pub fn insert(self, val: T) -> &'a mut T {
        self.slab.insert_at(self.key, val);

        match self.slab.entries_get_mut(self.key) {
            Some(&mut Entry::Occupied(ref mut v)) => v,
            _ => unreachable!(),
        }
    }

    /// Return the key associated with this entry.
    ///
    /// A value stored in this entry will be associated with this key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use slab::*;
    /// let mut slab = Slab::<_, usize>::new();
    ///
    /// let hello = {
    ///     let entry = slab.vacant_entry();
    ///     let key = entry.key();
    ///
    ///     entry.insert((key, "hello"));
    ///     key
    /// };
    ///
    /// assert_eq!(hello, slab[hello].0);
    /// assert_eq!("hello", slab[hello].1);
    /// ```
    pub fn key(&self) -> U {
        self.key
    }
}

// ===== IntoIter =====

impl<T, U: Uint> Iterator for IntoIter<T, U> {
    type Item = (U, T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next() {
            if let Entry::Occupied(v) = entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len.usize(), Some(self.len.usize()))
    }
}

impl<T, U: Uint> DoubleEndedIterator for IntoIter<T, U> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next_back() {
            if let Entry::Occupied(v) = entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }
}

impl<T, U: Uint> ExactSizeIterator for IntoIter<T, U> {
    fn len(&self) -> usize {
        self.len.usize()
    }
}

impl<T, U: Uint> FusedIterator for IntoIter<T, U> {}

// ===== Iter =====

impl<'a, T, U: Uint> Iterator for Iter<'a, T, U> {
    type Item = (U, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next() {
            if let Entry::Occupied(ref v) = *entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len.usize(), Some(self.len.usize()))
    }
}

impl<T, U: Uint> DoubleEndedIterator for Iter<'_, T, U> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next_back() {
            if let Entry::Occupied(ref v) = *entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }
}

impl<T, U: Uint> ExactSizeIterator for Iter<'_, T, U> {
    fn len(&self) -> usize {
        self.len.usize()
    }
}

impl<T, U: Uint> FusedIterator for Iter<'_, T, U> {}

// ===== IterMut =====

impl<'a, T, U: Uint> Iterator for IterMut<'a, T, U> {
    type Item = (U, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next() {
            if let Entry::Occupied(ref mut v) = *entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len.usize(), Some(self.len.usize()))
    }
}

impl<T, U: Uint> DoubleEndedIterator for IterMut<'_, T, U> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((key, entry)) = self.entries.next_back() {
            if let Entry::Occupied(ref mut v) = *entry {
                self.len.dec();
                return Some((U::from_usize(key), v));
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }
}

impl<T, U: Uint> ExactSizeIterator for IterMut<'_, T, U> {
    fn len(&self) -> usize {
        self.len.usize()
    }
}

impl<T, U: Uint> FusedIterator for IterMut<'_, T, U> {}

// ===== Drain =====

impl<T, U: Uint> Iterator for Drain<'_, T, U> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.inner.next() {
            if let Entry::Occupied(v) = entry {
                self.len.dec();
                return Some(v);
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len.usize(), Some(self.len.usize()))
    }
}

impl<T, U: Uint> DoubleEndedIterator for Drain<'_, T, U> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.inner.next_back() {
            if let Entry::Occupied(v) = entry {
                self.len.dec();
                return Some(v);
            }
        }

        debug_assert_eq!(self.len, U::zero());
        None
    }
}

impl<T, U: Uint> ExactSizeIterator for Drain<'_, T, U> {
    fn len(&self) -> usize {
        self.len.usize()
    }
}

impl<T, U: Uint> FusedIterator for Drain<'_, T, U> {}
