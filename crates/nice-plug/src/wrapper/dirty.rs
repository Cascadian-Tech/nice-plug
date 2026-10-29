//! Main-thread editor identity and nonblocking notification recursion guard.

use std::{
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    thread::{self, ThreadId},
};

pub(crate) struct EditorLease {
    owner: ThreadId,
    phase: AtomicU8,
}
impl EditorLease {
    pub fn preparing() -> Self {
        Self {
            owner: thread::current().id(),
            phase: AtomicU8::new(0),
        }
    }
    pub fn activate(&self) -> bool {
        self.phase
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
    pub fn close(&self) {
        self.phase.store(2, Ordering::Release);
    }
    pub fn is_live_owner(&self) -> bool {
        self.owner == thread::current().id() && self.phase.load(Ordering::Acquire) == 1
    }
}
pub(crate) struct InFlight<'a>(&'a AtomicBool);
impl<'a> InFlight<'a> {
    pub fn enter(flag: &'a AtomicBool) -> Option<Self> {
        flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self(flag))
    }
}
impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
