// =============================================================================
// symbol.rs
//
// interned identifier used for const-generic dimension names in DimExpr
// (e.g., "D" in `Tensor<f64, D>`). interning buys cheap equality and
// folds whitespace variants onto one symbol.
//
// implementation: thread-local HashMap<String, Arc<str>>. each unique
// trimmed string maps to one Arc; `Symbol` wraps the Arc. equality is
// O(1) via Arc::ptr_eq (fast path) with a string-eq fallback for the
// unlikely case of cross-interner symbols.
//
// thread_local is correct here: rustc expands macros on a single thread
// per compilation unit, matching the existing
// elemental_graph_registry / field_group_registry pattern.
// =============================================================================

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

thread_local! {
    static INTERNER: RefCell<HashMap<String, Arc<str>>> = RefCell::new(HashMap::new());
}

/// interned identifier. created via `Symbol::intern(name)`; two
/// `Symbol`s compare equal iff their underlying strings are equal,
/// regardless of which interner they came from.
#[derive(Clone, Debug)]
pub struct Symbol(Arc<str>);

impl Symbol {
    /// intern a name. trims surrounding whitespace so " D " and "D"
    /// produce the same symbol.
    pub fn intern(name: &str) -> Self {
        let trimmed = name.trim();
        INTERNER.with(|i| {
            let mut map = i.borrow_mut();
            if let Some(existing) = map.get(trimmed) {
                return Symbol(existing.clone());
            }
            let arc: Arc<str> = trimmed.into();
            map.insert(trimmed.to_string(), arc.clone());
            Symbol(arc)
        })
    }

    /// read the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        // fast path: same Arc, same interner.
        // fallback: cross-interner safety via string equality.
        Arc::ptr_eq(&self.0, &other.0) || self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for Symbol {}

impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // hash the string contents, so cross-interner
        // symbols still hash to the same bucket as their string-equal
        // siblings.
        self.0.as_ref().hash(state);
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// serde as the plain string: serialize the interned str, re-intern on
// deserialize. the Arc identity is per-process and never crosses the wire, so
// round-tripping through the interner is the only correct reconstruction.
impl serde::Serialize for Symbol {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Ok(Symbol::intern(&name))
    }
}

macro_rules! typed_symbol {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        #[serde(transparent)]
        pub struct $name(Symbol);

        impl $name {
            pub fn new(name: impl AsRef<str>) -> Self {
                Self(Symbol::intern(name.as_ref()))
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl From<&str> for $name {
            fn from(name: &str) -> Self {
                Self::new(name)
            }
        }

        impl From<String> for $name {
            fn from(name: String) -> Self {
                Self::new(name)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl PartialEq<str> for $name {
            fn eq(&self, other: &str) -> bool {
                self.as_str() == other
            }
        }

        impl PartialEq<&str> for $name {
            fn eq(&self, other: &&str) -> bool {
                self.as_str() == *other
            }
        }

        impl PartialEq<String> for $name {
            fn eq(&self, other: &String) -> bool {
                self.as_str() == other
            }
        }
    };
}

typed_symbol!(InputKey, "Interned identity of a graph field input.");
typed_symbol!(OutputKey, "Interned identity of a graph output.");
typed_symbol!(
    ScalarParam,
    "Interned identity of a scalar kernel parameter."
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn same_name_interns_to_same_arc() {
        let a = Symbol::intern("D");
        let b = Symbol::intern("D");
        // fast-path equality should hold: both Arcs point to the same allocation.
        assert!(Arc::ptr_eq(&a.0, &b.0));
        assert_eq!(a, b);
    }

    #[test]
    fn different_names_are_distinct() {
        let a = Symbol::intern("D");
        let b = Symbol::intern("N");
        assert!(!Arc::ptr_eq(&a.0, &b.0));
        assert_ne!(a, b);
    }

    #[test]
    fn whitespace_is_trimmed() {
        let a = Symbol::intern("D");
        let b = Symbol::intern("  D  ");
        let c = Symbol::intern("\tD\n");
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert!(Arc::ptr_eq(&a.0, &b.0));
        assert!(Arc::ptr_eq(&b.0, &c.0));
    }

    #[test]
    fn as_str_round_trip() {
        let s = Symbol::intern("Hello");
        assert_eq!(s.as_str(), "Hello");
    }

    #[test]
    fn display_matches_as_str() {
        let s = Symbol::intern("D");
        assert_eq!(format!("{}", s), "D");
    }

    #[test]
    fn hash_matches_equality() {
        use std::collections::hash_map::DefaultHasher;

        let a = Symbol::intern("D");
        let b = Symbol::intern(" D ");
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn usable_in_hashset() {
        // an interned symbol must round-trip through a HashSet correctly.
        let mut set: HashSet<Symbol> = HashSet::new();
        set.insert(Symbol::intern("D"));
        set.insert(Symbol::intern("N"));
        set.insert(Symbol::intern("D")); // dup — should not increase len

        assert_eq!(set.len(), 2);
        assert!(set.contains(&Symbol::intern("D")));
        assert!(set.contains(&Symbol::intern("N")));
        assert!(!set.contains(&Symbol::intern("K")));
    }

    #[test]
    fn empty_string_is_valid() {
        // empty symbol is allowed; it'd be a parser-level error to reach
        // here, but the interner doesn't reject it.
        let a = Symbol::intern("");
        let b = Symbol::intern("");
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "");
    }
}
