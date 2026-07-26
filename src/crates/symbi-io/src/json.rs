// =============================================================================
// json.rs
//
// `JsonBackend` — schema-only introspection. writes the tree's metadata +
// dataset shapes (NOT the field data) as a small JSON document. lets a tool
// dump "what's in this checkpoint" without h5py:
//
//   symbi-io --inspect kepler_final.h5 > schema.json
//
// matched with a small `Display` impl so the same Tree prints as a clean
// ASCII tree at the terminal (no JSON tooling required).
// =============================================================================

use std::fmt::Write;

use crate::attr::Attr;
use crate::tree::{Dataset, Tree};

/// render a `Tree` as a JSON object describing the schema. dataset data is
/// NOT serialized — only `name`, `shape`, and `dtype`. attributes carry
/// their typed value (so the JSON is self-describing — no schema-side
/// document required).
pub fn schema_json(tree: &Tree<'_>) -> String {
    let mut s = String::new();
    write_node(&mut s, tree, 0);
    s
}

fn write_node(s: &mut String, t: &Tree<'_>, indent: usize) {
    let pad = "  ".repeat(indent);
    writeln!(s, "{pad}{{").ok();
    let inner = "  ".repeat(indent + 1);
    writeln!(s, "{inner}\"name\": \"{}\",", esc(&t.name)).ok();
    // attrs
    writeln!(s, "{inner}\"attrs\": {{").ok();
    for (i, (k, v)) in t.attrs.iter().enumerate() {
        let sep = if i + 1 < t.attrs.len() { "," } else { "" };
        writeln!(s, "{inner}  \"{}\": {}{sep}", esc(k), attr_json(v)).ok();
    }
    writeln!(s, "{inner}}},").ok();
    // datasets
    writeln!(s, "{inner}\"datasets\": [").ok();
    for (i, d) in t.datasets.iter().enumerate() {
        let sep = if i + 1 < t.datasets.len() { "," } else { "" };
        writeln!(s, "{inner}  {}{sep}", dataset_json(d)).ok();
    }
    writeln!(s, "{inner}],").ok();
    // groups
    writeln!(s, "{inner}\"groups\": [").ok();
    for (i, g) in t.groups.iter().enumerate() {
        let sep = if i + 1 < t.groups.len() { "," } else { "" };
        write_node(s, g, indent + 2);
        // patch separator after the closing brace
        if !sep.is_empty() {
            let trimmed = s.trim_end_matches('\n');
            *s = format!("{trimmed}{sep}\n");
        }
    }
    writeln!(s, "{inner}]").ok();
    writeln!(s, "{pad}}}").ok();
}

fn attr_json(a: &Attr) -> String {
    match a {
        Attr::Bool(v) => v.to_string(),
        Attr::I64(v) => v.to_string(),
        Attr::U64(v) => v.to_string(),
        Attr::F64(v) => {
            // json doesn't allow NaN / Inf; surface them as strings instead.
            if v.is_finite() {
                v.to_string()
            } else {
                format!("\"{v}\"")
            }
        }
        Attr::Str(s) => format!("\"{}\"", esc(s)),
    }
}

fn dataset_json(d: &Dataset<'_>) -> String {
    format!(
        "{{ \"name\": \"{}\", \"dtype\": \"{:?}\", \"shape\": {:?} }}",
        esc(&d.name),
        d.data.dtype(),
        d.shape,
    )
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
