// =============================================================================
// display.rs
//
// `TreeDisplay` — pretty-print a `Tree` as an ASCII tree at the terminal.
// the SAME schema feeds HDF5 / JSON / display so a developer can `dbg!` the
// snapshot manifest without opening h5py. paired with `print_table` for a
// flatter "key | type | value" view of attrs.
// =============================================================================

use std::fmt;

use crate::attr::Attr;
use crate::tree::{Dataset, DatasetBuf, Tree, TreeBuf};

/// wrap a `&Tree` in a Display impl. usage:
/// ```ignore
/// println!("{}", TreeDisplay(&tree));
/// ```
pub struct TreeDisplay<'a, 'b>(pub &'a Tree<'b>);

impl<'a, 'b> fmt::Display for TreeDisplay<'a, 'b> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render(f, self.0, "", true)
    }
}

fn render(f: &mut fmt::Formatter<'_>, t: &Tree<'_>, prefix: &str, is_last: bool) -> fmt::Result {
    let connector = if prefix.is_empty() { "" } else if is_last { "└── " } else { "├── " };
    let name = if t.name.is_empty() { "<root>" } else { t.name.as_str() };
    writeln!(f, "{prefix}{connector}{name}/")?;

    // attribute + dataset count summary on a single line per group when
    // children are present — keeps the tree compact for big sims.
    let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });

    for (i, (k, v)) in t.attrs.iter().enumerate() {
        let is_last_attr = i + 1 == t.attrs.len()
            && t.datasets.is_empty() && t.groups.is_empty();
        let c = if is_last_attr { "└── " } else { "├── " };
        writeln!(f, "{child_prefix}{c}@{k} = {}", render_attr(v))?;
    }

    for (i, d) in t.datasets.iter().enumerate() {
        let is_last_ds = i + 1 == t.datasets.len() && t.groups.is_empty();
        let c = if is_last_ds { "└── " } else { "├── " };
        writeln!(f, "{child_prefix}{c}{}", render_dataset(d))?;
    }

    for (i, g) in t.groups.iter().enumerate() {
        let is_last_g = i + 1 == t.groups.len();
        render(f, g, &child_prefix, is_last_g)?;
    }
    Ok(())
}

fn render_attr(a: &Attr) -> String {
    match a {
        Attr::Bool(v) => format!("{v} : bool"),
        Attr::I64(v)  => format!("{v} : i64"),
        Attr::U64(v)  => format!("{v} : u64"),
        Attr::F64(v)  => format!("{v:.6e} : f64"),
        Attr::Str(s)  => format!("{s:?} : str"),
    }
}

fn render_dataset(d: &Dataset<'_>) -> String {
    let nelems: usize = d.shape.iter().product();
    let bytes = nelems * d.data.dtype().size_bytes();
    format!("{}: {:?}{} = {} elements ({} bytes)",
        d.name, d.data.dtype(), shape_str(&d.shape), nelems, bytes)
}

fn shape_str(shape: &[usize]) -> String {
    if shape.is_empty() {
        String::from("[scalar]")
    } else {
        let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
        format!("[{}]", parts.join("x"))
    }
}

// =============================================================================
// TreeBuf side — same rendering over owned `Vec<f64>` payloads. lets a
// checkpoint reader pretty-print the file in one line:
//
//     println!("{}", display_tree_buf(&backend.read(path)?, ""));
// =============================================================================

/// Display impl wrapping a `&TreeBuf` — owned counterpart of `TreeDisplay`.
pub struct TreeBufDisplay<'a>(pub &'a TreeBuf);

impl<'a> fmt::Display for TreeBufDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render_buf(f, self.0, "", true)
    }
}

/// shorthand: `eprintln!("{}", display_tree_buf(&tree, ""))`. the `prefix`
/// argument is reserved for future use (indenting under a header); pass `""`
/// for the standard layout.
pub fn display_tree_buf<'a>(tree: &'a TreeBuf, _prefix: &str) -> TreeBufDisplay<'a> {
    TreeBufDisplay(tree)
}

fn render_buf(f: &mut fmt::Formatter<'_>, t: &TreeBuf, prefix: &str, is_last: bool) -> fmt::Result {
    let connector = if prefix.is_empty() { "" } else if is_last { "└── " } else { "├── " };
    let name = if t.name.is_empty() { "<root>" } else { t.name.as_str() };
    writeln!(f, "{prefix}{connector}{name}/")?;
    let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });

    for (i, (k, v)) in t.attrs.iter().enumerate() {
        let is_last_attr = i + 1 == t.attrs.len()
            && t.datasets.is_empty() && t.groups.is_empty();
        let c = if is_last_attr { "└── " } else { "├── " };
        writeln!(f, "{child_prefix}{c}@{k} = {}", render_attr(v))?;
    }
    for (i, d) in t.datasets.iter().enumerate() {
        let is_last_ds = i + 1 == t.datasets.len() && t.groups.is_empty();
        let c = if is_last_ds { "└── " } else { "├── " };
        writeln!(f, "{child_prefix}{c}{}", render_dataset_buf(d))?;
    }
    for (i, g) in t.groups.iter().enumerate() {
        let is_last_g = i + 1 == t.groups.len();
        render_buf(f, g, &child_prefix, is_last_g)?;
    }
    Ok(())
}

fn render_dataset_buf(d: &DatasetBuf) -> String {
    let nelems: usize = d.shape.iter().product();
    let bytes = nelems * d.data.dtype().size_bytes();
    format!("{}: {:?}{} = {} elements ({} bytes)",
        d.name, d.data.dtype(), shape_str(&d.shape), nelems, bytes)
}
