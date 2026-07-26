// =============================================================================
// meta_table.rs
//
// pretty-print `symbi_io::Metadata` and `symbi_io::TreeBuf`
// as box-drawn terminal tables. one Schema feeds:
//   • `symbi_io::Hdf5Backend` (production checkpoint)
//   • `symbi_io::JsonBackend` (introspection)
//   • `symbi_io::TreeDisplay`  (ASCII tree)
//   • `symbi_display::render_metadata` / `render_tree_buf` (this file —
//     framed, column-aligned tables suitable for the live-monitor UI).
//
// the tables use the existing `terminal::ansi` + box-drawing characters from
// `renderer::BoxChars`, so they match the look of the live `Table` widget
// (`crates/symbi-display/src/table.rs`). no new dependencies.
// =============================================================================

use std::fmt::Write;

use symbi_io::{Attr, DatasetBuf, Metadata, TreeBuf};

use crate::renderer::BoxChars;

/// render a `Metadata` bag as a 3-column table: `Key | Type | Value`. the
/// returned `String` carries newline-separated lines, ready for `println!`.
///
/// ```ignore
/// let extras = Metadata::new()
///     .with("problem", "kepler")
///     .with("ring_r0", 1.0)
///     .with("gm",      1.0);
/// println!("{}", render_metadata("simulation extras", &extras));
/// ```
pub fn render_metadata(title: &str, meta: &Metadata) -> String {
    let rows: Vec<[String; 3]> = meta
        .iter()
        .map(|(k, v)| [k.to_string(), attr_type(v).to_string(), attr_value(v)])
        .collect();
    render_table(title, &["Key", "Type", "Value"], &rows)
}

/// render a `TreeBuf` (whole-file schema as read back by `Hdf5Backend.read`)
/// as a nested table: every group's attrs + dataset summaries laid out as
/// labeled sub-tables.
pub fn render_tree_buf(root: &TreeBuf) -> String {
    let mut out = String::new();
    render_tree_node(&mut out, root, "");
    out
}

fn render_tree_node(out: &mut String, t: &TreeBuf, path: &str) {
    let qual = if path.is_empty() {
        if t.name.is_empty() {
            "<root>".to_string()
        } else {
            t.name.clone()
        }
    } else {
        format!("{path}/{}", t.name)
    };

    if !t.attrs.is_empty() {
        let rows: Vec<[String; 3]> = t
            .attrs
            .iter()
            .map(|(k, v)| [k.to_string(), attr_type(v).to_string(), attr_value(v)])
            .collect();
        writeln!(
            out,
            "{}",
            render_table(
                &format!("{qual} — attributes"),
                &["Key", "Type", "Value"],
                &rows,
            )
        )
        .ok();
    }
    if !t.datasets.is_empty() {
        let rows: Vec<[String; 3]> = t
            .datasets
            .iter()
            .map(|d| {
                [
                    d.name.clone(),
                    dataset_dtype(d).to_string(),
                    dataset_shape(d),
                ]
            })
            .collect();
        writeln!(
            out,
            "{}",
            render_table(
                &format!("{qual} — datasets"),
                &["Name", "DType", "Shape"],
                &rows,
            )
        )
        .ok();
    }
    for g in &t.groups {
        render_tree_node(out, g, &qual);
    }
}

// ---- helpers ---------------------------------------------------------------

fn attr_type(a: &Attr) -> &'static str {
    match a {
        Attr::Bool(_) => "bool",
        Attr::I64(_) => "i64",
        Attr::U64(_) => "u64",
        Attr::F64(_) => "f64",
        Attr::Str(_) => "str",
    }
}

fn attr_value(a: &Attr) -> String {
    match a {
        Attr::Bool(v) => v.to_string(),
        Attr::I64(v) => v.to_string(),
        Attr::U64(v) => v.to_string(),
        Attr::F64(v) => format!("{v:.6e}"),
        Attr::Str(s) => format!("\"{s}\""),
    }
}

fn dataset_dtype(d: &DatasetBuf) -> &'static str {
    match d.data {
        symbi_io::DataBuf::F64(_) => "f64",
        symbi_io::DataBuf::F32(_) => "f32",
        symbi_io::DataBuf::I64(_) => "i64",
        symbi_io::DataBuf::U64(_) => "u64",
        symbi_io::DataBuf::U8(_) => "u8",
    }
}

fn dataset_shape(d: &DatasetBuf) -> String {
    if d.shape.is_empty() {
        "[scalar]".to_string()
    } else {
        let parts: Vec<String> = d.shape.iter().map(|n| n.to_string()).collect();
        format!("[{}]", parts.join(" × "))
    }
}

// ----- the core box-drawn table renderer ------------------------------------

/// minimal column-padded box-drawn table. each row is `&[String; N]` for
/// fixed-arity tables (Metadata + Dataset rows are both 3 cols here).
fn render_table(title: &str, headers: &[&str; 3], rows: &[[String; 3]]) -> String {
    let bx = BoxChars::unicode();
    let mut widths = [0usize; 3];
    for (i, h) in headers.iter().enumerate() {
        widths[i] = widths[i].max(h.chars().count());
    }
    for row in rows {
        for (i, c) in row.iter().enumerate() {
            widths[i] = widths[i].max(c.chars().count());
        }
    }
    // tiny padding inside each cell
    let pad = 1;
    let mut cell_widths: [usize; 3] = std::array::from_fn(|i| widths[i] + 2 * pad);
    let cell_sum_inner: usize = cell_widths.iter().sum::<usize>() + 2; // + 2 internal verticals
    let want_title_w = title.chars().count() + 2 * pad;
    // distribute extra width equally across cells when title is wider than
    // the column sum — keeps title border, separators, and body cells all
    // aligned at the same total width.
    if want_title_w > cell_sum_inner {
        let extra = want_title_w - cell_sum_inner;
        let per_col = extra / 3;
        let remainder = extra % 3;
        for i in 0..3 {
            cell_widths[i] += per_col + if i < remainder { 1 } else { 0 };
            widths[i] += per_col + if i < remainder { 1 } else { 0 };
        }
    }
    let total_inner: usize = cell_widths.iter().sum::<usize>() + 2;

    let mut out = String::new();
    // top border
    out.push_str(bx.top_left);
    for _ in 0..total_inner {
        out.push_str(bx.horizontal);
    }
    out.push_str(bx.top_right);
    out.push('\n');
    // title row
    out.push_str(bx.vertical);
    let title_padded = format!("{:^width$}", title, width = total_inner);
    out.push_str(&title_padded);
    out.push_str(bx.vertical);
    out.push('\n');
    // header separator
    out.push_str(bx.t_left);
    for i in 0..3 {
        for _ in 0..cell_widths[i] {
            out.push_str(bx.horizontal);
        }
        out.push_str(if i + 1 < 3 { bx.cross } else { bx.t_right });
    }
    out.push('\n');
    // header row
    push_row(&mut out, &bx, headers, &widths, pad);
    // header/body separator
    out.push_str(bx.t_left);
    for i in 0..3 {
        for _ in 0..cell_widths[i] {
            out.push_str(bx.horizontal);
        }
        out.push_str(if i + 1 < 3 { bx.cross } else { bx.t_right });
    }
    out.push('\n');
    // body rows
    for row in rows {
        push_row(
            &mut out,
            &bx,
            &[row[0].as_str(), row[1].as_str(), row[2].as_str()],
            &widths,
            pad,
        );
    }
    // bottom border
    out.push_str(bx.bottom_left);
    for i in 0..3 {
        for _ in 0..cell_widths[i] {
            out.push_str(bx.horizontal);
        }
        out.push_str(if i + 1 < 3 { bx.t_up } else { bx.bottom_right });
    }
    out
}

fn push_row(out: &mut String, bx: &BoxChars, cells: &[&str; 3], widths: &[usize; 3], pad: usize) {
    out.push_str(bx.vertical);
    for (i, c) in cells.iter().enumerate() {
        out.push_str(&" ".repeat(pad));
        let used = c.chars().count();
        out.push_str(c);
        out.push_str(&" ".repeat(widths[i] - used + pad));
        out.push_str(bx.vertical);
    }
    out.push('\n');
}
