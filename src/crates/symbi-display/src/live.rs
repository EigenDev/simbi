// =============================================================================
// live.rs
//
// the live tabbed diagnostic dashboard (design 40). a pure render over a
// `DiagnosticView` snapshot — the same code drives the throwaway skeleton
// (dummy data) and the production `Table` (a live per-cadence reduction), so
// look-and-feel iterates in the fast example while production inherits it.
//
// layout: a window header, a stat strip, a tab bar, a per-tab body, and a footer
// key strip. the overview tab is the hero panel (reserved for a future field
// view) beside the run-health cards, with the log below.
//
// usage:
//   terminal.draw(|frame| live::render(frame, &view))?;
// =============================================================================

use ratatui::Frame;
use serde::{Deserialize, Serialize};

use crate::hostinfo::HostStats;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Dataset, GraphType, LineGauge, Padding, Paragraph,
    Sparkline, Tabs, Wrap,
};

// catppuccin-ish palette + diagnostic accents.
const BORDER: Color = Color::Indexed(240); // dim card border
const BORDER_HERO: Color = Color::Indexed(67); // brighter hero border
const TITLE_DIM: Color = Color::Indexed(245);
const TEXT: Color = Color::Indexed(252);
const VALUE: Color = Color::Indexed(255);
const DIM: Color = Color::Indexed(244);
const GOLD: Color = Color::Indexed(220);
const LAV: Color = Color::Indexed(183);
const GREEN: Color = Color::Indexed(114);
const TEAL: Color = Color::Indexed(116);
const BLUE: Color = Color::Indexed(111);
const AMBER: Color = Color::Indexed(215);
const BADGE_BG: Color = Color::Indexed(29);
const BADGE_FG: Color = Color::Indexed(235);

/// the visible tab strip. the grid (regrid) tab is amr-only, so a uniform-grid
/// run (one level) drops it rather than showing a dead panel.
pub fn tab_names(has_amr: bool) -> &'static [&'static str] {
    if has_amr {
        &["overview", "diagnostics", "grid", "log", "config"]
    } else {
        &["overview", "diagnostics", "log", "config"]
    }
}
const SPIN: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// everything the tabbed dashboard renders for one frame. per-physics fields are
/// `Option`, so the overview draws only the cards whose data exists (a newtonian
/// run has no div·B or max W).
#[derive(Clone, Serialize, Deserialize)]
pub struct DiagnosticView {
    // header
    pub app_title: String, // "hydroflux — kelvin_helmholtz.toml"
    pub regime: String,    // stat-strip badge, e.g. "RHD"
    pub attached: String,  // "attached · rank 0 / 1" (or empty)
    pub paused: bool,
    pub frame: u64, // drives the spinner animation
    // stat strip
    pub t: f64,
    pub step: u64,
    pub dt: f64,
    pub wall_secs: f64,
    pub throughput_mzcups: f64,
    /// run progress toward t_final, 0..=100 (time / t_final). drives the header bar.
    pub progress: usize,
    // active tab
    pub tab: usize,
    pub config_scroll: u16,
    // charts
    pub throughput_hist: Vec<f64>,
    pub dt_hist: Vec<f64>,
    // conservation & constraints — each history is present only once the solver
    // reduces it, so the whole card (and each row) appears only when it has data.
    pub mass_drift: Option<Vec<f64>>,
    pub energy_drift: Option<Vec<f64>>,
    pub div_b: Option<Vec<f64>>, // mhd only
    pub max_w: Option<f64>,      // rhd / rmhd only
    pub cfl: f64,
    pub cfl_max: f64,
    pub blocks_per_level: Vec<u64>,
    // panels
    pub log: Vec<(String, String)>,          // (timestamp, text)
    pub config: Vec<(String, String, String)>, // config-tab rows: (section, property, value)
    // a decimated 2D field slice for the overview hero heatmap; None -> the hero
    // falls back to the throughput chart.
    pub field: Option<FieldSlice>,
    /// number of selectable fields for this run (density, + pressure / W / |B| when
    /// present); bounds the `f`-key cycle. 0 or 1 -> nothing to cycle.
    pub field_count: usize,
    /// compute-host + process resource sample (hostname, cores, rss vs total ram);
    /// None until sampled. drives the machine card.
    pub host: Option<HostStats>,
}

/// perceptually-uniform colormaps for the field heatmap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Colormap {
    Viridis,
    Inferno,
    Magma,
}

/// a 2D field slice, already decimated toward display resolution. `data` is
/// row-major `width * height`; the renderer samples it to the panel's pixel grid
/// and colormaps `[vmin, vmax]`. keeping this small (screen-sized, not grid-sized)
/// is the whole point — a 4096^2 grid and a 256^2 grid cost the same to draw.
#[derive(Clone, Serialize, Deserialize)]
pub struct FieldSlice {
    pub label: String, // "density · inferno · slice z=0"
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub vmin: f64,
    pub vmax: f64,
    pub cmap: Colormap,
    /// physical-shape (polar) slice: letterbox to the slice's aspect ratio instead
    /// of stretching to the panel (a circle drawn as an ellipse defeats the
    /// projection). half-block halves are ~square, so slice aspect is preserved
    /// in half-pixel space. index-space rectangles stretch to fill, as before.
    pub preserve_aspect: bool,
    /// colormap the field in log10 space (the `l` key, injected render-side like
    /// `cmap`): density / pressure span decades in an accretion flow, and a linear
    /// normalization paints everything but the peak one color. positive data only;
    /// a non-positive range falls back to linear.
    pub log_scale: bool,
}

fn fg(c: Color) -> Style {
    Style::default().fg(c)
}
fn fgb(c: Color) -> Style {
    Style::default().fg(c).add_modifier(Modifier::BOLD)
}

/// format seconds as MM:SS.
pub fn fmt_wall(s: f64) -> String {
    let s = s as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// si-suffix a magnitude for a compact axis / stat label.
pub fn humanize(v: f64) -> String {
    if v >= 1e9 {
        format!("{:.1}G", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.0}k", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

/// normalize a float history into relative sparkline bar heights (0..64).
fn spark(hist: &[f64]) -> Vec<u64> {
    if hist.is_empty() {
        return Vec::new();
    }
    let min = hist.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = hist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-30);
    hist.iter()
        .map(|&v| (((v - min) / range) * 64.0) as u64)
        .collect()
}

/// a rounded card with a dim border and subtle title.
fn card(title: &str) -> Block<'_> {
    Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(fg(BORDER))
        .padding(Padding::horizontal(1))
        .title(Span::styled(format!(" {title} "), fg(TITLE_DIM)))
}

/// render the whole tabbed dashboard for one frame.
pub fn render(frame: &mut Frame, view: &DiagnosticView) {
    let v = Layout::vertical([
        Constraint::Length(1), // window header
        Constraint::Length(1), // stat strip
        Constraint::Length(1), // tabs
        Constraint::Min(0),    // body
        Constraint::Length(1), // footer keys
    ])
    .split(frame.area());

    render_header(frame, v[0], view);
    render_statstrip(frame, v[1], view);

    // amr-only tabs drop out for uniform grids; dispatch by name so the active
    // index maps to the right panel regardless of which tabs are present.
    let tabs = tab_names(view.blocks_per_level.len() > 1);
    let active = view.tab.min(tabs.len().saturating_sub(1));
    render_tabs(frame, v[2], tabs, active);
    match tabs[active] {
        "overview" => render_overview(frame, v[3], view),
        "diagnostics" => render_diagnostics(frame, v[3], view),
        "grid" => render_grid(frame, v[3], view),
        "log" => render_log(frame, v[3], view),
        _ => render_config(frame, v[3], view),
    }
    render_footer(frame, v[4], tabs[active]);
}

fn render_header(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    // title on the left; the run-progress bar on the free right side (or the
    // attach note when attached to a running sim, which owns that slot instead).
    let cols = Layout::horizontal([Constraint::Min(20), Constraint::Length(30)]).split(area);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(view.app_title.clone(), fgb(LAV)))),
        cols[0],
    );
    if !view.attached.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(view.attached.clone(), fg(DIM)))).right_aligned(),
            cols[1],
        );
    } else {
        // the unfilled track is DIM (visible), not BORDER (near-background) — at low
        // progress the bar is almost all track, so an invisible track reads as just
        // a stray percentage. a leading label marks it unambiguously as progress.
        let gauge = LineGauge::default()
            .ratio((view.progress as f64 / 100.0).clamp(0.0, 1.0))
            .filled_style(fgb(GOLD))
            .unfilled_style(fg(DIM))
            .label(Span::styled(format!("{:>3}% ", view.progress), fg(DIM)));
        frame.render_widget(gauge, cols[1]);
    }
}

fn render_statstrip(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let spin = SPIN[view.frame as usize % SPIN.len()];
    let (dot, status) = if view.paused {
        (AMBER, "paused")
    } else {
        (GREEN, "integrating")
    };
    let line = Line::from(vec![
        Span::styled(
            format!(" {} ", view.regime),
            Style::default().fg(BADGE_FG).bg(BADGE_BG).add_modifier(Modifier::BOLD),
        ),
        Span::styled("   t ", fg(DIM)),
        Span::styled(format!("{:.4}", view.t), fgb(VALUE)),
        Span::styled("   step ", fg(DIM)),
        Span::styled(format!("{}", view.step), fgb(VALUE)),
        Span::styled("   dt ", fg(DIM)),
        Span::styled(format!("{:.1e}", view.dt), fgb(VALUE)),
        Span::styled("   wall ", fg(DIM)),
        Span::styled(fmt_wall(view.wall_secs), fgb(VALUE)),
        Span::styled("   throughput ", fg(DIM)),
        Span::styled(format!("{:.0} Mzcups", view.throughput_mzcups), fgb(VALUE)),
        Span::styled(format!("   {spin} {status}"), fgb(dot)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_tabs(frame: &mut Frame, area: Rect, tabs: &[&str], active: usize) {
    let widget = Tabs::new(tabs.iter().copied())
        .select(Some(active))
        .style(fg(DIM))
        .highlight_style(fgb(GOLD))
        .divider(Span::styled("·", fg(BORDER)));
    frame.render_widget(widget, area);
}

fn render_footer(frame: &mut Frame, area: Rect, active_tab: &str) {
    // page-aware hints: the arrow-scroll drives only the config panel, and the
    // field / orientation / colormap keys act on the overview heatmap.
    let mut keys: Vec<(&str, &str)> = vec![("space", "pause"), ("s", "step"), ("tab", "switch")];
    if active_tab == "config" {
        keys.push(("\u{2191}\u{2193}", "scroll"));
    }
    if active_tab == "overview" {
        keys.push(("f", "field"));
        keys.push(("o", "orient"));
        keys.push(("c", "cmap"));
        keys.push(("l", "log"));
    }
    keys.push(("w", "checkpoint"));
    keys.push(("q", "quit"));
    let mut spans = Vec::new();
    for (ii, (k, label)) in keys.iter().enumerate() {
        if ii > 0 {
            spans.push(Span::styled("  ·  ", fg(BORDER)));
        }
        spans.push(Span::styled(*k, fgb(GOLD)));
        spans.push(Span::styled(format!(" {label}"), fg(DIM)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// -- overview -----------------------------------------------------------------

fn render_overview(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(4)]).split(area);
    let cols =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(rows[0]);
    render_hero(frame, cols[0], view);
    render_cards(frame, cols[1], view);
    render_log(frame, rows[1], view);
}

/// the hero panel: a live field heatmap when a slice is present, otherwise the
/// throughput trace (the reserved-slot fallback).
fn render_hero(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    match &view.field {
        Some(f) => render_field(frame, area, f),
        None => render_throughput_hero(frame, area, view),
    }
}

/// the hero panel — reserved for the live field view (tier 2); holds
/// the throughput trace.
fn render_throughput_hero(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(fg(BORDER_HERO))
        .padding(Padding::horizontal(1))
        .title(Span::styled(" THROUGHPUT  Mzcups ", fgb(GOLD)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || view.throughput_hist.len() < 2 {
        return;
    }
    let points: Vec<(f64, f64)> = view
        .throughput_hist
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();
    let x_max = (view.throughput_hist.len() - 1) as f64;
    let y_max = view
        .throughput_hist
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(fg(TEAL))
        .data(&points);
    let chart = Chart::new(vec![dataset])
        .x_axis(Axis::default().style(fg(BORDER)).bounds([0.0, x_max]))
        .y_axis(
            Axis::default()
                .style(fg(BORDER))
                .bounds([0.0, y_max * 1.1])
                .labels([Span::styled("0", fg(DIM)), Span::styled(humanize(y_max), fg(DIM))]),
        );
    frame.render_widget(chart, inner);
}

// perceptually-uniform colormap control points (9 stops each), lerped per pixel.
const VIRIDIS: [(u8, u8, u8); 9] = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (110, 206, 88),
    (253, 231, 37),
];
const INFERNO: [(u8, u8, u8); 9] = [
    (0, 0, 4),
    (20, 11, 52),
    (66, 10, 104),
    (120, 28, 109),
    (175, 55, 84),
    (220, 93, 60),
    (247, 148, 32),
    (252, 209, 86),
    (252, 255, 164),
];
const MAGMA: [(u8, u8, u8); 9] = [
    (0, 0, 4),
    (28, 16, 68),
    (79, 18, 123),
    (129, 37, 129),
    (181, 54, 122),
    (229, 80, 100),
    (251, 135, 97),
    (254, 194, 135),
    (252, 253, 191),
];

/// map a normalized value [0,1] to a truecolor via a lerped colormap LUT.
fn colormap(t: f64, cmap: Colormap) -> Color {
    let stops: &[(u8, u8, u8); 9] = match cmap {
        Colormap::Viridis => &VIRIDIS,
        Colormap::Inferno => &INFERNO,
        Colormap::Magma => &MAGMA,
    };
    let t = t.clamp(0.0, 1.0);
    let n = stops.len() - 1;
    let f = t * n as f64;
    let i = (f.floor() as usize).min(n - 1);
    let frac = f - i as f64;
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * frac).round() as u8;
    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[i + 1];
    Color::Rgb(lerp(r0, r1), lerp(g0, g1), lerp(b0, b1))
}

/// average the slice over the fractional box [u0, u1) x [v0, v1) (unit slice
/// coordinates), NaN-aware; entirely outside or all-NaN -> NaN. when the slice is
/// FINER than the panel this is the panel-side block average: nearest sampling at
/// a 2:1 downsample drops half the slice pixels and turns a thin bright ring into
/// a dotted arc — the display-side twin of the producer's footprint supersampling.
fn box_avg(field: &FieldSlice, u0: f64, u1: f64, v0: f64, v1: f64) -> f64 {
    if u1 <= 0.0 || u0 >= 1.0 || v1 <= 0.0 || v0 >= 1.0 || field.data.is_empty() {
        return f64::NAN;
    }
    let fx0 = ((u0.max(0.0) * field.width as f64) as usize).min(field.width - 1);
    let fx1 = ((u1.min(1.0) * field.width as f64).ceil() as usize).clamp(fx0 + 1, field.width);
    let fy0 = ((v0.max(0.0) * field.height as f64) as usize).min(field.height - 1);
    let fy1 = ((v1.min(1.0) * field.height as f64).ceil() as usize).clamp(fy0 + 1, field.height);
    let (mut sum, mut cnt) = (0.0f64, 0u32);
    for fy in fy0..fy1 {
        for fx in fx0..fx1 {
            let v = field.data[fy * field.width + fx] as f64;
            if !v.is_nan() {
                sum += v;
                cnt += 1;
            }
        }
    }
    if cnt == 0 { f64::NAN } else { sum / cnt as f64 }
}

/// render the field as a truecolor half-block heatmap: each terminal cell is the
/// upper-half glyph `▀` whose fg is the top pixel and bg is the bottom pixel, so
/// one cell shows two stacked field samples (double vertical resolution). a thin
/// colorbar with min/max labels sits on the right.
/// display name of a colormap, for the hero title.
fn cmap_name(c: Colormap) -> &'static str {
    match c {
        Colormap::Viridis => "viridis",
        Colormap::Inferno => "inferno",
        Colormap::Magma => "magma",
    }
}

fn render_field(frame: &mut Frame, area: Rect, field: &FieldSlice) {
    // log10 colormap normalization needs a positive range; non-positive data falls
    // back to linear rather than lying with a clamped scale.
    let log_active = field.log_scale && field.vmax > 0.0;
    let block = Block::new()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(fg(BORDER_HERO))
        .title(Span::styled(
            format!(
                " {} · {}{} ",
                field.label,
                cmap_name(field.cmap),
                if log_active { " · log" } else { "" }
            ),
            fgb(GOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width < 6 || inner.height < 1 || field.data.is_empty() {
        return;
    }

    // a 1-row slice (1D run, or a 3D line-out) is a line profile, not a heatmap.
    if field.height <= 1 {
        render_field_line(frame, inner, field);
        return;
    }

    let bar_w: u16 = 7.min(inner.width / 4);
    let heat_w = inner.width - bar_w;
    let heat_h = inner.height;
    let range = (field.vmax - field.vmin).max(1e-30);
    // the log floor: vmin when positive, else a fixed dynamic range below vmax so a
    // zero-touching field (|B| with null points) still spans usable decades.
    let log_lo = if field.vmin > 0.0 { field.vmin } else { field.vmax * 1e-8 };
    let log_range = (field.vmax.ln() - log_lo.ln()).max(1e-30);
    let norm = move |v: f64| {
        if log_active {
            (v.max(log_lo).ln() - log_lo.ln()) / log_range
        } else {
            (v - field.vmin) / range
        }
    };

    // the target raster in half-pixel space (half-block halves are ~square). a
    // physical-shape slice letterboxes: fit the slice's aspect into the panel and
    // center it, so a circle renders circular; index-space slices stretch to fill.
    let (hp_w, hp_h) = (heat_w as f64, (heat_h * 2) as f64);
    let (tw, th, ox, oy) = if field.preserve_aspect {
        let scale = (hp_w / field.width as f64).min(hp_h / field.height as f64);
        let (tw, th) = (field.width as f64 * scale, field.height as f64 * scale);
        (tw, th, (hp_w - tw) / 2.0, (hp_h - th) / 2.0)
    } else {
        (hp_w, hp_h, 0.0, 0.0)
    };
    // sample the slice at a half-pixel, honoring the letterbox; outside -> NaN.
    let at = |px: u16, hy: u16| -> f64 {
        let u0 = (px as f64 - ox) / tw;
        let u1 = (px as f64 + 1.0 - ox) / tw;
        let v0 = (hy as f64 - oy) / th;
        let v1 = (hy as f64 + 1.0 - oy) / th;
        box_avg(field, u0, u1, v0, v1)
    };

    let buf = frame.buffer_mut();
    for cy in 0..heat_h {
        for cx in 0..heat_w {
            let top = at(cx, 2 * cy);
            let bot = at(cx, 2 * cy + 1);
            if let Some(cell) = buf.cell_mut((inner.x + cx, inner.y + cy)) {
                // NaN marks outside-domain pixels of a polar (physical-shape) slice:
                // draw only the valid half via the matching half-block, so the
                // annulus exterior and the excision hole stay blank.
                match (top.is_nan(), bot.is_nan()) {
                    (true, true) => {}
                    (false, true) => {
                        cell.set_symbol("\u{2580}") // upper half block
                            .set_fg(colormap(norm(top), field.cmap));
                    }
                    (true, false) => {
                        cell.set_symbol("\u{2584}") // lower half block
                            .set_fg(colormap(norm(bot), field.cmap));
                    }
                    (false, false) => {
                        cell.set_symbol("\u{2580}")
                            .set_fg(colormap(norm(top), field.cmap))
                            .set_bg(colormap(norm(bot), field.cmap));
                    }
                }
            }
        }
    }

    // colorbar: a one-column vertical gradient (high at top) + min/max labels.
    let bar_x = inner.x + heat_w + 1;
    for cy in 0..heat_h {
        let t_top = 1.0 - (2 * cy) as f64 / (2 * heat_h) as f64;
        let t_bot = 1.0 - (2 * cy + 1) as f64 / (2 * heat_h) as f64;
        if let Some(cell) = buf.cell_mut((bar_x, inner.y + cy)) {
            cell.set_symbol("\u{2580}")
                .set_fg(colormap(t_top, field.cmap))
                .set_bg(colormap(t_bot, field.cmap));
        }
    }
    buf.set_string(bar_x + 2, inner.y, humanize(field.vmax), fg(DIM));
    buf.set_string(bar_x + 2, inner.y + heat_h - 1, humanize(field.vmin), fg(DIM));
}

/// a 1D field as a line profile (value vs position). the trace takes a bright stop
/// of the active colormap, so the `c`-key recolors it too.
fn render_field_line(frame: &mut Frame, area: Rect, field: &FieldSlice) {
    if field.data.len() < 2 {
        return;
    }
    let points: Vec<(f64, f64)> = field
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();
    let x_max = (field.data.len() - 1) as f64;
    let dataset = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(colormap(0.72, field.cmap)))
        .data(&points);
    let chart = Chart::new(vec![dataset])
        .x_axis(Axis::default().style(fg(BORDER)).bounds([0.0, x_max]))
        .y_axis(
            Axis::default()
                .style(fg(BORDER))
                .bounds([field.vmin, field.vmax])
                .labels([
                    Span::styled(format!("{:.3}", field.vmin), fg(DIM)),
                    Span::styled(format!("{:.3}", field.vmax), fg(DIM)),
                ]),
        );
    frame.render_widget(chart, area);
}

fn render_cards(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let n_drift = view.mass_drift.is_some() as u16
        + view.energy_drift.is_some() as u16
        + view.div_b.is_some() as u16;
    let has_cons = n_drift > 0;
    let has_blocks = !view.blocks_per_level.is_empty();

    // build the right column from only the cards that have data, so an unwired
    // diagnostic leaves no empty box.
    let mut spec = Vec::new();
    if has_cons {
        spec.push(Constraint::Length(n_drift + 2));
    }
    spec.push(Constraint::Length(4)); // max W | dt history
    if has_blocks {
        spec.push(Constraint::Length(view.blocks_per_level.len() as u16 + 2));
    }
    spec.push(Constraint::Length(3)); // cfl
    let has_host = view.host.is_some();
    if has_host {
        spec.push(Constraint::Length(5)); // machine card
    }
    spec.push(Constraint::Min(0));
    let r = Layout::vertical(spec).split(area);

    let mut i = 0;
    if has_cons {
        render_conservation(frame, r[i], view);
        i += 1;
    }
    render_maxw_dt(frame, r[i], view);
    i += 1;
    if has_blocks {
        render_blocks(frame, r[i], view);
        i += 1;
    }
    render_cfl(frame, r[i], view);
    i += 1;
    if has_host {
        render_machine(frame, r[i], view);
    }
}

/// bytes as a compact binary-prefixed size (`3.2G`, `512M`, `48K`), matching how
/// ram + rss are read on a node.
fn fmt_bytes(b: u64) -> String {
    const G: f64 = 1_073_741_824.0;
    const M: f64 = 1_048_576.0;
    const K: f64 = 1024.0;
    let x = b as f64;
    if x >= G {
        format!("{:.1}G", x / G)
    } else if x >= M {
        format!("{:.0}M", x / M)
    } else if x >= K {
        format!("{:.0}K", x / K)
    } else {
        format!("{b}B")
    }
}

/// the machine card: compute-host name, core/thread count, and a gauge of this
/// run's resident memory against the node's physical ram (the oom watch). shown
/// only once host stats are sampled.
fn render_machine(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("machine");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let Some(h) = &view.host else {
        return;
    };
    if inner.height == 0 {
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(h.hostname.clone(), fgb(VALUE)))),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{} cores · {} threads", h.cpu_count, h.threads),
            fg(DIM),
        ))),
        rows[1],
    );

    // memory gauge: rss / total, amber past 90% (near-oom).
    let ratio = if h.mem_total > 0 {
        (h.mem_rss as f64 / h.mem_total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cols =
        Layout::horizontal([Constraint::Min(6), Constraint::Length(14)]).split(rows[2]);
    let color = if ratio >= 0.9 { AMBER } else { LAV };
    let gauge = LineGauge::default()
        .ratio(ratio)
        .filled_style(fg(color))
        .unfilled_style(fg(BORDER))
        .label(Span::styled("mem", fg(DIM)));
    frame.render_widget(gauge, cols[0]);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{}/{}", fmt_bytes(h.mem_rss), fmt_bytes(h.mem_total)),
            fgb(VALUE),
        )))
        .right_aligned(),
        cols[1],
    );
}

fn render_conservation(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("conservation & constraints");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let n = view.mass_drift.is_some() as usize
        + view.energy_drift.is_some() as usize
        + view.div_b.is_some() as usize;
    if n == 0 {
        return;
    }
    let spec: Vec<Constraint> = (0..n).map(|_| Constraint::Length(1)).collect();
    let rows = Layout::vertical(spec).split(inner);
    let mut i = 0;
    if let Some(h) = &view.mass_drift {
        drift_row(frame, rows[i], "mass", TEAL, h);
        i += 1;
    }
    if let Some(h) = &view.energy_drift {
        drift_row(frame, rows[i], "energy", GREEN, h);
        i += 1;
    }
    if let Some(h) = &view.div_b {
        drift_row(frame, rows[i], "div·B", BLUE, h);
    }
}

fn drift_row(frame: &mut Frame, area: Rect, label: &str, color: Color, hist: &[f64]) {
    let cols = Layout::horizontal([
        Constraint::Length(8),
        Constraint::Min(4),
        Constraint::Length(10),
    ])
    .split(area);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(label.to_string(), fg(DIM)))),
        cols[0],
    );
    frame.render_widget(Sparkline::default().data(spark(hist)).style(fg(color)), cols[1]);
    let val = hist.last().copied().unwrap_or(0.0);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!("{val:.1e}"), fgb(VALUE)))).right_aligned(),
        cols[2],
    );
}

fn render_maxw_dt(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    // the max-W card applies only to relativistic regimes (the Lorentz factor is
    // undefined for a non-relativistic gas, so `max_w` is None there). when absent,
    // dt history takes the full row rather than a "—" placeholder holding half of it.
    let dt_area = if let Some(w) = view.max_w {
        let cols =
            Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(area);
        let w_block = card("max W");
        let w_inner = w_block.inner(cols[0]);
        frame.render_widget(w_block, cols[0]);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(format!("{w:.2}"), fgb(VALUE)))),
            w_inner,
        );
        cols[1]
    } else {
        area
    };

    let dt_block = card("dt history");
    let dt_inner = dt_block.inner(dt_area);
    frame.render_widget(dt_block, dt_area);
    frame.render_widget(
        Sparkline::default().data(spark(&view.dt_hist)).style(fg(BLUE)),
        dt_inner,
    );
}

fn render_blocks(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("zones / level");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let n = view.blocks_per_level.len();
    if inner.height == 0 || n == 0 {
        return;
    }
    let peak = view.blocks_per_level.iter().copied().max().unwrap_or(1).max(1) as f64;
    let spec: Vec<Constraint> = (0..n).map(|_| Constraint::Length(1)).collect();
    let rows = Layout::vertical(spec).split(inner);
    for lvl in 0..n {
        let cols = Layout::horizontal([Constraint::Min(6), Constraint::Length(6)]).split(rows[lvl]);
        let gauge = LineGauge::default()
            .ratio((view.blocks_per_level[lvl] as f64 / peak).clamp(0.0, 1.0))
            .filled_style(fg(GREEN))
            .unfilled_style(fg(BORDER))
            .label(Span::styled(format!("L{lvl}"), fg(DIM)));
        frame.render_widget(gauge, cols[0]);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("{}", view.blocks_per_level[lvl]),
                fgb(VALUE),
            )))
            .right_aligned(),
            cols[1],
        );
    }
}

fn render_cfl(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("CFL");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let ratio = if view.cfl_max > 0.0 {
        (view.cfl / view.cfl_max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gauge = LineGauge::default()
        .ratio(ratio)
        .filled_style(fg(TEAL))
        .unfilled_style(fg(BORDER))
        .label(Span::styled(
            format!("{:.2} / {:.2}", view.cfl, view.cfl_max),
            fgb(VALUE),
        ));
    frame.render_widget(gauge, inner);
}

fn render_log(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("LOG");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 {
        return;
    }
    let take = inner.height as usize;
    let start = view.log.len().saturating_sub(take);
    let lines: Vec<Line> = view.log[start..]
        .iter()
        .map(|(ts, text)| {
            Line::from(vec![
                Span::styled(format!("[{ts}] "), fg(DIM)),
                Span::styled(text.clone(), fg(TEXT)),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

// -- other tabs (sketched) ----------------------------------------------------

fn render_diagnostics(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(area);
    render_throughput_hero(frame, rows[0], view);
    let cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[1]);
    render_conservation(frame, cols[0], view);
    render_maxw_dt(frame, cols[1], view);
}

fn render_grid(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let rows = Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).split(area);
    render_blocks(frame, rows[0], view);
    let block = card("regrid events");
    let inner = block.inner(rows[1]);
    frame.render_widget(block, rows[1]);
    let lines: Vec<Line> = view
        .log
        .iter()
        .rev()
        .filter(|(_, t)| t.contains("regrid"))
        .take(inner.height as usize)
        .map(|(ts, text)| {
            Line::from(vec![
                Span::styled(format!("[{ts}] "), fg(DIM)),
                Span::styled(text.clone(), fg(TEXT)),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_config(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    let block = card("config");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    // render each section under a full-width divider header (`SECTION ────────`), blank line between.
    // sections are taken in first-seen order, CASE-INSENSITIVELY merged — so a config's `physics`
    // group folds into the core `Physics` section instead of printing a duplicate header. rows within
    // a section keep input order.
    let width = inner.width as usize;
    let mut sections: Vec<&str> = Vec::new();
    for (s, _, _) in &view.config {
        if !sections.iter().any(|x| x.eq_ignore_ascii_case(s)) {
            sections.push(s.as_str());
        }
    }
    // label column sized to the LONGEST label panel-wide (values stay aligned
    // across sections), floored so short-label configs keep the classic look
    // and capped to leave the value at least a third of the panel. the
    // explicit two-space gap survives even at the cap, so a long label can
    // never run into its value.
    let label_w = view
        .config
        .iter()
        .map(|(_, k, _)| k.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(16, width.saturating_mul(2) / 3);
    let mut lines: Vec<Line> = Vec::new();
    for (i, sec) in sections.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        let head = format!("{} ", sec.to_uppercase());
        let rule = "\u{2500}".repeat(width.saturating_sub(head.chars().count()));
        lines.push(Line::from(Span::styled(format!("{head}{rule}"), fgb(TEAL))));
        for (s, k, v) in view.config.iter().filter(|(s, _, _)| s.eq_ignore_ascii_case(sec)) {
            lines.push(Line::from(vec![
                Span::styled(format!("  {k:<label_w$}  "), fg(DIM)),
                Span::styled(v.clone(), fgb(VALUE)),
            ]));
            let _ = s;
        }
    }
    // clamp the scroll to the actual overflow, then apply it (the up/down arrows drive
    // `config_scroll`); a panel that fits stays put.
    let max_scroll = (lines.len() as u16).saturating_sub(inner.height);
    let offset = view.config_scroll.min(max_scroll);
    frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample() -> DiagnosticView {
        DiagnosticView {
            app_title: "hydroflux — kelvin_helmholtz.toml".into(),
            regime: "RHD".into(),
            attached: "attached · rank 0 / 1".into(),
            paused: false,
            frame: 3,
            t: 1.8423,
            step: 24_110,
            dt: 3.1e-4,
            wall_secs: 252.0,
            throughput_mzcups: 148.0,
            progress: 42,
            tab: 0,
            config_scroll: 0,
            throughput_hist: vec![100.0, 120.0, 148.0, 150.0, 145.0, 148.0],
            dt_hist: vec![3.0e-4, 3.1e-4, 3.05e-4, 3.1e-4],
            mass_drift: Some(vec![2.0e-13, 2.4e-13, 2.2e-13]),
            energy_drift: Some(vec![7.0e-9, 8.1e-9, 8.0e-9]),
            div_b: Some(vec![4.0e-16, 4.6e-16, 4.5e-16]),
            max_w: Some(4.82),
            cfl: 0.40,
            cfl_max: 0.80,
            blocks_per_level: vec![64, 210, 96],
            log: vec![
                ("00:28".into(), "regrid → level 1 : +4 blocks".into()),
                ("00:30".into(), "checkpoint written : chk.h5".into()),
            ],
            config: vec![
                ("Physics".into(), "regime".into(), "rhd".into()),
                ("Geometry".into(), "resolution".into(), "256 x 256  (65536 zones)".into()),
            ],
            field: None,
            field_count: 1,
            host: None,
        }
    }

    fn dump(view: &DiagnosticView) -> String {
        let backend = TestBackend::new(140, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, view)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn overview_renders_shell_and_cards() {
        let d = dump(&sample());
        assert!(d.contains("RHD"));
        assert!(d.contains("overview"));
        assert!(d.contains("THROUGHPUT"));
        assert!(d.contains("conservation & constraints"));
        assert!(d.contains("div·B"));
        assert!(d.contains("zones / level"));
        assert!(d.contains("CFL"));
    }

    /// a run with no mhd / rhd data omits the div·B row and shows max W as a dash.
    #[test]
    fn per_physics_fields_are_optional() {
        let mut v = sample();
        v.div_b = None;
        v.max_w = None;
        let d = dump(&v);
        assert!(!d.contains("div·B"));
        assert!(d.contains("conservation & constraints"));
        // max W is relativistic-only: OMITTED (not a "—" placeholder) when max_w is
        // None, so dt history takes the full row instead of squatting on half of it.
        assert!(!d.contains("max W"));
        assert!(d.contains("dt history"));
        // present again once the run is relativistic.
        v.max_w = Some(3.2);
        assert!(dump(&v).contains("max W"));
    }

    /// each tab renders without panicking.
    #[test]
    fn every_tab_renders() {
        for tab in 0..tab_names(true).len() {
            let mut v = sample();
            v.tab = tab;
            let _ = dump(&v);
        }
    }

    /// the overview hero renders a field heatmap without panicking when a slice is
    /// present, and carries the field label in the panel title.
    #[test]
    fn thin_ring_survives_panel_downsampling() {
        // a one-pixel-wide bright horizontal line in a 200x200 slice, box-averaged
        // down to a ~90-column panel: EVERY column's covering half-pixel must blend
        // the line in (nearest sampling would drop it on ~half the columns).
        let (w, h) = (200usize, 200usize);
        let mut data = vec![1.0f32; w * h];
        for i in 0..w {
            data[100 * w + i] = 10.0;
        }
        let field = FieldSlice {
            label: "ring".into(),
            width: w,
            height: h,
            data,
            vmin: 1.0,
            vmax: 10.0,
            cmap: Colormap::Inferno,
            preserve_aspect: true,
            log_scale: false,
        };
        // a 90x45 panel = 90x90 half-pixels; the line at v = 0.5 falls in half-row 45.
        let (pw, phh) = (90usize, 90usize);
        for px in 0..pw {
            let u0 = px as f64 / pw as f64;
            let u1 = (px + 1) as f64 / pw as f64;
            let mut best = f64::NAN;
            for hy in 43..48 {
                let v0 = hy as f64 / phh as f64;
                let v1 = (hy + 1) as f64 / phh as f64;
                let v = super::box_avg(&field, u0, u1, v0, v1);
                if !v.is_nan() && (best.is_nan() || v > best) {
                    best = v;
                }
            }
            assert!(best > 2.0, "column {px} lost the ring: best {best}");
        }
    }

    #[test]
    fn field_heatmap_renders() {
        let backend = TestBackend::new(140, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut v = sample();
        let (w, h) = (32usize, 24usize);
        let data: Vec<f32> = (0..w * h).map(|k| (k % 7) as f32).collect();
        v.field = Some(FieldSlice {
            label: "density · inferno · slice z=0".into(),
            width: w,
            height: h,
            data,
            vmin: 0.0,
            vmax: 6.0,
            preserve_aspect: false,
            log_scale: false,
            cmap: Colormap::Inferno,
        });
        terminal.draw(|frame| render(frame, &v)).unwrap();
        let dump: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(dump.contains("density"));
        assert!(dump.contains('\u{2580}')); // half-block heatmap glyphs drawn
    }

    #[test]
    fn colormap_endpoints_and_clamp() {
        // clamps out-of-range and hits the LUT endpoints.
        assert_eq!(colormap(-1.0, Colormap::Viridis), Color::Rgb(68, 1, 84));
        assert_eq!(colormap(2.0, Colormap::Inferno), Color::Rgb(252, 255, 164));
    }

    /// a uniform-grid run (one level) drops the amr-only grid tab.
    #[test]
    fn uniform_grid_hides_grid_tab() {
        assert!(tab_names(true).contains(&"grid"));
        assert!(!tab_names(false).contains(&"grid"));
        // one level -> uniform -> renders without panic and still shows the shell.
        let mut v = sample();
        v.blocks_per_level = vec![65536];
        let d = dump(&v);
        assert!(d.contains("overview"));
        assert!(d.contains("config"));
    }
}
