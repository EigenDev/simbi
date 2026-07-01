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
pub struct DiagnosticView {
    // header
    pub app_title: String, // "hydroflux — kelvin_helmholtz.toml"
    pub regime: String,    // stat-strip badge, e.g. "SRHD"
    pub attached: String,  // "attached · rank 0 / 1" (or empty)
    pub paused: bool,
    pub frame: u64, // drives the spinner animation
    // stat strip
    pub t: f64,
    pub step: u64,
    pub dt: f64,
    pub wall_secs: f64,
    pub throughput_mzcups: f64,
    // active tab
    pub tab: usize,
    // charts
    pub throughput_hist: Vec<f64>,
    pub dt_hist: Vec<f64>,
    // conservation & constraints — each history is present only once the solver
    // reduces it, so the whole card (and each row) appears only when it has data.
    pub mass_drift: Option<Vec<f64>>,
    pub energy_drift: Option<Vec<f64>>,
    pub div_b: Option<Vec<f64>>, // mhd only
    pub max_w: Option<f64>,      // srhd / rmhd only
    pub cfl: f64,
    pub cfl_max: f64,
    pub blocks_per_level: Vec<u64>,
    // panels
    pub log: Vec<(String, String)>,    // (timestamp, text)
    pub config: Vec<(String, String)>, // config-tab rows
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
    render_footer(frame, v[4]);
}

fn render_header(frame: &mut Frame, area: Rect, view: &DiagnosticView) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(view.app_title.clone(), fgb(LAV)))),
        area,
    );
    if !view.attached.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(view.attached.clone(), fg(DIM)))).right_aligned(),
            area,
        );
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

fn render_footer(frame: &mut Frame, area: Rect) {
    let keys = [
        ("space", "pause"),
        ("s", "step"),
        ("tab", "switch"),
        ("w", "checkpoint"),
        ("q", "quit"),
    ];
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
    render_throughput_hero(frame, cols[0], view);
    render_cards(frame, cols[1], view);
    render_log(frame, rows[1], view);
}

/// the hero panel — reserved for the live field view (tier 2); for now it holds
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
    let cols = Layout::horizontal([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)]).split(area);

    let w_block = card("max W");
    let w_inner = w_block.inner(cols[0]);
    frame.render_widget(w_block, cols[0]);
    let w_str = view
        .max_w
        .map(|w| format!("{w:.2}"))
        .unwrap_or_else(|| "—".to_string());
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(w_str, fgb(VALUE)))),
        w_inner,
    );

    let dt_block = card("dt history");
    let dt_inner = dt_block.inner(cols[1]);
    frame.render_widget(dt_block, cols[1]);
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
    let lines: Vec<Line> = view
        .config
        .iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(format!("{k:<16}"), fg(DIM)),
                Span::styled(v.clone(), fgb(VALUE)),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample() -> DiagnosticView {
        DiagnosticView {
            app_title: "hydroflux — kelvin_helmholtz.toml".into(),
            regime: "SRHD".into(),
            attached: "attached · rank 0 / 1".into(),
            paused: false,
            frame: 3,
            t: 1.8423,
            step: 24_110,
            dt: 3.1e-4,
            wall_secs: 252.0,
            throughput_mzcups: 148.0,
            tab: 0,
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
                ("regime".into(), "srhd".into()),
                ("resolution".into(), "256 x 256  (65536 zones)".into()),
            ],
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
        assert!(d.contains("SRHD"));
        assert!(d.contains("overview"));
        assert!(d.contains("THROUGHPUT"));
        assert!(d.contains("conservation & constraints"));
        assert!(d.contains("div·B"));
        assert!(d.contains("zones / level"));
        assert!(d.contains("CFL"));
    }

    /// a run with no mhd / srhd data omits the div·B row and shows max W as a dash.
    #[test]
    fn per_physics_fields_are_optional() {
        let mut v = sample();
        v.div_b = None;
        v.max_w = None;
        let d = dump(&v);
        assert!(!d.contains("div·B"));
        assert!(d.contains("conservation & constraints"));
        assert!(d.contains("max W"));
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
