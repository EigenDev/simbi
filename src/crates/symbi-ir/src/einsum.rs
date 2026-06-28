// =============================================================================
// einsum.rs
//
// einsum spec parser. NumPy-style string ("ij,jk->ik", "...ij,...jk->...ik",
// "i,i->", "ii->", etc.) -> EinsumSpec AST. the validator + builder in
// graph.rs (R.2.e) consume the AST to drive shape inference + variance
// pairing.
//
// grammar (spec § 4.1):
//   spec        := inputs "->" output
//   inputs      := label_list ("," label_list)*
//   output      := label_list
//   label_list  := atom*
//   atom        := label | ellipsis
//   label       := [a-zA-Z]
//   ellipsis    := "..."
//
// whitespace is ignored. labels are case-sensitive. max 8 distinct
// named labels per spec (V1 cap). max one ellipsis per side
// (input-spec or output-spec). non-alpha / non-"..." / non-"," / non-
// whitespace chars are rejected.
// =============================================================================

/// max distinct named labels per spec (§ 4.1; physics rarely exceeds 8).
pub const MAX_LABELS_PER_SPEC: usize = 8;

/// one atom in an input or output spec: a named label or the batch
/// ellipsis. labels are single characters preserving case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    Label(char),
    Ellipsis,
}

/// one side of the spec: an ordered list of atoms. corresponds to the
/// rank of one input (or the output) after ellipsis is unrolled.
pub type AtomList = Vec<Atom>;

/// parsed einsum spec. ready for the validator to walk.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinsumSpec {
    pub inputs: Vec<AtomList>,
    pub output: AtomList,
}

/// parse-time error from `parse_einsum_spec`. these get converted to
/// `ShapeError` variants by the einsum builder at IR build time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EinsumParseError {
    /// no "->" in the spec.
    MissingArrow,
    /// more than one "->" in the spec.
    MultipleArrows,
    /// a side has more than one "..." atom.
    MultipleEllipses { side_index: SideIndex },
    /// invalid character (not alpha, not part of "...", not "," or "->").
    BadCharacter { ch: char, pos: usize },
    /// dot-run that wasn't exactly 3 dots (e.g., "..", "....").
    InvalidEllipsis { pos: usize, run_len: usize },
    /// spec uses more than MAX_LABELS_PER_SPEC distinct named labels.
    LabelLimitExceeded { count: usize, max: usize },
}

/// identifies which side of the spec produced an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideIndex {
    /// 0-based index into the inputs list.
    Input(usize),
    /// the output side.
    Output,
}

impl SideIndex {
    pub fn label(self) -> String {
        match self {
            SideIndex::Input(i) => format!("input {}", i),
            SideIndex::Output => "output".to_string(),
        }
    }
}

impl std::fmt::Display for EinsumParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EinsumParseError::MissingArrow => write!(f, "einsum spec missing '->'"),
            EinsumParseError::MultipleArrows => write!(f, "einsum spec has more than one '->'"),
            EinsumParseError::MultipleEllipses { side_index } => {
                write!(
                    f,
                    "einsum spec {} has more than one '...'",
                    side_index.label()
                )
            }
            EinsumParseError::BadCharacter { ch, pos } => {
                write!(
                    f,
                    "einsum spec: unexpected character '{}' at position {}",
                    ch, pos
                )
            }
            EinsumParseError::InvalidEllipsis { pos, run_len } => write!(
                f,
                "einsum spec: dot-run of length {} at position {} is not a valid ellipsis (must be exactly 3)",
                run_len, pos
            ),
            EinsumParseError::LabelLimitExceeded { count, max } => write!(
                f,
                "einsum spec uses {} distinct named labels; V1 limit is {}",
                count, max
            ),
        }
    }
}

/// parse an einsum spec string into an `EinsumSpec` AST.
pub fn parse_einsum_spec(s: &str) -> Result<EinsumSpec, EinsumParseError> {
    // split on '->'; reject 0 or > 1 occurrences.
    let parts: Vec<&str> = s.split("->").collect();
    if parts.len() < 2 {
        return Err(EinsumParseError::MissingArrow);
    }
    if parts.len() > 2 {
        return Err(EinsumParseError::MultipleArrows);
    }
    let (input_side, output_side) = (parts[0], parts[1]);

    // input side: comma-separated atom lists.
    let mut inputs: Vec<AtomList> = Vec::new();
    for (i, segment) in input_side.split(',').enumerate() {
        let atoms = parse_atom_list(segment, SideIndex::Input(i))?;
        inputs.push(atoms);
    }
    let output = parse_atom_list(output_side, SideIndex::Output)?;

    // V1 label-count cap.
    let mut seen: Vec<char> = Vec::new();
    for list in inputs.iter().chain(std::iter::once(&output)) {
        for a in list {
            if let Atom::Label(c) = a
                && !seen.contains(c)
            {
                seen.push(*c);
            }
        }
    }
    if seen.len() > MAX_LABELS_PER_SPEC {
        return Err(EinsumParseError::LabelLimitExceeded {
            count: seen.len(),
            max: MAX_LABELS_PER_SPEC,
        });
    }

    Ok(EinsumSpec { inputs, output })
}

/// parse a single side (atoms only, no commas, no arrow). validates at
/// most one ellipsis. whitespace ignored.
fn parse_atom_list(segment: &str, side: SideIndex) -> Result<AtomList, EinsumParseError> {
    let mut out: AtomList = Vec::new();
    let mut ellipsis_count = 0;
    let bytes = segment.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '.' {
            // count the dot-run starting here.
            let start = i;
            let mut run = 0;
            while i < bytes.len() && bytes[i] == b'.' {
                run += 1;
                i += 1;
            }
            if run != 3 {
                return Err(EinsumParseError::InvalidEllipsis {
                    pos: start,
                    run_len: run,
                });
            }
            ellipsis_count += 1;
            if ellipsis_count > 1 {
                return Err(EinsumParseError::MultipleEllipses { side_index: side });
            }
            out.push(Atom::Ellipsis);
            continue;
        }
        if c.is_ascii_alphabetic() {
            out.push(Atom::Label(c));
            i += 1;
            continue;
        }
        return Err(EinsumParseError::BadCharacter { ch: c, pos: i });
    }
    Ok(out)
}

// ----- tests -----

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> EinsumSpec {
        parse_einsum_spec(s).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", s, e))
    }

    // ---- happy paths ----

    #[test]
    fn dot_product_spec() {
        let r = parse("i,i->");
        assert_eq!(r.inputs.len(), 2);
        assert_eq!(r.inputs[0], vec![Atom::Label('i')]);
        assert_eq!(r.inputs[1], vec![Atom::Label('i')]);
        assert!(r.output.is_empty());
    }

    #[test]
    fn matmul_spec() {
        let r = parse("ij,jk->ik");
        assert_eq!(r.inputs.len(), 2);
        assert_eq!(r.inputs[0], vec![Atom::Label('i'), Atom::Label('j')]);
        assert_eq!(r.inputs[1], vec![Atom::Label('j'), Atom::Label('k')]);
        assert_eq!(r.output, vec![Atom::Label('i'), Atom::Label('k')]);
    }

    #[test]
    fn trace_spec() {
        let r = parse("ii->");
        assert_eq!(r.inputs[0], vec![Atom::Label('i'), Atom::Label('i')]);
        assert!(r.output.is_empty());
    }

    #[test]
    fn matrix_vector_spec() {
        let r = parse("ij,j->i");
        assert_eq!(r.inputs[0], vec![Atom::Label('i'), Atom::Label('j')]);
        assert_eq!(r.inputs[1], vec![Atom::Label('j')]);
        assert_eq!(r.output, vec![Atom::Label('i')]);
    }

    #[test]
    fn bilinear_form_spec() {
        let r = parse("ij,i,j->");
        assert_eq!(r.inputs.len(), 3);
        assert!(r.output.is_empty());
    }

    #[test]
    fn outer_product_spec() {
        let r = parse("i,j->ij");
        assert_eq!(r.output, vec![Atom::Label('i'), Atom::Label('j')]);
    }

    #[test]
    fn batched_matmul_with_ellipsis() {
        let r = parse("...ij,...jk->...ik");
        assert_eq!(
            r.inputs[0],
            vec![Atom::Ellipsis, Atom::Label('i'), Atom::Label('j')]
        );
        assert_eq!(
            r.inputs[1],
            vec![Atom::Ellipsis, Atom::Label('j'), Atom::Label('k')]
        );
        assert_eq!(
            r.output,
            vec![Atom::Ellipsis, Atom::Label('i'), Atom::Label('k')]
        );
    }

    #[test]
    fn batched_dot_with_ellipsis() {
        let r = parse("...i,...i->...");
        assert_eq!(r.inputs[0], vec![Atom::Ellipsis, Atom::Label('i')]);
        assert_eq!(r.inputs[1], vec![Atom::Ellipsis, Atom::Label('i')]);
        assert_eq!(r.output, vec![Atom::Ellipsis]);
    }

    #[test]
    fn case_sensitive_labels() {
        let r = parse("Ij,jK->IK");
        assert_eq!(r.inputs[0], vec![Atom::Label('I'), Atom::Label('j')]);
        assert_eq!(r.inputs[1], vec![Atom::Label('j'), Atom::Label('K')]);
        assert_eq!(r.output, vec![Atom::Label('I'), Atom::Label('K')]);
    }

    #[test]
    fn whitespace_is_ignored() {
        let r = parse(" i j , j k -> i k ");
        assert_eq!(r.inputs[0], vec![Atom::Label('i'), Atom::Label('j')]);
        assert_eq!(r.inputs[1], vec![Atom::Label('j'), Atom::Label('k')]);
        assert_eq!(r.output, vec![Atom::Label('i'), Atom::Label('k')]);
    }

    #[test]
    fn empty_output_keeps_inputs() {
        // sum-reduce-all: "...i->..." was tested; here's "ij->"
        let r = parse("ij->");
        assert_eq!(r.inputs[0].len(), 2);
        assert!(r.output.is_empty());
    }

    // ---- errors ----

    #[test]
    fn missing_arrow_errors() {
        assert_eq!(parse_einsum_spec("ij"), Err(EinsumParseError::MissingArrow));
    }

    #[test]
    fn multiple_arrows_errors() {
        assert_eq!(
            parse_einsum_spec("ij->k->"),
            Err(EinsumParseError::MultipleArrows)
        );
    }

    #[test]
    fn multiple_ellipses_errors() {
        let r = parse_einsum_spec("...i...,j->ij");
        match r {
            Err(EinsumParseError::MultipleEllipses { side_index }) => {
                assert_eq!(side_index, SideIndex::Input(0));
            }
            other => panic!("expected MultipleEllipses, got {:?}", other),
        }
    }

    #[test]
    fn invalid_ellipsis_two_dots() {
        let r = parse_einsum_spec("..ij->ij");
        match r {
            Err(EinsumParseError::InvalidEllipsis { run_len, .. }) => assert_eq!(run_len, 2),
            other => panic!("expected InvalidEllipsis(2), got {:?}", other),
        }
    }

    #[test]
    fn invalid_ellipsis_four_dots() {
        let r = parse_einsum_spec("....ij->ij");
        match r {
            Err(EinsumParseError::InvalidEllipsis { run_len, .. }) => assert_eq!(run_len, 4),
            other => panic!("expected InvalidEllipsis(4), got {:?}", other),
        }
    }

    #[test]
    fn bad_character_errors() {
        // digit isn't a valid label
        let r = parse_einsum_spec("i1->i");
        match r {
            Err(EinsumParseError::BadCharacter { ch, .. }) => assert_eq!(ch, '1'),
            other => panic!("expected BadCharacter, got {:?}", other),
        }
    }

    #[test]
    fn label_limit_enforced() {
        // 9 distinct labels exceeds V1 cap of 8.
        let r = parse_einsum_spec("abcdefghi->");
        match r {
            Err(EinsumParseError::LabelLimitExceeded { count, max }) => {
                assert_eq!(count, 9);
                assert_eq!(max, MAX_LABELS_PER_SPEC);
            }
            other => panic!("expected LabelLimitExceeded, got {:?}", other),
        }
    }

    #[test]
    fn label_limit_with_8_allowed() {
        let r = parse_einsum_spec("abcdefgh->");
        assert!(r.is_ok());
    }

    #[test]
    fn ellipsis_does_not_count_toward_label_limit() {
        // 8 named labels + ellipsis is still OK.
        let r = parse_einsum_spec("...abcdefgh->...");
        assert!(r.is_ok(), "got {:?}", r);
    }

    #[test]
    fn side_index_label_strings() {
        assert_eq!(SideIndex::Input(2).label(), "input 2");
        assert_eq!(SideIndex::Output.label(), "output");
    }

    #[test]
    fn error_display_contains_useful_info() {
        let err = EinsumParseError::LabelLimitExceeded {
            count: 10,
            max: MAX_LABELS_PER_SPEC,
        };
        let s = format!("{}", err);
        assert!(s.contains("10"), "{}", s);
        assert!(s.contains("8"), "{}", s);
    }
}
