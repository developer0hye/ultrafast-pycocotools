//! Flat coordinate decoding for JSON spans already validated by serde_json.
//!
//! Keep the general serde visitor as a fallback. These paths only recognize
//! finite numbers and booleans; anything else returns `false`/`None` so the
//! caller re-parses the span with serde and reports serde's error.

pub(crate) fn decode(raw: &[u8], destination: &mut [f64], joints: usize) -> bool {
    if !destination.is_empty() && destination.len() != joints * 2 {
        return false;
    }
    let Ok(text) = std::str::from_utf8(raw) else {
        return false;
    };
    let Some(inner) = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return false;
    };
    let mut component = 0;
    for token in inner.split(',') {
        let token = token.trim_matches(|c| matches!(c, ' ' | '\n' | '\r' | '\t'));
        if component >= joints * 3 {
            return false;
        }
        let axis = component % 3;
        let needed = axis < 2 && !destination.is_empty();
        let value = match token {
            "true" => 1.0,
            "false" => 0.0,
            // The enclosing span has already passed strict JSON syntax
            // validation. A plain decimal token of at most 300 bytes has
            // magnitude < 10^300, hence cannot overflow f64. For unused
            // coordinates/visibility, numeric type + this bound is sufficient;
            // exponent forms and longer tokens still undergo real conversion.
            _ if !needed
                && token.len() <= 300
                && matches!(token.as_bytes().first(), Some(b'-' | b'0'..=b'9'))
                && !token.as_bytes().iter().any(|&b| b == b'e' || b == b'E') =>
            {
                0.0
            }
            _ => match token.parse::<f64>() {
                Ok(value) if value.is_finite() => value,
                _ => return false,
            },
        };
        if needed {
            destination[component / 3 * 2 + axis] = value;
        }
        component += 1;
    }
    component == joints * 3
}

/// Polygon rings `[[x, y, ...], ...]` into flat coordinates and ring ends,
/// with the same values as the serde `Coordinate` visitor. `None` means
/// "not a plain polygon list" and leaves the decision to serde.
pub(crate) fn decode_polygons(raw: &[u8]) -> Option<(Vec<f64>, Vec<u32>)> {
    const WHITESPACE: [char; 4] = [' ', '\n', '\r', '\t'];
    let text = std::str::from_utf8(raw).ok()?.trim_matches(WHITESPACE);
    let mut rest = text
        .strip_prefix('[')?
        .strip_suffix(']')?
        .trim_start_matches(WHITESPACE);
    let mut coords = Vec::new();
    let mut ends = Vec::new();
    while !rest.is_empty() {
        rest = rest.strip_prefix('[')?;
        // A ring of numbers contains no bracket; anything nested fails below.
        let close = rest.find(']')?;
        let ring = rest[..close].trim_matches(WHITESPACE);
        if !ring.is_empty() {
            for token in ring.split(',') {
                coords.push(number(token.trim_matches(WHITESPACE))?);
            }
        }
        ends.push(u32::try_from(coords.len()).ok()?);
        rest = rest[close + 1..].trim_start_matches(WHITESPACE);
        if let Some(next) = rest.strip_prefix(',') {
            rest = next.trim_start_matches(WHITESPACE);
            if rest.is_empty() {
                return None;
            }
        } else if !rest.is_empty() {
            return None;
        }
    }
    Some((coords, ends))
}

/// One JSON number or boolean, as the serde coordinate visitors convert it.
fn number(token: &str) -> Option<f64> {
    match token {
        "true" => Some(1.0),
        "false" => Some(0.0),
        _ if matches!(token.as_bytes().first(), Some(b'-' | b'0'..=b'9')) => {
            token.parse::<f64>().ok().filter(|value| value.is_finite())
        }
        _ => None,
    }
}
