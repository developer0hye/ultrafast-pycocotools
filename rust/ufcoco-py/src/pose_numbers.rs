//! Flat coordinate decoding for JSON spans already validated by serde_json.
//!
//! Keep the general serde visitor as a fallback. This path only recognizes
//! finite numbers and booleans, and writes directly into the final x/y slice.

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
