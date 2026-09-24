// The decoder is pure Rust. Exercise it without loading or exporting Python APIs.
#[path = "../src/pose_numbers.rs"]
mod pose_numbers;

fn check(token: &str) {
    let raw = format!("[ {token},\t{token},\ntrue ]");
    let reference: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
    let expected = reference[0].as_f64().unwrap();
    let mut actual = [0.0; 2];
    assert!(
        pose_numbers::decode(raw.as_bytes(), &mut actual, 1),
        "{token}"
    );
    assert_eq!(actual[0].to_bits(), expected.to_bits(), "{token}");
    assert_eq!(actual[1].to_bits(), expected.to_bits(), "{token}");
    assert!(
        pose_numbers::decode(raw.as_bytes(), &mut [], 1),
        "discarded {token}"
    );
}

#[test]
fn decimal_boundaries_and_integer_casts_match_serde_bits() {
    for token in [
        "0",
        "-0",
        "0.0",
        "-0.0",
        "-0e0",
        "32002.703",
        "0.1",
        "9007199254740991",
        "9007199254740993",
        "-9007199254740993",
        "9223372036854775807",
        "-9223372036854775808",
        "-9223372036854775809",
        "18446744073709551615",
        "18446744073709551616",
        "1.7976931348623157e308",
        "-1.7976931348623157e308",
        "5e-324",
        "2.2250738585072014e-308",
        "2.2250738585072012e-308",
        "1e-400",
        "-1e-400",
        "0.1000000000000000055511151231257827021181583404541015625",
        "100000000000000000000000000000000000000000000000001",
    ] {
        check(token);
    }
}

#[test]
fn seeded_float_and_decimal_corpus_matches_serde_bits() {
    let mut state = 0x123456789abcdef0u64;
    for _ in 0..100_000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = f64::from_bits(state);
        if value.is_finite() {
            check(&format!("{value:?}"));
        }
        // Also test decimal strings that were not produced by formatting an f64.
        let coefficient = state % 1_000_000_000_000_000_000;
        let exponent = (state >> 48) as i32 % 600 - 320;
        check(&format!("{coefficient}e{exponent}"));
    }
}

#[test]
fn booleans_visibility_and_fallbacks() {
    let mut xy = [0.0; 4];
    assert!(pose_numbers::decode(
        b"[true,false,0.1,1,2,false]",
        &mut xy,
        2
    ));
    assert_eq!(xy, [1.0, 0.0, 1.0, 2.0]);
    for raw in [
        b"[1,2]".as_slice(),
        b"[1,2,3,4]",
        b"[1,2,null]",
        b"[1,2,\"3\"]",
        b"[1,2,[]]",
        b"[1,2,{}]",
        b"[1,2,1e400]",
        b"null",
        b"[]",
    ] {
        assert!(!pose_numbers::decode(raw, &mut [0.0; 2], 1), "{raw:?}");
        assert!(!pose_numbers::decode(raw, &mut [], 1), "discarded {raw:?}");
    }
}

#[test]
fn unused_plain_decimals_are_bounded_and_still_type_checked() {
    for width in [299, 300, 301, 308, 309, 400] {
        let token = "9".repeat(width);
        let raw = format!("[1,2,{token}]");
        let valid = serde_json::from_str::<serde_json::Value>(&raw).is_ok();
        assert_eq!(pose_numbers::decode(raw.as_bytes(), &mut [], 1), valid);
        assert_eq!(
            pose_numbers::decode(raw.as_bytes(), &mut [0.0; 2], 1),
            valid
        );
    }
    for raw in [
        b"[1,2,1e+309]".as_slice(),
        b"[1,2,-1E309]",
        b"[1,2,\"3,4\"]",
        b"[1,[2,3],4]",
        b"[1,2,{\"3\":4}]",
    ] {
        assert!(!pose_numbers::decode(raw, &mut [], 1));
    }
}

fn serde_polygons(raw: &str) -> (Vec<f64>, Vec<u32>) {
    let rings: Vec<Vec<f64>> = serde_json::from_str(raw).unwrap();
    let mut coords = Vec::new();
    let mut ends = Vec::new();
    for ring in rings {
        coords.extend(ring);
        ends.push(coords.len() as u32);
    }
    (coords, ends)
}

#[test]
fn polygons_match_serde_bits() {
    for raw in [
        "[]",
        " [ ] ",
        "[[]]",
        "[[1, 2, 3, 4, 5, 6]]",
        "[[1.5,2.25,3e2,-0,-0.0,32002.703],[ 9007199254740993 , 0.1 ]]",
        "[\n[ 18446744073709551616,\t-9223372036854775809 ]\n,[1,2]]",
        "[[1,2],[],[3,4]]",
    ] {
        let (expected_coords, expected_ends) = serde_polygons(raw);
        let (coords, ends) = pose_numbers::decode_polygons(raw.as_bytes()).expect(raw);
        assert_eq!(ends, expected_ends, "{raw}");
        assert_eq!(coords.len(), expected_coords.len(), "{raw}");
        for (a, b) in coords.iter().zip(&expected_coords) {
            assert_eq!(a.to_bits(), b.to_bits(), "{raw}");
        }
    }
    assert_eq!(
        pose_numbers::decode_polygons(b"[[true, false]]"),
        Some((vec![1.0, 0.0], vec![2]))
    );
}

#[test]
fn polygons_defer_everything_else_to_serde() {
    for raw in [
        "{}",
        "[1, 2]",
        "[[1, [2]]]",
        "[[1, \"2\"]]",
        "[[1, null]]",
        "[[1, 2],]",
        "[[1, 2] [3]]",
        "[[1e400]]",
        "[[1, 2]] x",
    ] {
        assert!(
            pose_numbers::decode_polygons(raw.as_bytes()).is_none(),
            "{raw}"
        );
    }
}
