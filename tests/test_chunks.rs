mod common;

use bytemuck::{cast_slice, Pod};

use qoi::consts::{
    QOI_HEADER_SIZE, QOI_OP_DIFF, QOI_OP_INDEX, QOI_OP_LONG_RUN, QOI_OP_LONG_RUN_MAX_0,
    QOI_OP_LONG_RUN_MAX_1, QOI_OP_LUMA, QOI_OP_PREV, QOI_OP_RGB, QOI_OP_RGBA, QOI_OP_RUN,
    QOI_PADDING_SIZE,
};
use qoi::{decode_to_vec, encode_to_vec};

use self::common::hash;

fn test_chunk<P, E, const N: usize>(pixels: P, expected: E)
where
    P: AsRef<[[u8; N]]>,
    E: AsRef<[u8]>,
    [u8; N]: Pod,
{
    let pixels = pixels.as_ref();
    let expected = expected.as_ref();
    let pixels_raw = cast_slice::<_, u8>(pixels);
    let encoded = encode_to_vec::<false>(pixels_raw, pixels.len() as _, 1).unwrap();
    let decoded = decode_to_vec::<false>(&encoded).unwrap().1;
    assert_eq!(pixels_raw, decoded.as_slice(), "roundtrip failed (encoded={:?}))", encoded);
    assert!(encoded.len() >= expected.len() + QOI_HEADER_SIZE + QOI_PADDING_SIZE);
    assert_eq!(&encoded[QOI_HEADER_SIZE..][..expected.len()], expected);
}

#[test]
fn test_encode_rgb_3ch() {
    test_chunk([[11, 121, 231]], [QOI_OP_RGB, 11, 121, 231]);
}

#[test]
fn test_encode_rgb_4ch() {
    test_chunk([[11, 121, 231, 0xff]], [QOI_OP_RGB, 11, 121, 231]);
}

#[test]
fn test_encode_rgba() {
    test_chunk([[11, 121, 231, 55]], [QOI_OP_RGBA, 11, 121, 231, 55]);
}

#[test]
fn test_encode_run_1_3ch() {
    let v = vec![[0, 0, 0], [11, 22, 33]];
    test_chunk(v, [QOI_OP_PREV, QOI_OP_RGB]);
}

#[test]
fn test_encode_run_1_4ch() {
    let v = vec![[0, 0, 0, 0xff], [11, 22, 33, 44]];
    test_chunk(v, [QOI_OP_PREV, QOI_OP_RGBA]);
}

#[test]
fn test_encode_run_2to63_3ch() {
    for n in 2..=63 {
        let mut v = vec![[0, 0, 0]; n];
        v.push([11, 22, 33]);
        test_chunk(v, [QOI_OP_RUN | (n as u8 - 2), QOI_OP_RGB]);
    }
}

#[test]
fn test_encode_run_2to63_4ch() {
    for n in 2..=63 {
        let mut v = vec![[0, 0, 0, 0xff]; n];
        v.push([11, 22, 33, 44]);
        test_chunk(v, [QOI_OP_RUN | (n as u8 - 2), QOI_OP_RGBA]);
    }
}

#[test]
fn test_encode_run_64to1023_3ch() {
    for n in 64..=1023 {
        let mut v = vec![[0, 0, 0]; n];
        v.push([11, 22, 33]);
        let n = n - 64;
        test_chunk(
            v,
            [
                QOI_OP_LUMA | (n & 0x3f) as u8,
                QOI_OP_LONG_RUN | (n >> 6) as u8,
                QOI_OP_RGB,
                11,
                22,
                33,
            ],
        );
    }
}

#[test]
fn test_encode_run_64to1023_4ch() {
    for n in 64..=1023 {
        let mut v = vec![[0, 0, 0, 0xff]; n];
        v.push([11, 22, 33, 44]);
        let n = n - 64;
        test_chunk(
            v,
            [
                QOI_OP_LUMA | (n & 0x3f) as u8,
                QOI_OP_LONG_RUN | (n >> 6) as u8,
                QOI_OP_RGBA,
                11,
                22,
                33,
                44,
            ],
        );
    }
}

#[test]
fn test_encode_run_1024_3ch() {
    let px = [11, 33, 55];
    let mut v = vec![[1, 99, 2]];
    const N: usize = 1024;
    v.extend([px; N + 1]);
    test_chunk(
        v,
        [
            QOI_OP_RGB,
            1,
            99,
            2,
            QOI_OP_RGB,
            px[0],
            px[1],
            px[2],
            QOI_OP_LONG_RUN_MAX_0,
            QOI_OP_LONG_RUN_MAX_1,
        ],
    );
}

#[test]
fn test_encode_run_1024_4ch() {
    let px = [11, 33, 55, 77];
    let mut v = vec![[1, 99, 2, 3]];
    const N: usize = 1024;
    v.extend([px; N + 1]);
    test_chunk(
        v,
        [
            QOI_OP_RGBA,
            1,
            99,
            2,
            3,
            QOI_OP_RGBA,
            px[0],
            px[1],
            px[2],
            px[3],
            QOI_OP_LONG_RUN_MAX_0,
            QOI_OP_LONG_RUN_MAX_1,
        ],
    );
}

#[test]
fn test_encode_run_long_end_3ch() {
    let px = [11, 33, 55];
    let mut v = vec![[1, 99, 2]];
    const N: usize = 960;
    v.extend([px; N + 1]);
    let n = N - 64;
    test_chunk(
        v,
        [
            QOI_OP_RGB,
            1,
            99,
            2,
            QOI_OP_RGB,
            px[0],
            px[1],
            px[2],
            QOI_OP_LUMA | (n & 0x3f) as u8,
            QOI_OP_LONG_RUN | (n >> 6) as u8,
        ],
    );
}

#[test]
fn test_encode_run_long_end_4ch() {
    let px = [11, 33, 55, 77];
    let mut v = vec![[1, 99, 2, 3]];
    const N: usize = 960;
    v.extend([px; N + 1]);
    let n = N - 64;
    test_chunk(
        v,
        [
            QOI_OP_RGBA,
            1,
            99,
            2,
            3,
            QOI_OP_RGBA,
            px[0],
            px[1],
            px[2],
            px[3],
            QOI_OP_LUMA | (n & 0x3f) as u8,
            QOI_OP_LONG_RUN | (n >> 6) as u8,
        ],
    );
}

#[test]
fn test_encode_run_short_end_3ch() {
    let px = [11, 33, 55];
    test_chunk(
        [[1, 99, 2], px, px, px],
        [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGB, px[0], px[1], px[2], QOI_OP_RUN],
    );
}

#[test]
fn test_encode_run_short_end_4ch() {
    let px = [11, 33, 55, 77];
    test_chunk(
        [[1, 99, 2, 3], px, px, px],
        [QOI_OP_RGBA, 1, 99, 2, 3, QOI_OP_RGBA, px[0], px[1], px[2], px[3], QOI_OP_RUN],
    );
}

#[test]
fn test_encode_run_mid_3ch() {
    let px = [11, 33, 55];
    test_chunk(
        [[1, 99, 2], px, px, px, [1, 2, 3]],
        [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGB, px[0], px[1], px[2], QOI_OP_RUN],
    );
}

#[test]
fn test_encode_run_mid_4ch() {
    let px = [11, 33, 55, 77];
    test_chunk(
        [[1, 99, 2, 3], px, px, px, [1, 2, 3, 4]],
        [QOI_OP_RGBA, 1, 99, 2, 3, QOI_OP_RGBA, px[0], px[1], px[2], px[3], QOI_OP_RUN],
    );
}

#[test]
fn test_encode_index_3ch() {
    let px = [101, 102, 103];
    test_chunk(
        [px, [1, 2, 3], px],
        [QOI_OP_RGB, 101, 102, 103, QOI_OP_RGB, 1, 2, 3, QOI_OP_INDEX | hash(px)],
    );
}

#[test]
fn test_encode_index_4ch() {
    let px = [101, 102, 103, 104];
    test_chunk(
        [px, [1, 2, 3, 4], px],
        [QOI_OP_RGBA, 101, 102, 103, 104, QOI_OP_RGBA, 1, 2, 3, 4, QOI_OP_INDEX | hash(px)],
    );
}

#[test]
fn test_encode_index_zero_3ch() {
    let px = [0, 0, 0];
    test_chunk([[101, 102, 103], px], [QOI_OP_RGB, 101, 102, 103, QOI_OP_RGB, 0, 0, 0]);
}

#[test]
fn test_encode_index_zero_0x00_4ch() {
    let px = [0, 0, 0, 0];
    test_chunk(
        [[101, 102, 103, 104], px],
        [QOI_OP_RGBA, 101, 102, 103, 104, QOI_OP_INDEX | hash(px)],
    );
}

#[test]
fn test_encode_index_zero_0xff_4ch() {
    let px = [0, 0, 0, 0xff];
    test_chunk(
        [[101, 102, 103, 104], px],
        [QOI_OP_RGBA, 101, 102, 103, 104, QOI_OP_RGBA, 0, 0, 0, 0xff],
    );
}

#[test]
fn test_encode_diff() {
    for x in 0..8_u8 {
        let x = [x.wrapping_sub(5), x.wrapping_sub(4), x.wrapping_sub(3)];
        for dr in 0..3 {
            for dg in 0..3 {
                for db in 0..3 {
                    if dr != 2 || dg != 2 || db != 2 {
                        let r = x[0].wrapping_add(dr).wrapping_sub(2);
                        let g = x[1].wrapping_add(dg).wrapping_sub(2);
                        let b = x[2].wrapping_add(db).wrapping_sub(2);
                        let d = QOI_OP_DIFF | dr << 4 | dg << 2 | db;
                        test_chunk(
                            [[1, 99, 2], x, [r, g, b]],
                            [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGB, x[0], x[1], x[2], d],
                        );
                        test_chunk(
                            [[1, 99, 2, 0xff], [x[0], x[1], x[2], 9], [r, g, b, 9]],
                            [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGBA, x[0], x[1], x[2], 9, d],
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_encode_luma() {
    for x in (0..200_u8).step_by(4) {
        let x = [x.wrapping_mul(3), x.wrapping_sub(5), x.wrapping_sub(7)];
        for dr_g in (0..15).step_by(4) {
            for dg in (0..64).step_by(8) {
                for db_g in (0..15).step_by(4) {
                    if dr_g != 8 || dg != 32 || db_g != 8 {
                        let r = x[0].wrapping_add(dr_g).wrapping_add(dg).wrapping_sub(39);
                        let g = x[1].wrapping_add(dg).wrapping_sub(32);
                        let b = x[2].wrapping_add(db_g).wrapping_add(dg).wrapping_sub(39);
                        let d1 = QOI_OP_LUMA | dg;
                        let d2 = (dr_g << 4) | db_g;
                        test_chunk(
                            [[1, 99, 2], x, [r, g, b]],
                            [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGB, x[0], x[1], x[2], d1, d2],
                        );
                        test_chunk(
                            [[1, 99, 2, 0xff], [x[0], x[1], x[2], 9], [r, g, b, 9]],
                            [QOI_OP_RGB, 1, 99, 2, QOI_OP_RGBA, x[0], x[1], x[2], 9, d1, d2],
                        );
                    }
                }
            }
        }
    }
}
