// 2-bit tag
pub const QOI_INDEX: u8 = 0x00; // (00)000000
pub const QOI_DIFF_8: u8 = 0x80; // (10)000000

// 3-bit tag
pub const QOI_RUN_8: u8 = 0x40; // (010)00000
pub const QOI_RUN_16: u8 = 0x60; // (011)00000
pub const QOI_DIFF_16: u8 = 0xc0; // (110)00000

// 4-bit tag
pub const QOI_DIFF_24: u8 = 0xe0; // (1110)0000
pub const QOI_COLOR: u8 = 0xf0; // (1111)0000

// tag masks
#[allow(unused)]
pub const QOI_MASK_2: u8 = 0xc0; // (11)000000
#[allow(unused)]
pub const QOI_MASK_3: u8 = 0xe0; // (111)00000
#[allow(unused)]
pub const QOI_MASK_4: u8 = 0xf0; // (1111)0000

pub const QOI_HEADER_SIZE: usize = 14;
pub const QOI_PADDING: usize = 4;

pub const QOI_MAGIC: [u8; 4] = [b'q', b'o', b'i', b'f'];
