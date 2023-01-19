#[cfg(any(feature = "std", feature = "alloc"))]
use alloc::{vec, vec::Vec};
use core::convert::TryFrom;
use core::mem::replace;
#[cfg(feature = "std")]
use std::io::Write;

use bytemuck::Pod;

use crate::consts::{
    QOI_HEADER_SIZE, QOI_OP_INDEX, QOI_OP_LONG_INDEX, QOI_OP_LONG_RUN, QOI_OP_LONG_RUN_MAX_0,
    QOI_OP_LONG_RUN_MAX_1, QOI_OP_LUMA, QOI_OP_PREV, QOI_OP_RUN, QOI_PADDING, QOI_PADDING_SIZE,
};
use crate::error::{Error, Result};
use crate::header::Header;
use crate::pixel::{Pixel, SupportedChannels};
use crate::types::{Channels, ColorSpace};
#[cfg(feature = "std")]
use crate::utils::GenericWriter;
use crate::utils::{unlikely, BytesMut, Writer};
use crate::State;

#[allow(clippy::cast_possible_truncation, unused_assignments, unused_variables)]
fn encode_impl<W: Writer, const N: usize>(
    state: &mut State, mut buf: W, data: &[u8],
) -> Result<usize>
where
    Pixel<N>: SupportedChannels,
    [u8; N]: Pod,
{
    let cap = buf.capacity();

    let mut px_prev = Pixel::<4>::new().with_a(0xff);
    let mut run = 0_u16;
    let mut px = px_prev;

    let n_pixels = data.len() / N;

    for (i, chunk) in data.chunks_exact(N).enumerate() {
        px.read(chunk);
        if px == px_prev {
            run += 1;
            if run == 1024 {
                buf = buf.write_one(QOI_OP_LONG_RUN_MAX_0)?;
                buf = buf.write_one(QOI_OP_LONG_RUN_MAX_1)?;
                run = 0;
            } else if unlikely(i == n_pixels - 1) {
                if run == 1 {
                    buf = buf.write_one(QOI_OP_PREV)?;
                } else if run <= 63 {
                    buf = buf.write_one(QOI_OP_RUN | (run as u8 - 2))?;
                } else {
                    let run = run - 64;
                    buf = buf.write_one(QOI_OP_LUMA | (run & 0x3f) as u8)?;
                    buf = buf.write_one(QOI_OP_LONG_RUN | (run >> 6) as u8)?;
                }
                run = 0;
            }
        } else {
            if run != 0 {
                if run == 1 {
                    buf = buf.write_one(QOI_OP_PREV)?;
                } else if run <= 63 {
                    buf = buf.write_one(QOI_OP_RUN | (run as u8 - 2))?;
                } else {
                    let run = run - 64;
                    buf = buf.write_one(QOI_OP_LUMA | (run & 0x3f) as u8)?;
                    buf = buf.write_one(QOI_OP_LONG_RUN | (run >> 6) as u8)?;
                }
                run = 0;
            }
            let px_rgba = px.as_rgba(0xff);
            let px_hash = px_rgba.hash_index();
            let index_px = state.index_l1(px_hash);
            if *index_px == px_rgba {
                buf = buf.write_one(QOI_OP_INDEX | (px_hash as u8 & 0x3f))?;
            } else {
                let old_px_l1 = replace(index_px, px_rgba);
                let (mut len, mut encoded) = px.encode(px_prev);
                if len <= 2 && *state.index_l2(px_hash) == px_rgba {
                    len = 2;
                    encoded = [
                        QOI_OP_LUMA | (px_hash & 0x3f) as u8,
                        (px_hash >> 2) as u8 | QOI_OP_LONG_INDEX,
                        0,
                        0,
                        0,
                    ];
                }
                buf = buf.write_many(&encoded[..len])?;
                *state.index_l2(old_px_l1.hash_index()) = old_px_l1;
            }
            px_prev = px;
        }
    }
    Ok(cap.saturating_sub(buf.capacity()))
}

#[inline]
fn encode_impl_all<W: Writer>(
    state: &mut State, out: W, data: &[u8], channels: Channels,
) -> Result<usize> {
    match channels {
        Channels::Rgb => encode_impl::<_, 3>(state, out, data),
        Channels::Rgba => encode_impl::<_, 4>(state, out, data),
    }
}

/// The maximum number of bytes the encoded image will take.
///
/// Can be used to pre-allocate the buffer to encode the image into.
#[inline]
pub fn encode_max_len<const DATA_ONLY: bool>(
    width: u32, height: u32, channels: impl Into<u8>,
) -> usize {
    let (width, height) = (width as usize, height as usize);
    let n_pixels = width.saturating_mul(height);
    n_pixels.saturating_mul(channels.into() as usize + 1)
        + if DATA_ONLY { 0 } else { QOI_HEADER_SIZE + QOI_PADDING_SIZE }
}

/// Encode the image into a pre-allocated buffer.
///
/// Returns the total number of bytes written.
#[inline]
pub fn encode_to_buf<const DATA_ONLY: bool>(
    buf: impl AsMut<[u8]>, data: impl AsRef<[u8]>, width: u32, height: u32,
) -> Result<usize> {
    Encoder::new(&data, width, height)?.encode_to_buf::<DATA_ONLY>(buf)
}

/// Encode the image into a newly allocated vector.
#[cfg(any(feature = "alloc", feature = "std"))]
#[inline]
pub fn encode_to_vec<const DATA_ONLY: bool>(
    data: impl AsRef<[u8]>, width: u32, height: u32,
) -> Result<Vec<u8>> {
    Encoder::new(&data, width, height)?.encode_to_vec::<DATA_ONLY>()
}

/// Encode QOI images into buffers or into streams.
pub struct Encoder<'a> {
    data: &'a [u8],
    header: Header,
    state: State,
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder from a given array of pixel data and image dimensions.
    ///
    /// The number of channels will be inferred automatically (the valid values
    /// are 3 or 4). The color space will be set to sRGB by default.
    #[inline]
    pub fn new(data: &'a (impl AsRef<[u8]> + ?Sized), width: u32, height: u32) -> Result<Self> {
        Self::new_with(State::default(), data, width, height)
    }
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_with(
        state: State, data: &'a (impl AsRef<[u8]> + ?Sized), width: u32, height: u32,
    ) -> Result<Self> {
        let data = data.as_ref();
        let mut header =
            Header::try_new(width, height, Channels::default(), ColorSpace::default())?;
        let size = data.len();
        let n_channels = size / header.n_pixels();
        if header.n_pixels() * n_channels != size {
            return Err(Error::InvalidImageLength { size, width, height });
        }
        header.channels = Channels::try_from(n_channels.min(0xff) as u8)?;
        Ok(Self { data, header, state })
    }

    /// Returns a new encoder with modified color space.
    ///
    /// Note: the color space doesn't affect encoding or decoding in any way, it's
    /// a purely informative field that's stored in the image header.
    #[inline]
    pub const fn with_colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.header = self.header.with_colorspace(colorspace);
        self
    }

    /// Returns the inferred number of channels.
    #[inline]
    pub const fn channels(&self) -> Channels {
        self.header.channels
    }

    /// Returns the header that will be stored in the encoded image.
    #[inline]
    pub const fn header(&self) -> &Header {
        &self.header
    }

    /// The maximum number of bytes the encoded image will take.
    ///
    /// Can be used to pre-allocate the buffer to encode the image into.
    #[inline]
    pub fn required_buf_len<const DATA_ONLY: bool>(&self) -> usize {
        self.header.encode_max_len::<DATA_ONLY>()
    }

    /// Encodes the image to a pre-allocated buffer and returns the number of bytes written.
    ///
    /// The minimum size of the buffer can be found via [`Encoder::required_buf_len`].
    #[inline]
    pub fn encode_to_buf<const DATA_ONLY: bool>(
        &mut self, mut buf: impl AsMut<[u8]>,
    ) -> Result<usize> {
        let buf = buf.as_mut();
        let size_required = self.required_buf_len::<DATA_ONLY>();
        if unlikely(buf.len() < size_required) {
            return Err(Error::OutputBufferTooSmall { size: buf.len(), required: size_required });
        }
        let mut n_written = 0;
        if !DATA_ONLY {
            buf[..QOI_HEADER_SIZE].copy_from_slice(&self.header.encode());
            n_written += QOI_HEADER_SIZE;
        }
        n_written += encode_impl_all(
            &mut self.state,
            BytesMut::new(&mut buf[n_written..]),
            self.data,
            self.header.channels,
        )?;
        if !DATA_ONLY {
            buf[n_written..n_written + QOI_PADDING_SIZE].copy_from_slice(&QOI_PADDING);
            n_written += QOI_PADDING_SIZE;
        }
        Ok(n_written)
    }

    /// Encodes the image into a newly allocated vector of bytes and returns it.
    #[cfg(any(feature = "alloc", feature = "std"))]
    #[inline]
    pub fn encode_to_vec<const DATA_ONLY: bool>(&mut self) -> Result<Vec<u8>> {
        let mut out = vec![0_u8; self.required_buf_len::<DATA_ONLY>()];
        let size = self.encode_to_buf::<DATA_ONLY>(&mut out)?;
        out.truncate(size);
        Ok(out)
    }

    /// Encodes the image directly to a generic writer that implements [`Write`](std::io::Write).
    ///
    /// Note: while it's possible to pass a `&mut [u8]` slice here since it implements `Write`,
    /// it would more effficient to use a specialized method instead: [`Encoder::encode_to_buf`].
    #[cfg(feature = "std")]
    #[inline]
    pub fn encode_to_stream<W: Write, const DATA_ONLY: bool>(
        &mut self, mut writer: W,
    ) -> Result<usize> {
        let mut n_written = 0;
        if !DATA_ONLY {
            writer.write_all(&self.header.encode())?;
            n_written += QOI_HEADER_SIZE;
        }
        n_written += encode_impl_all(
            &mut self.state,
            GenericWriter::new(&mut writer),
            self.data,
            self.header.channels,
        )?;
        if !DATA_ONLY {
            writer.write_all(&QOI_PADDING)?;
            n_written += QOI_PADDING_SIZE;
        }
        Ok(n_written)
    }

    #[inline]
    pub fn into_state(self) -> State {
        self.state
    }
}
