#[cfg(any(feature = "std", feature = "alloc"))]
use alloc::{vec, vec::Vec};
use core::mem::replace;
use core::ops::Add;
#[cfg(feature = "std")]
use std::io::Read;

// TODO: can be removed once https://github.com/rust-lang/rust/issues/74985 is stable
use bytemuck::{cast_slice_mut, Pod};

use crate::consts::{
    QOI_HEADER_SIZE, QOI_OP_DIFF, QOI_OP_INDEX, QOI_OP_LONG_INDEX, QOI_OP_LONG_RUN, QOI_OP_LUMA,
    QOI_OP_RGB, QOI_OP_RGBA, QOI_OP_RUN, QOI_PADDING, QOI_PADDING_SIZE,
};
use crate::error::{Error, Result};
use crate::header::Header;
use crate::pixel::{Pixel, SupportedChannels};
use crate::types::Channels;
use crate::utils::{cold, unlikely};
use crate::State;

const QOI_OP_INDEX_END: u8 = QOI_OP_INDEX | 0x3f;
const QOI_OP_RUN_END: u8 = QOI_OP_RUN | 0x3d; // <- note, 0x3d (not 0x3f)
const QOI_OP_DIFF_END: u8 = QOI_OP_DIFF | 0x3f;
const QOI_OP_LUMA_END: u8 = QOI_OP_LUMA | 0x3f;

#[inline]
fn decode_impl_slice<const N: usize>(
    state: &mut State, mut data: &[u8], out: &mut [u8],
) -> Result<usize>
where
    Pixel<N>: SupportedChannels,
    [u8; N]: Pod,
{
    let mut pixels = cast_slice_mut::<_, [u8; N]>(out);
    let data_len = data.len();

    let mut px = Pixel::<4>::new().with_a(0xff);

    while let [px_out, ptail @ ..] = pixels {
        pixels = ptail;
        match data {
            [b1 @ QOI_OP_INDEX..=QOI_OP_INDEX_END, dtail @ ..] => {
                px = *state.index_l1(*b1 as u16);
                *px_out = px.into();
                data = dtail;
                continue;
            }
            [QOI_OP_RGB, r, g, b, dtail @ ..] => {
                px.update_rgb(*r, *g, *b);
                data = dtail;
            }
            [QOI_OP_RGBA, r, g, b, a, dtail @ ..] => {
                px.update_rgba(*r, *g, *b, *a);
                data = dtail;
            }
            [b1 @ QOI_OP_RUN..=QOI_OP_RUN_END, dtail @ ..] => {
                *px_out = px.into();
                let run = ((b1 & 0x3f) as usize).min(pixels.len());
                let (phead, ptail) = pixels.split_at_mut(run); // can't panic
                phead.fill(px.into());
                pixels = ptail;
                data = dtail;
                continue;
            }
            [b1 @ QOI_OP_DIFF..=QOI_OP_DIFF_END, dtail @ ..] => {
                px.update_diff(*b1);
                data = dtail;
            }
            [b1 @ QOI_OP_LUMA..=QOI_OP_LUMA_END, b2, dtail @ ..] => {
                match b2 {
                    _ if b2 & QOI_OP_LONG_RUN == QOI_OP_LONG_RUN => {
                        *px_out = px.into();
                        let run = (((b1 & 0x3f) as usize) << 4 | (b2 & QOI_OP_LONG_INDEX) as usize)
                            .add((1 + QOI_OP_RUN_END - QOI_OP_RUN) as usize)
                            .min(pixels.len());
                        let (phead, ptail) = pixels.split_at_mut(run); // can't panic
                        phead.fill(px.into());
                        pixels = ptail;
                        data = dtail;
                        continue;
                    }
                    _ if b2 & QOI_OP_LONG_INDEX == QOI_OP_LONG_INDEX => {
                        let hash_index =
                            (((b1 & 0x3f) as u16) << 4) | (((b2 & QOI_OP_LONG_RUN) as u16) >> 4);
                        px = *state.index_l2(hash_index);
                        *px_out = px.into();
                        // Move chosen l2 into l1 and evicted l1 into l2
                        let old_px_l1 = replace(state.index_l1(hash_index), px);
                        *state.index_l2(old_px_l1.hash_index()) = old_px_l1;
                        // cleanup
                        data = dtail;
                        continue;
                    }
                    _ => {
                        px.update_luma(*b1, *b2);
                        data = dtail;
                    }
                }
            }
            _ => {
                cold();
                return Err(Error::UnexpectedBufferEnd);
            }
        }
        // Move px into l1 and evicted l1 into l2
        let old_px_l1 = replace(state.index_l1(px.hash_index()), px);
        *state.index_l2(old_px_l1.hash_index()) = old_px_l1;

        *px_out = px.into();
    }

    Ok(data_len.saturating_sub(data.len()))
}

#[inline]
fn decode_impl_slice_all(
    state: &mut State, data: &[u8], out: &mut [u8], channels: u8, src_channels: u8,
) -> Result<usize> {
    match (channels, src_channels) {
        (3, 3) => decode_impl_slice::<3>(state, data, out),
        (3, 4) => decode_impl_slice::<3>(state, data, out),
        (4, 3) => decode_impl_slice::<4>(state, data, out),
        (4, 4) => decode_impl_slice::<4>(state, data, out),
        _ => {
            cold();
            Err(Error::InvalidChannels { channels })
        }
    }
}

/// Decode the image into a pre-allocated buffer.
///
/// Note: the resulting number of channels will match the header. In order to change
/// the number of channels, use [`Decoder::with_channels`].
#[inline]
pub fn decode_to_buf<const DATA_ONLY: bool>(
    buf: impl AsMut<[u8]>, data: impl AsRef<[u8]>,
) -> Result<Header> {
    let mut decoder = Decoder::new(&data)?;
    decoder.decode_to_buf::<DATA_ONLY>(buf)?;
    Ok(*decoder.header())
}

/// Decode the image into a newly allocated vector.
///
/// Note: the resulting number of channels will match the header. In order to change
/// the number of channels, use [`Decoder::with_channels`].
#[cfg(any(feature = "std", feature = "alloc"))]
#[inline]
pub fn decode_to_vec<const DATA_ONLY: bool>(data: impl AsRef<[u8]>) -> Result<(Header, Vec<u8>)> {
    let mut decoder = Decoder::new(&data)?;
    let out = decoder.decode_to_vec::<DATA_ONLY>()?;
    Ok((*decoder.header(), out))
}

/// Decode the image header from a slice of bytes.
#[inline]
pub fn decode_header(data: impl AsRef<[u8]>) -> Result<Header> {
    Header::decode(data)
}

#[cfg(any(feature = "std"))]
#[inline]
fn decode_impl_stream<R: Read, const N: usize>(
    state: &mut State, data: &mut R, out: &mut [u8],
) -> Result<()>
where
    Pixel<N>: SupportedChannels,
    [u8; N]: Pod,
{
    let mut pixels = cast_slice_mut::<_, [u8; N]>(out);

    let mut px = Pixel::<4>::new().with_a(0xff);

    while let [px_out, ptail @ ..] = pixels {
        pixels = ptail;
        let mut p = [0];
        data.read_exact(&mut p)?;
        let [b1] = p;
        match b1 {
            QOI_OP_INDEX..=QOI_OP_INDEX_END => {
                px = *state.index_l1(b1 as u16);
                *px_out = px.into();
                continue;
            }
            QOI_OP_RGB => {
                let mut p = [0; 3];
                data.read_exact(&mut p)?;
                px.update_rgb(p[0], p[1], p[2]);
            }
            QOI_OP_RGBA => {
                let mut p = [0; 4];
                data.read_exact(&mut p)?;
                px.update_rgba(p[0], p[1], p[2], p[3]);
            }
            QOI_OP_RUN..=QOI_OP_RUN_END => {
                *px_out = px.into();
                let run = ((b1 & 0x3f) as usize).min(pixels.len());
                let (phead, ptail) = pixels.split_at_mut(run); // can't panic
                phead.fill(px.into());
                pixels = ptail;
                continue;
            }
            QOI_OP_DIFF..=QOI_OP_DIFF_END => {
                px.update_diff(b1);
            }
            QOI_OP_LUMA..=QOI_OP_LUMA_END => {
                let mut p = [0];
                data.read_exact(&mut p)?;
                let [b2] = p;
                match b2 {
                    _ if b2 & QOI_OP_LONG_RUN == QOI_OP_LONG_RUN => {
                        *px_out = px.into();
                        let run = (((b1 & 0x3f) as usize) << 4 | (b2 & QOI_OP_LONG_INDEX) as usize)
                            .add((1 + QOI_OP_RUN_END - QOI_OP_RUN) as usize)
                            .min(pixels.len());
                        let (phead, ptail) = pixels.split_at_mut(run); // can't panic
                        phead.fill(px.into());
                        pixels = ptail;
                        continue;
                    }
                    _ if b2 & QOI_OP_LONG_INDEX == QOI_OP_LONG_INDEX => {
                        let hash_index =
                            (((b1 & 0x3f) as u16) << 4) | (((b2 & QOI_OP_LONG_RUN) as u16) >> 4);
                        px = *state.index_l2(hash_index);
                        *px_out = px.into();
                        // Move chosen l2 into l1 and evicted l1 into l2
                        let old_px_l1 = replace(state.index_l1(hash_index), px);
                        *state.index_l2(old_px_l1.hash_index()) = old_px_l1;
                        continue;
                    }
                    _ => {
                        px.update_luma(b1, b2);
                    }
                }
            }
        }
        // Move px into l1 and evicted l1 into l2
        let old_px_l1 = replace(state.index_l1(px.hash_index()), px);
        *state.index_l2(old_px_l1.hash_index()) = old_px_l1;

        *px_out = px.into();
    }
    Ok(())
}

#[cfg(feature = "std")]
#[inline]
fn decode_impl_stream_all<R: Read>(
    state: &mut State, data: &mut R, out: &mut [u8], channels: u8, src_channels: u8,
) -> Result<()> {
    match (channels, src_channels) {
        (3, 3) => decode_impl_stream::<_, 3>(state, data, out),
        (3, 4) => decode_impl_stream::<_, 3>(state, data, out),
        (4, 3) => decode_impl_stream::<_, 4>(state, data, out),
        (4, 4) => decode_impl_stream::<_, 4>(state, data, out),
        _ => {
            cold();
            Err(Error::InvalidChannels { channels })
        }
    }
}

#[doc(hidden)]
pub trait Reader: Sized {
    fn decode_header(&mut self) -> Result<Header>;
    fn decode_image<const DATA_ONLY: bool>(
        &mut self, state: &mut State, out: &mut [u8], channels: u8, src_channels: u8,
    ) -> Result<()>;
}

pub struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    #[inline]
    pub const fn new(buf: &'a [u8]) -> Self {
        Self(buf)
    }

    #[inline]
    pub const fn as_slice(&self) -> &[u8] {
        self.0
    }
}

impl<'a> Reader for Bytes<'a> {
    #[inline]
    fn decode_header(&mut self) -> Result<Header> {
        let header = Header::decode(self.0)?;
        self.0 = &self.0[QOI_HEADER_SIZE..]; // can't panic
        Ok(header)
    }

    #[inline]
    fn decode_image<const DATA_ONLY: bool>(
        &mut self, state: &mut State, out: &mut [u8], channels: u8, src_channels: u8,
    ) -> Result<()> {
        let n_read = decode_impl_slice_all(state, self.0, out, channels, src_channels)?;
        self.0 = &self.0[n_read..];
        if !DATA_ONLY && unlikely(self.0.len() < QOI_PADDING_SIZE) {
            Err(Error::UnexpectedBufferEnd)
        } else if !DATA_ONLY && unlikely(self.0[..QOI_PADDING_SIZE] != QOI_PADDING) {
            Err(Error::InvalidPadding)
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "std")]
impl<R: Read> Reader for R {
    #[inline]
    fn decode_header(&mut self) -> Result<Header> {
        let mut b = [0; QOI_HEADER_SIZE];
        self.read_exact(&mut b)?;
        Header::decode(b)
    }

    #[inline]
    fn decode_image<const DATA_ONLY: bool>(
        &mut self, state: &mut State, out: &mut [u8], channels: u8, src_channels: u8,
    ) -> Result<()> {
        decode_impl_stream_all(state, self, out, channels, src_channels)?;
        if !DATA_ONLY && {
            let mut p = [0; QOI_PADDING_SIZE];
            self.read_exact(&mut p)?;
            p != QOI_PADDING
        } {
            Err(Error::InvalidPadding)
        } else {
            Ok(())
        }
    }
}

/// Decode QOI images from slices or from streams.
#[derive(Clone)]
pub struct Decoder<R> {
    reader: R,
    header: Header,
    channels: Channels,
    state: State,
}

impl<'a> Decoder<Bytes<'a>> {
    /// Creates a new decoder from a slice of bytes.
    ///
    /// The header will be decoded immediately upon construction.
    ///
    /// Note: this provides the most efficient decoding, but requires the source data to
    /// be loaded in memory in order to decode it. In order to decode from a generic
    /// stream, use [`Decoder::from_stream`] instead.
    /// #[inline]
    pub fn new(data: &'a (impl AsRef<[u8]> + ?Sized)) -> Result<Self> {
        let mut reader = Bytes::new(data.as_ref());
        Ok(Self::new_with(reader.decode_header()?, State::default(), reader))
    }
    #[inline]
    pub fn new_with(header: Header, state: State, reader: Bytes<'a>) -> Self {
        Self::new_impl(header, state, reader)
    }

    /// Returns the undecoded tail of the input slice of bytes.
    #[inline]
    pub const fn data(&self) -> &[u8] {
        self.reader.as_slice()
    }
}

#[cfg(feature = "std")]
impl<R: Read> Decoder<R> {
    /// Creates a new decoder from a generic reader that implements [`Read`](std::io::Read).
    ///
    /// The header will be decoded immediately upon construction.
    ///
    /// Note: while it's possible to pass a `&[u8]` slice here since it implements `Read`, it
    /// would be more efficient to use a specialized constructor instead: [`Decoder::new`].
    #[inline]
    pub fn from_stream(reader: R) -> Result<Self> {
        Self::from_stream_with(State::default(), reader)
    }
    #[inline]
    pub fn from_stream_with(state: State, mut reader: R) -> Result<Self> {
        Ok(Self::new_impl(reader.decode_header()?, state, reader))
    }

    /// Returns an immutable reference to the underlying reader.
    #[inline]
    pub const fn reader(&self) -> &R {
        &self.reader
    }

    /// Consumes the decoder and returns the underlying reader back.
    #[inline]
    #[allow(clippy::missing_const_for_fn)]
    pub fn into_reader(self) -> R {
        self.reader
    }
}

impl<R: Reader> Decoder<R> {
    #[inline]
    fn new_impl(header: Header, state: State, reader: R) -> Self {
        Self { reader, header, channels: header.channels, state }
    }

    /// Returns a new decoder with modified number of channels.
    ///
    /// By default, the number of channels in the decoded image will be equal
    /// to whatever is specified in the header. However, it is also possible
    /// to decode RGB into RGBA (in which case the alpha channel will be set
    /// to 255), and vice versa (in which case the alpha channel will be ignored).
    #[inline]
    pub const fn with_channels(mut self, channels: Channels) -> Self {
        self.channels = channels;
        self
    }

    /// Returns the number of channels in the decoded image.
    ///
    /// Note: this may differ from the number of channels specified in the header.
    #[inline]
    pub const fn channels(&self) -> Channels {
        self.channels
    }

    /// Returns the decoded image header.
    #[inline]
    pub const fn header(&self) -> &Header {
        &self.header
    }

    /// The number of bytes the decoded image will take.
    ///
    /// Can be used to pre-allocate the buffer to decode the image into.
    #[inline]
    pub const fn required_buf_len(&self) -> usize {
        self.header.n_pixels().saturating_mul(self.channels.as_u8() as usize)
    }

    /// Decodes the image to a pre-allocated buffer and returns the number of bytes written.
    ///
    /// The minimum size of the buffer can be found via [`Decoder::required_buf_len`].
    #[inline]
    pub fn decode_to_buf<const DATA_ONLY: bool>(
        &mut self, mut buf: impl AsMut<[u8]>,
    ) -> Result<usize> {
        let buf = buf.as_mut();
        let size = self.required_buf_len();
        if unlikely(buf.len() < size) {
            return Err(Error::OutputBufferTooSmall { size: buf.len(), required: size });
        }
        self.reader.decode_image::<DATA_ONLY>(
            &mut self.state,
            buf,
            self.channels.as_u8(),
            self.header.channels.as_u8(),
        )?;
        Ok(size)
    }

    /// Decodes the image into a newly allocated vector of bytes and returns it.
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[inline]
    pub fn decode_to_vec<const DATA_ONLY: bool>(&mut self) -> Result<Vec<u8>> {
        let mut out = vec![0; self.header.n_pixels() * self.channels.as_u8() as usize];
        let _ = self.decode_to_buf::<DATA_ONLY>(&mut out)?;
        Ok(out)
    }
    #[inline]
    pub fn extract_state(self) -> State {
        self.state
    }
}
