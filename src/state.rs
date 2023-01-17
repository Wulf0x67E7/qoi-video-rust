use crate::pixel::Pixel;

#[derive(Debug, Clone)]
pub struct State {
    pub(crate) index: [Pixel<4>; 0x40],
    pub(crate) index_allowed: bool,
}
impl Default for State {
    fn default() -> Self {
        Self { index: [Pixel::new(); 0x40], index_allowed: false }
    }
}
