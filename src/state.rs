use crate::pixel::Pixel;

#[derive(Debug, Clone)]
pub struct State {
    pub(crate) index: [Pixel<4>; 256],
    pub(crate) px_prev: Pixel<4>,
    pub(crate) index_allowed: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            index: [Pixel::new(); 256],
            px_prev: Pixel::new().with_a(0xff),
            index_allowed: false,
        }
    }
}
