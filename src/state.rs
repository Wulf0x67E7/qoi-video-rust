use crate::pixel::Pixel;

#[derive(Debug, Clone)]
pub struct State {
    index_l1: [Pixel<4>; 0x40],
    index_l2: [Pixel<4>; 0x400],
}
impl State {
    pub(crate) fn index_l1(&mut self, hash_index: u16) -> &mut Pixel<4> {
        &mut self.index_l1[hash_index as usize & 0x3f]
    }
    pub(crate) fn index_l2(&mut self, hash_index: u16) -> &mut Pixel<4> {
        &mut self.index_l2[hash_index as usize & 0x03ff]
    }
}
impl Default for State {
    fn default() -> Self {
        Self { index_l1: [Pixel::new(); 0x40], index_l2: [Pixel::new(); 0x400] }
    }
}
