// K_SEGMENTS = SOFTMAX_C
// LIMB_BIS = log2(SOFTMAX_M)로 맞추어야 함. 개념상 분리해두었습니다.

// 세크먼트 및 림브 설정(파이썬 설정값과 일치해야 함)
pub const K_SEGMENTS: usize = 5;
pub const LIMB_BITS: usize = 16;
pub const LIMB_BASE: u128 = 1u128 << LIMB_BITS; 

// 내가 만든 서브테이블 SPEC과 동일하게 맞춰 주어야 함.
pub const SOFTMAX_C: usize = 5;   // number of segments (C)
pub const SOFTMAX_M: usize = 65536; // table width (M) - must be a power of two/ limb bits = 16