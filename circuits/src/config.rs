// K_SEGMENTS = SOFTMAX_C
// LIMB_BIS = log2(SOFTMAX_M)로 맞추어야 함. 개념상 분리해두었습니다.

// 세크먼트 및 림브 설정(파이썬 설정값과 일치해야 함)
// v3: single table lookup (no multi-segment decomposition)
pub const K_SEGMENTS: usize = 1;
pub const LIMB_BITS: usize = 16;
pub const LIMB_BASE: u128 = 1u128 << LIMB_BITS;

// v3: single table (C=1) — one 65536-entry exp table
pub const SOFTMAX_C: usize = 1;   // number of segments (C) — was 5 in v2
pub const SOFTMAX_M: usize = 65536; // table width (M) - must be a power of two/ limb bits = 16

pub const RMSNORM_C: usize = 1; 
pub const RMSNORM_M: usize = 65536; // 1 << 16

pub const GELU_C: usize = 1;
pub const GELU_M: usize = 65536; // 65536

pub const KBITS_C : usize = 2; // 26-bit: 16비트 전체 + 10비트 컷오프, 총 2개의 서브테이블
pub const KBITS_M : usize = 65536; // 2^16 = 65536, 16비트 전체를 채우는 테이블 크기

// softmax r_sm 52-bit 범위체크용 (52 = 16*3 + 4, C=4 segments)
pub const SM_KBITS_C : usize = 4;
pub const SM_KBITS_M : usize = 65536; // 2^16

pub const FRACBITS_C : usize = 1; // 16비트 전체를 커버하는 테이블 하나
pub const FRACBITS_M : usize = 65536; // 2^16 = 65536, 16비트 전체를 채우는 테이블 크기