// K_SEGMENTS = SOFTMAX_C
// LIMB_BIS = log2(SOFTMAX_M)로 맞추어야 함. 개념상 분리해두었습니다.

// 세크먼트 및 림브 설정(파이썬 설정값과 일치해야 함)
pub const K_SEGMENTS: usize = 5;
pub const LIMB_BITS: usize = 16;
pub const LIMB_BASE: u128 = 1u128 << LIMB_BITS; 

// 내가 만든 서브테이블 SPEC과 동일하게 맞춰 주어야 함.
pub const SOFTMAX_C: usize = 5;   // number of segments (C)
pub const SOFTMAX_M: usize = 65536; // table width (M) - must be a power of two/ limb bits = 16

pub const RMSNORM_C: usize = 1; 
pub const RMSNORM_M: usize = 65536; // 1 << 16

pub const GELU_C: usize = 1;
pub const GELU_M: usize = 65536; // 65536

pub const KBITS_C : usize = 2; // 16비트 전체 + 6비트 컷오프, 총 2개의 서브테이블
pub const KBITS_M : usize = 65536; // 2^16 = 65536, 16비트 전체를 채우는 테이블 크기

pub const FRACBITS_C : usize = 1; // 16비트 전체를 커버하는 테이블 하나
pub const FRACBITS_M : usize = 65536; // 2^16 = 65536, 16비트 전체를 채우는 테이블 크기