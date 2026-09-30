// AVX2 and FMA kernels. Fused multiply-adds round differently from the
// generic Go; the golden tests bound the difference against torch either way.

#include "textflag.h"

// func gemmAVX2(out *float32, os int, w, x *float32, xs int, b *float32, rows, inner, n int)
//
// out[r*os+j] = b[r] + sum_i w[r*inner+i] * x[i*xs+j] for j < n. Columns go
// 32 at a time in four accumulators, then 8, then the last n%8 through masked
// loads and stores; each weight is broadcast once per column block.
//
// DI out row, SI w row, DX x, R8 os bytes, R9 xs bytes, R10 rows left,
// R11 inner, R12 bytes in whole groups of 8, BX column offset, R13 w cursor,
// AX x cursor, CX inner count. Y5 holds the tail mask, Y6 the row's bias.
// The bias pointer is advanced in its argument slot for want of a register.
TEXT ·gemmAVX2(SB), NOSPLIT, $0-72
	MOVQ out+0(FP), DI
	MOVQ os+8(FP), R8
	MOVQ w+16(FP), SI
	MOVQ x+24(FP), DX
	MOVQ xs+32(FP), R9
	MOVQ rows+48(FP), R10
	MOVQ inner+56(FP), R11
	MOVQ n+64(FP), R12
	SHLQ $2, R8
	SHLQ $2, R9

	// Tail mask: the first n%8 lanes set.
	MOVQ    R12, AX
	ANDQ    $7, AX
	NEGQ    AX
	LEAQ    masktab<>+32(SB), CX
	VMOVUPS (CX)(AX*4), Y5

	ANDQ $-8, R12
	SHLQ $2, R12

row:
	MOVQ         b+40(FP), CX
	VBROADCASTSS (CX), Y6
	ADDQ         $4, b+40(FP)
	XORQ         BX, BX

col32:
	LEAQ    128(BX), CX
	CMPQ    CX, R12
	JGT     col8
	VMOVAPS Y6, Y0
	VMOVAPS Y6, Y1
	VMOVAPS Y6, Y2
	VMOVAPS Y6, Y3
	MOVQ    SI, R13
	LEAQ    (DX)(BX*1), AX
	MOVQ    R11, CX

inner32:
	VBROADCASTSS (R13), Y4
	VFMADD231PS  (AX), Y4, Y0
	VFMADD231PS  32(AX), Y4, Y1
	VFMADD231PS  64(AX), Y4, Y2
	VFMADD231PS  96(AX), Y4, Y3
	ADDQ         $4, R13
	ADDQ         R9, AX
	DECQ         CX
	JNZ          inner32
	VMOVUPS      Y0, (DI)(BX*1)
	VMOVUPS      Y1, 32(DI)(BX*1)
	VMOVUPS      Y2, 64(DI)(BX*1)
	VMOVUPS      Y3, 96(DI)(BX*1)
	ADDQ         $128, BX
	JMP          col32

col8:
	CMPQ    BX, R12
	JGE     tail
	VMOVAPS Y6, Y0
	MOVQ    SI, R13
	LEAQ    (DX)(BX*1), AX
	MOVQ    R11, CX

inner8:
	VBROADCASTSS (R13), Y4
	VFMADD231PS  (AX), Y4, Y0
	ADDQ         $4, R13
	ADDQ         R9, AX
	DECQ         CX
	JNZ          inner8
	VMOVUPS      Y0, (DI)(BX*1)
	ADDQ         $32, BX
	JMP          col8

tail:
	MOVQ    n+64(FP), AX
	ANDQ    $7, AX
	JZ      nextrow
	VMOVAPS Y6, Y0
	MOVQ    SI, R13
	LEAQ    (DX)(BX*1), AX
	MOVQ    R11, CX

innertail:
	VBROADCASTSS (R13), Y4
	VMASKMOVPS   (AX), Y5, Y7
	VFMADD231PS  Y7, Y4, Y0
	ADDQ         $4, R13
	ADDQ         R9, AX
	DECQ         CX
	JNZ          innertail
	VMASKMOVPS   Y0, Y5, (DI)(BX*1)

nextrow:
	ADDQ R8, DI
	LEAQ (SI)(R11*4), SI
	DECQ R10
	JNZ  row
	VZEROUPPER
	RET

// func dwconvAVX2(out *float32, os int, x0, x1, x2 *float32, xs int, w, b *float32, ch, n int)
//
// The GT block's depthwise 3x3: for each channel c and column j < n,
// out[c*os+j] = b[c] + sum over frame t and tap k of w[c*9+t*3+k] * xt[c*xs+j+k],
// where x0, x1, x2 are the frames at t-2d, t-d and t, each row zero-padded by
// one on the left. The nine weights stay in Y7-Y15 for the whole channel.
//
// DI out row, AX BX CX frame rows, SI weights, DX bias, R8 channels left,
// R9 bytes in whole groups of 8, R10 column offset, R11 xs bytes, R12 os bytes.
TEXT ·dwconvAVX2(SB), NOSPLIT, $0-80
	MOVQ out+0(FP), DI
	MOVQ os+8(FP), R12
	MOVQ x0+16(FP), AX
	MOVQ x1+24(FP), BX
	MOVQ x2+32(FP), CX
	MOVQ xs+40(FP), R11
	MOVQ w+48(FP), SI
	MOVQ b+56(FP), DX
	MOVQ ch+64(FP), R8
	MOVQ n+72(FP), R9
	SHLQ $2, R12
	SHLQ $2, R11

	MOVQ    R9, R13
	ANDQ    $7, R13
	NEGQ    R13
	LEAQ    masktab<>+32(SB), R10
	VMOVUPS (R10)(R13*4), Y5
	ANDQ    $-8, R9
	SHLQ    $2, R9

dwch:
	VBROADCASTSS (DX), Y6
	VBROADCASTSS 0(SI), Y7
	VBROADCASTSS 4(SI), Y8
	VBROADCASTSS 8(SI), Y9
	VBROADCASTSS 12(SI), Y10
	VBROADCASTSS 16(SI), Y11
	VBROADCASTSS 20(SI), Y12
	VBROADCASTSS 24(SI), Y13
	VBROADCASTSS 28(SI), Y14
	VBROADCASTSS 32(SI), Y15
	XORQ         R10, R10

dw32:
	LEAQ    128(R10), R13
	CMPQ    R13, R9
	JGT     dw8
	VMOVAPS Y6, Y0
	VMOVAPS Y6, Y1
	VMOVAPS Y6, Y2
	VMOVAPS Y6, Y3
	VFMADD231PS 0(AX)(R10*1), Y7, Y0
	VFMADD231PS 32(AX)(R10*1), Y7, Y1
	VFMADD231PS 64(AX)(R10*1), Y7, Y2
	VFMADD231PS 96(AX)(R10*1), Y7, Y3
	VFMADD231PS 4(AX)(R10*1), Y8, Y0
	VFMADD231PS 36(AX)(R10*1), Y8, Y1
	VFMADD231PS 68(AX)(R10*1), Y8, Y2
	VFMADD231PS 100(AX)(R10*1), Y8, Y3
	VFMADD231PS 8(AX)(R10*1), Y9, Y0
	VFMADD231PS 40(AX)(R10*1), Y9, Y1
	VFMADD231PS 72(AX)(R10*1), Y9, Y2
	VFMADD231PS 104(AX)(R10*1), Y9, Y3
	VFMADD231PS 0(BX)(R10*1), Y10, Y0
	VFMADD231PS 32(BX)(R10*1), Y10, Y1
	VFMADD231PS 64(BX)(R10*1), Y10, Y2
	VFMADD231PS 96(BX)(R10*1), Y10, Y3
	VFMADD231PS 4(BX)(R10*1), Y11, Y0
	VFMADD231PS 36(BX)(R10*1), Y11, Y1
	VFMADD231PS 68(BX)(R10*1), Y11, Y2
	VFMADD231PS 100(BX)(R10*1), Y11, Y3
	VFMADD231PS 8(BX)(R10*1), Y12, Y0
	VFMADD231PS 40(BX)(R10*1), Y12, Y1
	VFMADD231PS 72(BX)(R10*1), Y12, Y2
	VFMADD231PS 104(BX)(R10*1), Y12, Y3
	VFMADD231PS 0(CX)(R10*1), Y13, Y0
	VFMADD231PS 32(CX)(R10*1), Y13, Y1
	VFMADD231PS 64(CX)(R10*1), Y13, Y2
	VFMADD231PS 96(CX)(R10*1), Y13, Y3
	VFMADD231PS 4(CX)(R10*1), Y14, Y0
	VFMADD231PS 36(CX)(R10*1), Y14, Y1
	VFMADD231PS 68(CX)(R10*1), Y14, Y2
	VFMADD231PS 100(CX)(R10*1), Y14, Y3
	VFMADD231PS 8(CX)(R10*1), Y15, Y0
	VFMADD231PS 40(CX)(R10*1), Y15, Y1
	VFMADD231PS 72(CX)(R10*1), Y15, Y2
	VFMADD231PS 104(CX)(R10*1), Y15, Y3
	VMOVUPS Y0, (DI)(R10*1)
	VMOVUPS Y1, 32(DI)(R10*1)
	VMOVUPS Y2, 64(DI)(R10*1)
	VMOVUPS Y3, 96(DI)(R10*1)
	ADDQ    $128, R10
	JMP     dw32

dw8:
	CMPQ    R10, R9
	JGE     dwtail
	VMOVAPS Y6, Y0
	VFMADD231PS 0(AX)(R10*1), Y7, Y0
	VFMADD231PS 4(AX)(R10*1), Y8, Y0
	VFMADD231PS 8(AX)(R10*1), Y9, Y0
	VFMADD231PS 0(BX)(R10*1), Y10, Y0
	VFMADD231PS 4(BX)(R10*1), Y11, Y0
	VFMADD231PS 8(BX)(R10*1), Y12, Y0
	VFMADD231PS 0(CX)(R10*1), Y13, Y0
	VFMADD231PS 4(CX)(R10*1), Y14, Y0
	VFMADD231PS 8(CX)(R10*1), Y15, Y0
	VMOVUPS Y0, (DI)(R10*1)
	ADDQ    $32, R10
	JMP     dw8

dwtail:
	MOVQ    n+72(FP), R13
	ANDQ    $7, R13
	JZ      dwnext
	VMOVAPS Y6, Y0
	VMASKMOVPS 0(AX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y7, Y0
	VMASKMOVPS 4(AX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y8, Y0
	VMASKMOVPS 8(AX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y9, Y0
	VMASKMOVPS 0(BX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y10, Y0
	VMASKMOVPS 4(BX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y11, Y0
	VMASKMOVPS 8(BX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y12, Y0
	VMASKMOVPS 0(CX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y13, Y0
	VMASKMOVPS 4(CX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y14, Y0
	VMASKMOVPS 8(CX)(R10*1), Y5, Y4
	VFMADD231PS Y4, Y15, Y0
	VMASKMOVPS Y0, Y5, (DI)(R10*1)

dwnext:
	ADDQ R12, DI
	ADDQ R11, AX
	ADDQ R11, BX
	ADDQ R11, CX
	ADDQ $36, SI
	ADDQ $4, DX
	DECQ R8
	JNZ  dwch
	VZEROUPPER
	RET

// func vecPReLUAVX2(x *float32, n int, alpha float32)
//
// In place, n a multiple of 8: x = max(x, 0) + alpha*min(x, 0).
TEXT ·vecPReLUAVX2(SB), NOSPLIT, $0-20
	MOVQ         x+0(FP), DI
	MOVQ         n+8(FP), CX
	VBROADCASTSS alpha+16(FP), Y1
	VXORPS       Y2, Y2, Y2
	SHRQ         $3, CX

preluloop:
	VMOVUPS     (DI), Y0
	VMINPS      Y2, Y0, Y3
	VMAXPS      Y2, Y0, Y0
	VFMADD231PS Y3, Y1, Y0
	VMOVUPS     Y0, (DI)
	ADDQ        $32, DI
	DECQ        CX
	JNZ         preluloop
	VZEROUPPER
	RET

// EXP sets Y0 = exp(Y0), clobbering Y1-Y3: the same range reduction and
// polynomial as expf: t = x*log2e + 1.5*2^23 leaves k = round(x*log2e) in
// the low mantissa bits, r = x - k*ln2, and the result is poly(r) * 2^k.
#define EXP \
	VMINPS      hi<>(SB), Y0, Y0; \
	VMAXPS      lo<>(SB), Y0, Y0; \
	VMOVUPS     log2e<>(SB), Y1; \
	VFMADD213PS magic<>(SB), Y0, Y1; \
	VPSUBD      magicbits<>(SB), Y1, Y2; \
	VSUBPS      magic<>(SB), Y1, Y1; \
	VFNMADD231PS ln2hi<>(SB), Y1, Y0; \
	VFNMADD231PS ln2lo<>(SB), Y1, Y0; \
	VMOVUPS     c6<>(SB), Y3; \
	VFMADD213PS c5<>(SB), Y0, Y3; \
	VFMADD213PS c4<>(SB), Y0, Y3; \
	VFMADD213PS c3<>(SB), Y0, Y3; \
	VFMADD213PS half<>(SB), Y0, Y3; \
	VFMADD213PS one<>(SB), Y0, Y3; \
	VFMADD213PS one<>(SB), Y0, Y3; \
	VPADDD      bias<>(SB), Y2, Y2; \
	VPSLLD      $23, Y2, Y2; \
	VMULPS      Y2, Y3, Y0

// func vecSigmoidAVX2(dst, src *float32, n int)
TEXT ·vecSigmoidAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $3, CX

sigloop:
	VMOVUPS (SI), Y0
	VXORPS  sign<>(SB), Y0, Y0
	EXP
	VADDPS  one<>(SB), Y0, Y0
	VMOVUPS one<>(SB), Y1
	VDIVPS  Y0, Y1, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	DECQ    CX
	JNZ     sigloop
	VZEROUPPER
	RET

// func vecTanhAVX2(dst, src *float32, n int)
TEXT ·vecTanhAVX2(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	SHRQ $3, CX

tanhloop:
	VMOVUPS (SI), Y0
	VMULPS  neg2<>(SB), Y0, Y0
	EXP
	VADDPS  one<>(SB), Y0, Y0
	VMOVUPS two<>(SB), Y1
	VDIVPS  Y0, Y1, Y0
	VSUBPS  one<>(SB), Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	DECQ    CX
	JNZ     tanhloop
	VZEROUPPER
	RET

DATA hi<>+0(SB)/4, $0x42b00000
DATA hi<>+4(SB)/4, $0x42b00000
DATA hi<>+8(SB)/4, $0x42b00000
DATA hi<>+12(SB)/4, $0x42b00000
DATA hi<>+16(SB)/4, $0x42b00000
DATA hi<>+20(SB)/4, $0x42b00000
DATA hi<>+24(SB)/4, $0x42b00000
DATA hi<>+28(SB)/4, $0x42b00000
GLOBL hi<>(SB), RODATA|NOPTR, $32

DATA lo<>+0(SB)/4, $0xc2ae0000
DATA lo<>+4(SB)/4, $0xc2ae0000
DATA lo<>+8(SB)/4, $0xc2ae0000
DATA lo<>+12(SB)/4, $0xc2ae0000
DATA lo<>+16(SB)/4, $0xc2ae0000
DATA lo<>+20(SB)/4, $0xc2ae0000
DATA lo<>+24(SB)/4, $0xc2ae0000
DATA lo<>+28(SB)/4, $0xc2ae0000
GLOBL lo<>(SB), RODATA|NOPTR, $32

DATA log2e<>+0(SB)/4, $0x3fb8aa3b
DATA log2e<>+4(SB)/4, $0x3fb8aa3b
DATA log2e<>+8(SB)/4, $0x3fb8aa3b
DATA log2e<>+12(SB)/4, $0x3fb8aa3b
DATA log2e<>+16(SB)/4, $0x3fb8aa3b
DATA log2e<>+20(SB)/4, $0x3fb8aa3b
DATA log2e<>+24(SB)/4, $0x3fb8aa3b
DATA log2e<>+28(SB)/4, $0x3fb8aa3b
GLOBL log2e<>(SB), RODATA|NOPTR, $32

DATA magic<>+0(SB)/4, $0x4b400000
DATA magic<>+4(SB)/4, $0x4b400000
DATA magic<>+8(SB)/4, $0x4b400000
DATA magic<>+12(SB)/4, $0x4b400000
DATA magic<>+16(SB)/4, $0x4b400000
DATA magic<>+20(SB)/4, $0x4b400000
DATA magic<>+24(SB)/4, $0x4b400000
DATA magic<>+28(SB)/4, $0x4b400000
GLOBL magic<>(SB), RODATA|NOPTR, $32

DATA magicbits<>+0(SB)/4, $0x4b400000
DATA magicbits<>+4(SB)/4, $0x4b400000
DATA magicbits<>+8(SB)/4, $0x4b400000
DATA magicbits<>+12(SB)/4, $0x4b400000
DATA magicbits<>+16(SB)/4, $0x4b400000
DATA magicbits<>+20(SB)/4, $0x4b400000
DATA magicbits<>+24(SB)/4, $0x4b400000
DATA magicbits<>+28(SB)/4, $0x4b400000
GLOBL magicbits<>(SB), RODATA|NOPTR, $32

DATA ln2hi<>+0(SB)/4, $0x3f317200
DATA ln2hi<>+4(SB)/4, $0x3f317200
DATA ln2hi<>+8(SB)/4, $0x3f317200
DATA ln2hi<>+12(SB)/4, $0x3f317200
DATA ln2hi<>+16(SB)/4, $0x3f317200
DATA ln2hi<>+20(SB)/4, $0x3f317200
DATA ln2hi<>+24(SB)/4, $0x3f317200
DATA ln2hi<>+28(SB)/4, $0x3f317200
GLOBL ln2hi<>(SB), RODATA|NOPTR, $32

DATA ln2lo<>+0(SB)/4, $0x35bfbe8e
DATA ln2lo<>+4(SB)/4, $0x35bfbe8e
DATA ln2lo<>+8(SB)/4, $0x35bfbe8e
DATA ln2lo<>+12(SB)/4, $0x35bfbe8e
DATA ln2lo<>+16(SB)/4, $0x35bfbe8e
DATA ln2lo<>+20(SB)/4, $0x35bfbe8e
DATA ln2lo<>+24(SB)/4, $0x35bfbe8e
DATA ln2lo<>+28(SB)/4, $0x35bfbe8e
GLOBL ln2lo<>(SB), RODATA|NOPTR, $32

DATA c6<>+0(SB)/4, $0x3ab60b61
DATA c6<>+4(SB)/4, $0x3ab60b61
DATA c6<>+8(SB)/4, $0x3ab60b61
DATA c6<>+12(SB)/4, $0x3ab60b61
DATA c6<>+16(SB)/4, $0x3ab60b61
DATA c6<>+20(SB)/4, $0x3ab60b61
DATA c6<>+24(SB)/4, $0x3ab60b61
DATA c6<>+28(SB)/4, $0x3ab60b61
GLOBL c6<>(SB), RODATA|NOPTR, $32

DATA c5<>+0(SB)/4, $0x3c088889
DATA c5<>+4(SB)/4, $0x3c088889
DATA c5<>+8(SB)/4, $0x3c088889
DATA c5<>+12(SB)/4, $0x3c088889
DATA c5<>+16(SB)/4, $0x3c088889
DATA c5<>+20(SB)/4, $0x3c088889
DATA c5<>+24(SB)/4, $0x3c088889
DATA c5<>+28(SB)/4, $0x3c088889
GLOBL c5<>(SB), RODATA|NOPTR, $32

DATA c4<>+0(SB)/4, $0x3d2aaaab
DATA c4<>+4(SB)/4, $0x3d2aaaab
DATA c4<>+8(SB)/4, $0x3d2aaaab
DATA c4<>+12(SB)/4, $0x3d2aaaab
DATA c4<>+16(SB)/4, $0x3d2aaaab
DATA c4<>+20(SB)/4, $0x3d2aaaab
DATA c4<>+24(SB)/4, $0x3d2aaaab
DATA c4<>+28(SB)/4, $0x3d2aaaab
GLOBL c4<>(SB), RODATA|NOPTR, $32

DATA c3<>+0(SB)/4, $0x3e2aaaab
DATA c3<>+4(SB)/4, $0x3e2aaaab
DATA c3<>+8(SB)/4, $0x3e2aaaab
DATA c3<>+12(SB)/4, $0x3e2aaaab
DATA c3<>+16(SB)/4, $0x3e2aaaab
DATA c3<>+20(SB)/4, $0x3e2aaaab
DATA c3<>+24(SB)/4, $0x3e2aaaab
DATA c3<>+28(SB)/4, $0x3e2aaaab
GLOBL c3<>(SB), RODATA|NOPTR, $32

DATA half<>+0(SB)/4, $0x3f000000
DATA half<>+4(SB)/4, $0x3f000000
DATA half<>+8(SB)/4, $0x3f000000
DATA half<>+12(SB)/4, $0x3f000000
DATA half<>+16(SB)/4, $0x3f000000
DATA half<>+20(SB)/4, $0x3f000000
DATA half<>+24(SB)/4, $0x3f000000
DATA half<>+28(SB)/4, $0x3f000000
GLOBL half<>(SB), RODATA|NOPTR, $32

DATA one<>+0(SB)/4, $0x3f800000
DATA one<>+4(SB)/4, $0x3f800000
DATA one<>+8(SB)/4, $0x3f800000
DATA one<>+12(SB)/4, $0x3f800000
DATA one<>+16(SB)/4, $0x3f800000
DATA one<>+20(SB)/4, $0x3f800000
DATA one<>+24(SB)/4, $0x3f800000
DATA one<>+28(SB)/4, $0x3f800000
GLOBL one<>(SB), RODATA|NOPTR, $32

DATA two<>+0(SB)/4, $0x40000000
DATA two<>+4(SB)/4, $0x40000000
DATA two<>+8(SB)/4, $0x40000000
DATA two<>+12(SB)/4, $0x40000000
DATA two<>+16(SB)/4, $0x40000000
DATA two<>+20(SB)/4, $0x40000000
DATA two<>+24(SB)/4, $0x40000000
DATA two<>+28(SB)/4, $0x40000000
GLOBL two<>(SB), RODATA|NOPTR, $32

DATA neg2<>+0(SB)/4, $0xc0000000
DATA neg2<>+4(SB)/4, $0xc0000000
DATA neg2<>+8(SB)/4, $0xc0000000
DATA neg2<>+12(SB)/4, $0xc0000000
DATA neg2<>+16(SB)/4, $0xc0000000
DATA neg2<>+20(SB)/4, $0xc0000000
DATA neg2<>+24(SB)/4, $0xc0000000
DATA neg2<>+28(SB)/4, $0xc0000000
GLOBL neg2<>(SB), RODATA|NOPTR, $32

DATA bias<>+0(SB)/4, $0x0000007f
DATA bias<>+4(SB)/4, $0x0000007f
DATA bias<>+8(SB)/4, $0x0000007f
DATA bias<>+12(SB)/4, $0x0000007f
DATA bias<>+16(SB)/4, $0x0000007f
DATA bias<>+20(SB)/4, $0x0000007f
DATA bias<>+24(SB)/4, $0x0000007f
DATA bias<>+28(SB)/4, $0x0000007f
GLOBL bias<>(SB), RODATA|NOPTR, $32

DATA sign<>+0(SB)/4, $0x80000000
DATA sign<>+4(SB)/4, $0x80000000
DATA sign<>+8(SB)/4, $0x80000000
DATA sign<>+12(SB)/4, $0x80000000
DATA sign<>+16(SB)/4, $0x80000000
DATA sign<>+20(SB)/4, $0x80000000
DATA sign<>+24(SB)/4, $0x80000000
DATA sign<>+28(SB)/4, $0x80000000
GLOBL sign<>(SB), RODATA|NOPTR, $32

DATA masktab<>+0(SB)/4, $0xffffffff
DATA masktab<>+4(SB)/4, $0xffffffff
DATA masktab<>+8(SB)/4, $0xffffffff
DATA masktab<>+12(SB)/4, $0xffffffff
DATA masktab<>+16(SB)/4, $0xffffffff
DATA masktab<>+20(SB)/4, $0xffffffff
DATA masktab<>+24(SB)/4, $0xffffffff
DATA masktab<>+28(SB)/4, $0xffffffff
DATA masktab<>+32(SB)/4, $0x00000000
DATA masktab<>+36(SB)/4, $0x00000000
DATA masktab<>+40(SB)/4, $0x00000000
DATA masktab<>+44(SB)/4, $0x00000000
DATA masktab<>+48(SB)/4, $0x00000000
DATA masktab<>+52(SB)/4, $0x00000000
DATA masktab<>+56(SB)/4, $0x00000000
DATA masktab<>+60(SB)/4, $0x00000000
GLOBL masktab<>(SB), RODATA|NOPTR, $64
