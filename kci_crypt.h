/* kci_crypt.h — the standard primitives behind KCI-Vec's encrypted files, shared by kci_vec.c and
 * kci_sparse.c: SHA-256 (FIPS 180-4), HMAC-SHA256, PBKDF2-HMAC-SHA256, AES-256 (FIPS 197) in CTR mode,
 * AES-256-CMAC (NIST SP 800-38B), and a source of random salt.
 *
 * (2026-11-19) Before: these lived inside kci_vec.c as static functions, so the sparse index
 * (kci_sparse.c, which does not include kci_vec) could not save an encrypted file without a second copy
 * of the same code. Now: moved here unchanged, byte for byte; both files include this header. Files
 * written before the move read as before (tests/test_crypto.c, tests/data/kcivec04_chacha.kv).
 * Everything is static: include it from one .c file per library, after <stdint.h>-style basics.
 * Not a public API — the names and signatures may change. */
#ifndef KCI_CRYPT_H
#define KCI_CRYPT_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ── SHA-256 (FIPS 180-4) → HMAC → PBKDF2: proper key derivation ──────────── */
typedef struct { uint32_t h[8]; uint64_t bits; uint8_t buf[64]; size_t n; } SHA256;
static const uint32_t SHA256_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define SHR(x,n) ((x)>>(n))
#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256_blk(SHA256*c,const uint8_t*p){
    uint32_t w[64],a,b,cc,d,e,f,g,h,t1,t2;
    for(int i=0;i<16;i++) w[i]=(uint32_t)p[i*4]<<24|p[i*4+1]<<16|p[i*4+2]<<8|p[i*4+3];
    for(int i=16;i<64;i++){ uint32_t s0=ROTR(w[i-15],7)^ROTR(w[i-15],18)^SHR(w[i-15],3);
        uint32_t s1=ROTR(w[i-2],17)^ROTR(w[i-2],19)^SHR(w[i-2],10); w[i]=w[i-16]+s0+w[i-7]+s1; }
    a=c->h[0];b=c->h[1];cc=c->h[2];d=c->h[3];e=c->h[4];f=c->h[5];g=c->h[6];h=c->h[7];
    for(int i=0;i<64;i++){ uint32_t S1=ROTR(e,6)^ROTR(e,11)^ROTR(e,25), ch=(e&f)^(~e&g);
        t1=h+S1+ch+SHA256_K[i]+w[i]; uint32_t S0=ROTR(a,2)^ROTR(a,13)^ROTR(a,22), maj=(a&b)^(a&cc)^(b&cc);
        t2=S0+maj; h=g;g=f;f=e;e=d+t1;d=cc;cc=b;b=a;a=t1+t2; }
    c->h[0]+=a;c->h[1]+=b;c->h[2]+=cc;c->h[3]+=d;c->h[4]+=e;c->h[5]+=f;c->h[6]+=g;c->h[7]+=h;
}
static void sha256_init(SHA256*c){ static const uint32_t H0[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h,H0,sizeof H0); c->bits=0; c->n=0; }
static void sha256_upd(SHA256*c,const uint8_t*d,size_t n){ c->bits+=(uint64_t)n*8;
    while(n){ size_t k=64-c->n; if(k>n)k=n; memcpy(c->buf+c->n,d,k); c->n+=k; d+=k; n-=k;
        if(c->n==64){ sha256_blk(c,c->buf); c->n=0; } } }
static void sha256_fin(SHA256*c,uint8_t out[32]){ uint64_t bits=c->bits; uint8_t pad=0x80;
    sha256_upd(c,&pad,1); uint8_t z=0; while(c->n!=56) sha256_upd(c,&z,1);
    uint8_t lb[8]; for(int i=0;i<8;i++) lb[i]=(uint8_t)(bits>>(56-i*8)); sha256_upd(c,lb,8);
    for(int i=0;i<8;i++){ out[i*4]=c->h[i]>>24;out[i*4+1]=c->h[i]>>16;out[i*4+2]=c->h[i]>>8;out[i*4+3]=c->h[i]; } }
static void hmac_sha256(const uint8_t*key,size_t kl,const uint8_t*msg,size_t ml,uint8_t out[32]){
    uint8_t k[64]={0}, ip[64], op[64], ih[32]; SHA256 c;
    if(kl>64){ sha256_init(&c); sha256_upd(&c,key,kl); sha256_fin(&c,k); } else memcpy(k,key,kl);
    for(int i=0;i<64;i++){ ip[i]=k[i]^0x36; op[i]=k[i]^0x5c; }
    sha256_init(&c); sha256_upd(&c,ip,64); sha256_upd(&c,msg,ml); sha256_fin(&c,ih);
    sha256_init(&c); sha256_upd(&c,op,64); sha256_upd(&c,ih,32); sha256_fin(&c,out);
}
/* PBKDF2-HMAC-SHA256, dkLen=32 (one block) */
static void pbkdf2(const uint8_t*pw,size_t pl,const uint8_t*salt,size_t sl,uint32_t iters,uint8_t out32[32]){
    uint8_t s[64],u[32],t[32]; size_t n=sl<60?sl:60; memcpy(s,salt,n);
    s[n]=0;s[n+1]=0;s[n+2]=0;s[n+3]=1;                      /* INT(1) big-endian */
    hmac_sha256(pw,pl,s,n+4,u); memcpy(t,u,32);
    for(uint32_t i=1;i<iters;i++){ hmac_sha256(pw,pl,u,32,u); for(int j=0;j<32;j++) t[j]^=u[j]; }
    memcpy(out32,t,32);
}
#define KV_KDF_ITERS 100000u
static void kdf_derive(const uint8_t *key, int keylen, const uint8_t *salt, int saltlen, uint8_t out32[32]){
    pbkdf2(key,(size_t)keylen,salt,(size_t)saltlen,KV_KDF_ITERS,out32);
}
/* Random bytes for a salt, from the operating system's generator: /dev/urandom, on Windows RtlGenRandom
 * (advapi32, which every program is linked with). Returns 0, or -1 with `out` zeroed when the system gives
 * none — the caller must then refuse to write the file.
 * (2026-11-19) Before: when /dev/urandom could not be read, the bytes were a SHA-256 of the clock, two
 * addresses and a counter. Windows has no /dev/urandom, so there every salt came from that fallback: unique in
 * practice, but guessable, and not what "random salt" promises. Now there is no fallback. */
#ifdef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__declspec(dllimport) unsigned char __stdcall SystemFunction036(void *buf, unsigned long len);
#endif
static int kv_randbytes(uint8_t *out, int n){
    if(n<=0) return -1;
#ifdef _WIN32
    if(SystemFunction036(out,(unsigned long)n)) return 0;
#else
    FILE *f=fopen("/dev/urandom","rb");
    if(f){ size_t r=fread(out,1,(size_t)n,f); fclose(f); if(r==(size_t)n) return 0; }
#endif
    memset(out,0,(size_t)n);
    return -1;
}

/* ── AES-256-CTR + AES-256-CMAC: snapshot encryption at rest (KCIVEC05) ────────
 * (2026-11-01) Before: ChaCha20 in plain C (0.65 GB/s on a Ryzen 7 5800H) and no
 * tamper detection: a damaged or altered file loaded as wrong ids / a broken
 * graph. Now: AES-256 (FIPS 197) in CTR mode (7.8 GB/s with AES-NI on the same
 * machine) and an AES-256-CMAC tag (NIST SP 800-38B) over the whole file, checked
 * before anything is parsed (encrypt-then-MAC). PBKDF2 gives a master key; two
 * HMAC-SHA256 labels split it into the encryption key and the MAC key (one key,
 * one use). The same construction as the author's encrypted KV store.
 * AES paths: AES-NI (x86, picked at run time, so a build without -maes still
 * uses it), the ARMv8 AES instructions (when compiled with them), else portable
 * table AES. A fast path is used only if it reproduces the FIPS-197 C.3 answer;
 * otherwise the portable one is. ChaCha20 above stays only to read KCIVEC04. */
static const uint8_t kv_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16 };
static inline uint8_t kv_xt(uint8_t x){ return (uint8_t)((x<<1)^((x>>7)*0x1b)); }

/* FIPS-197 key expansion: 15 round keys in standard byte order */
static void kv_aes_expand(const uint8_t key[32], uint8_t w[240]){
    uint8_t rcon=1;
    memcpy(w,key,32);
    for(int i=8;i<60;i++){
        uint8_t t[4]; memcpy(t,w+4*(i-1),4);
        if(i%8==0){ uint8_t t0=t[0];
            t[0]=kv_sbox[t[1]]^rcon; t[1]=kv_sbox[t[2]]; t[2]=kv_sbox[t[3]]; t[3]=kv_sbox[t0];
            rcon=kv_xt(rcon); }
        else if(i%8==4){ for(int k=0;k<4;k++) t[k]=kv_sbox[t[k]]; }
        for(int k=0;k<4;k++) w[4*i+k]=w[4*(i-8)+k]^t[k];
    }
}
/* portable AES-256 encryption of one block */
static void kv_aes_block_soft(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    uint8_t s[16], t[16];
    for(int i=0;i<16;i++) s[i]=in[i]^w[i];
    for(int r=1;r<=14;r++){
        for(int i=0;i<16;i++) t[i]=kv_sbox[s[i]];                                   /* SubBytes  */
        for(int c=0;c<4;c++) for(int q=0;q<4;q++) s[q+4*c]=t[q+4*((c+q)&3)];          /* ShiftRows */
        if(r!=14) for(int c=0;c<4;c++){                                              /* MixColumns */
            uint8_t *col=s+4*c, a0=col[0], a1=col[1], a2=col[2], a3=col[3], x=a0^a1^a2^a3;
            col[0]=a0^x^kv_xt(a0^a1); col[1]=a1^x^kv_xt(a1^a2);
            col[2]=a2^x^kv_xt(a2^a3); col[3]=a3^x^kv_xt(a3^a0); }
        for(int i=0;i<16;i++) s[i]^=w[16*r+i];
    }
    memcpy(out,s,16);
}

#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#define KV_AES_X86 1
#include <wmmintrin.h>
#include <emmintrin.h>
#define KV_AESNI_FN __attribute__((target("aes,sse2")))
KV_AESNI_FN static void kv_aes_block_ni(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    const __m128i *k=(const __m128i*)w;
    __m128i b=_mm_xor_si128(_mm_loadu_si128((const __m128i*)in), _mm_loadu_si128(k));
    for(int r=1;r<14;r++) b=_mm_aesenc_si128(b,_mm_loadu_si128(k+r));
    _mm_storeu_si128((__m128i*)out, _mm_aesenclast_si128(b,_mm_loadu_si128(k+14)));
}
/* CTR keystream XOR, 4 blocks in flight. Counter block = [nonce LE 8 | block LE 8]. */
KV_AESNI_FN static void kv_ctr_ni(const uint8_t w[240], uint64_t nonce, uint8_t *buf, size_t len){
    __m128i rk[15]; for(int i=0;i<15;i++) rk[i]=_mm_loadu_si128((const __m128i*)(w+16*i));
    __m128i ctr=_mm_set_epi64x(0,(long long)nonce); const __m128i one=_mm_set_epi64x(1,0);
    size_t pos=0;
    for(; pos+64<=len; pos+=64){
        __m128i c0=ctr, c1=_mm_add_epi64(c0,one), c2=_mm_add_epi64(c1,one), c3=_mm_add_epi64(c2,one);
        ctr=_mm_add_epi64(c3,one);
        c0=_mm_xor_si128(c0,rk[0]); c1=_mm_xor_si128(c1,rk[0]); c2=_mm_xor_si128(c2,rk[0]); c3=_mm_xor_si128(c3,rk[0]);
        for(int r=1;r<14;r++){ c0=_mm_aesenc_si128(c0,rk[r]); c1=_mm_aesenc_si128(c1,rk[r]);
                               c2=_mm_aesenc_si128(c2,rk[r]); c3=_mm_aesenc_si128(c3,rk[r]); }
        c0=_mm_aesenclast_si128(c0,rk[14]); c1=_mm_aesenclast_si128(c1,rk[14]);
        c2=_mm_aesenclast_si128(c2,rk[14]); c3=_mm_aesenclast_si128(c3,rk[14]);
        __m128i *p=(__m128i*)(buf+pos);
        _mm_storeu_si128(p+0,_mm_xor_si128(_mm_loadu_si128(p+0),c0));
        _mm_storeu_si128(p+1,_mm_xor_si128(_mm_loadu_si128(p+1),c1));
        _mm_storeu_si128(p+2,_mm_xor_si128(_mm_loadu_si128(p+2),c2));
        _mm_storeu_si128(p+3,_mm_xor_si128(_mm_loadu_si128(p+3),c3));
    }
    for(; pos<len; pos+=16){
        __m128i b=_mm_xor_si128(ctr,rk[0]); ctr=_mm_add_epi64(ctr,one);
        for(int r=1;r<14;r++) b=_mm_aesenc_si128(b,rk[r]);
        b=_mm_aesenclast_si128(b,rk[14]);
        uint8_t ks[16]; _mm_storeu_si128((__m128i*)ks,b);
        size_t c=len-pos<16?len-pos:16;
        for(size_t i=0;i<c;i++) buf[pos+i]^=ks[i];
    }
}
/* CBC-MAC chain over n whole blocks (CMAC body): x = E(x ^ block) */
KV_AESNI_FN static void kv_cbcmac_ni(const uint8_t w[240], uint8_t x[16], const uint8_t *p, size_t n){
    __m128i rk[15]; for(int i=0;i<15;i++) rk[i]=_mm_loadu_si128((const __m128i*)(w+16*i));
    __m128i s=_mm_loadu_si128((const __m128i*)x);
    for(size_t j=0;j<n;j++){
        s=_mm_xor_si128(_mm_xor_si128(s,_mm_loadu_si128((const __m128i*)(p+16*j))),rk[0]);
        for(int r=1;r<14;r++) s=_mm_aesenc_si128(s,rk[r]);
        s=_mm_aesenclast_si128(s,rk[14]);
    }
    _mm_storeu_si128((__m128i*)x,s);
}
#elif defined(__aarch64__) && (defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO))
#define KV_AES_ARM 1
/* AESE = AddRoundKey + SubBytes + ShiftRows; AESMC = MixColumns */
static void kv_aes_block_arm(const uint8_t w[240], const uint8_t in[16], uint8_t out[16]){
    uint8x16_t b=vld1q_u8(in);
    for(int r=0;r<13;r++) b=vaesmcq_u8(vaeseq_u8(b,vld1q_u8(w+16*r)));
    b=veorq_u8(vaeseq_u8(b,vld1q_u8(w+16*13)),vld1q_u8(w+16*14));
    vst1q_u8(out,b);
}
#endif

typedef struct { uint8_t rk[240]; int path; } KvAes;   /* path: 0 portable, 1 AES-NI, 2 ARMv8 */
static void kv_aes_block(const KvAes *a, const uint8_t in[16], uint8_t out[16]){
#if defined(KV_AES_X86)
    if(a->path==1){ kv_aes_block_ni(a->rk,in,out); return; }
#elif defined(KV_AES_ARM)
    if(a->path==2){ kv_aes_block_arm(a->rk,in,out); return; }
#endif
    kv_aes_block_soft(a->rk,in,out);
}
/* expand the key and pick the fastest path that gives the FIPS-197 C.3 answer */
static void kv_aes_init(KvAes *a, const uint8_t key[32]){
    kv_aes_expand(key,a->rk); a->path=0;
#if defined(KV_AES_X86) || defined(KV_AES_ARM)
    static const uint8_t pt[16]={0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff};
    static const uint8_t ct[16]={0x8e,0xa2,0xb7,0xca,0x51,0x67,0x45,0xbf,0xea,0xfc,0x49,0x90,0x4b,0x49,0x60,0x89};
    uint8_t k[32], w[240], o[16];
    for(int i=0;i<32;i++) k[i]=(uint8_t)i;
    kv_aes_expand(k,w);
#if defined(KV_AES_X86)
    if(__builtin_cpu_supports("aes")){ kv_aes_block_ni(w,pt,o); if(memcmp(o,ct,16)==0) a->path=1; }
#else
    kv_aes_block_arm(w,pt,o); if(memcmp(o,ct,16)==0) a->path=2;
#endif
#endif
}
/* XOR the AES-256-CTR keystream for `nonce` over buf (block counter from 0) */
static void kv_ctr(const KvAes *a, uint64_t nonce, uint8_t *buf, size_t len){
#if defined(KV_AES_X86)
    if(a->path==1){ kv_ctr_ni(a->rk,nonce,buf,len); return; }
#endif
    uint8_t cb[16], ks[16]; uint64_t blk=0;
    for(size_t pos=0; pos<len; pos+=16, blk++){
        for(int k=0;k<8;k++){ cb[k]=(uint8_t)(nonce>>(8*k)); cb[8+k]=(uint8_t)(blk>>(8*k)); }
        kv_aes_block(a,cb,ks);
        size_t c=len-pos<16?len-pos:16;
        for(size_t i=0;i<c;i++) buf[pos+i]^=ks[i];
    }
}

/* AES-256-CMAC (NIST SP 800-38B), streaming: the last full block is held back
 * until final, because it is padded with K1 instead of K2 */
typedef struct { KvAes a; uint8_t k1[16], k2[16], x[16], buf[16]; uint32_t fill; } KvCmac;
static void kv_cmac_dbl(const uint8_t in[16], uint8_t out[16]){
    uint8_t carry=(uint8_t)(in[0]>>7);
    for(int i=0;i<15;i++) out[i]=(uint8_t)((in[i]<<1)|(in[i+1]>>7));
    out[15]=(uint8_t)((in[15]<<1)^(carry*0x87));
}
static void kv_cmac_init(KvCmac *m, const uint8_t key[32]){
    memset(m,0,sizeof *m);
    kv_aes_init(&m->a,key);
    uint8_t l[16]={0}; kv_aes_block(&m->a,l,l);
    kv_cmac_dbl(l,m->k1); kv_cmac_dbl(m->k1,m->k2);
}
static void kv_cmac_absorb(KvCmac *m, const uint8_t *p, size_t nblk){
#if defined(KV_AES_X86)
    if(m->a.path==1){ kv_cbcmac_ni(m->a.rk,m->x,p,nblk); return; }
#endif
    for(size_t j=0;j<nblk;j++){ for(int i=0;i<16;i++) m->x[i]^=p[16*j+i]; kv_aes_block(&m->a,m->x,m->x); }
}
static void kv_cmac_update(KvCmac *m, const void *data, size_t len){
    const uint8_t *p=(const uint8_t*)data;
    while(len){
        if(m->fill==16){ kv_cmac_absorb(m,m->buf,1); m->fill=0; }   /* more data follows */
        if(m->fill==0 && len>16){                   /* whole blocks straight from the input */
            size_t n=(len-1)/16;                    /* keep at least 1 byte for the held block */
            kv_cmac_absorb(m,p,n); p+=16*n; len-=16*n;
            continue;
        }
        size_t take=16-m->fill; if(take>len) take=len;
        memcpy(m->buf+m->fill,p,take); m->fill+=(uint32_t)take; p+=take; len-=take;
    }
}
static void kv_cmac_final(KvCmac *m, uint8_t tag[16]){
    const uint8_t *k=m->k1;
    if(m->fill<16){ m->buf[m->fill]=0x80; memset(m->buf+m->fill+1,0,15-m->fill); k=m->k2; }
    for(int i=0;i<16;i++) m->x[i]^=m->buf[i]^k[i];
    kv_aes_block(&m->a,m->x,tag);
}
/* one key, one use: HMAC-SHA256(master, label) */
static void kv_subkey(const uint8_t master[32], const char *label, uint8_t out[32]){
    hmac_sha256(master,32,(const uint8_t*)label,strlen(label),out);
}

#endif /* KCI_CRYPT_H */
