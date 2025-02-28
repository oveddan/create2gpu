/**
* Based on the following, with small tweaks and optimizations:
*
* https://github.com/lwYeo/SoliditySHA3Miner/blob/master/SoliditySHA3Miner/
*   Miner/Kernels/OpenCL/sha3KingKernel.cl
*
* Originally modified for openCL processing by lwYeo
*
* Original implementor: David Leon Gil
*
* License: CC0, attribution kindly requested. Blame taken too, but not
* liability.
*/

// Add this function declaration at the top of the file, before it's used
char get_hex_char(uchar byte, bool high_nibble);

// Add this at the top of the file to make it more compatible with Metal
#ifdef __APPLE__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Add this at the top of the file after the existing includes
#define PREFIX_0 '1'
#define PREFIX_1 '1'
#define PREFIX_2 '1'
#define HAS_PREFIX 1

// Add this at the top of the file
#define DEBUG_MODE 1

/******** Keccak-f[1600] (for finding efficient Ethereum addresses) ********/

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_AMD   2

#ifndef PLATFORM
# define PLATFORM       OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
# pragma OPENCL EXTENSION   cl_amd_media_ops : enable
#endif

typedef union _nonce_t
{
  ulong   uint64_t;
  uint    uint32_t[2];
  uchar   uint8_t[8];
} nonce_t;

#if PLATFORM == OPENCL_PLATFORM_AMD
static inline ulong rol(const ulong x, const uint s)
{
  uint2 output;
  uint2 x2 = as_uint2(x);

  output = (s > 32u) ? amd_bitalign((x2).yx, (x2).xy, 64u - s) : amd_bitalign((x2).xy, (x2).yx, 32u - s);
  return as_ulong(output);
}
#else
#define rol(x, s) (((x) << s) | ((x) >> (64u - s)))
#endif

#define rol1(x) rol(x, 1u)

#define theta_(m, n, o) \
t = b[m] ^ rol1(b[n]); \
a[o + 0] ^= t; \
a[o + 5] ^= t; \
a[o + 10] ^= t; \
a[o + 15] ^= t; \
a[o + 20] ^= t; \

#define theta() \
b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]; \
b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]; \
b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]; \
b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]; \
b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]; \
theta_(4, 1, 0); \
theta_(0, 2, 1); \
theta_(1, 3, 2); \
theta_(2, 4, 3); \
theta_(3, 0, 4);

#define rhoPi_(m, n) t = b[0]; b[0] = a[m]; a[m] = rol(t, n); \

#define rhoPi() t = a[1]; b[0] = a[10]; a[10] = rol1(t); \
rhoPi_(7, 3); \
rhoPi_(11, 6); \
rhoPi_(17, 10); \
rhoPi_(18, 15); \
rhoPi_(3, 21); \
rhoPi_(5, 28); \
rhoPi_(16, 36); \
rhoPi_(8, 45); \
rhoPi_(21, 55); \
rhoPi_(24, 2); \
rhoPi_(4, 14); \
rhoPi_(15, 27); \
rhoPi_(23, 41); \
rhoPi_(19, 56); \
rhoPi_(13, 8); \
rhoPi_(12, 25); \
rhoPi_(2, 43); \
rhoPi_(20, 62); \
rhoPi_(14, 18); \
rhoPi_(22, 39); \
rhoPi_(9, 61); \
rhoPi_(6, 20); \
rhoPi_(1, 44);

#define chi_(n) \
b[0] = a[n + 0]; \
b[1] = a[n + 1]; \
b[2] = a[n + 2]; \
b[3] = a[n + 3]; \
b[4] = a[n + 4]; \
a[n + 0] = b[0] ^ ((~b[1]) & b[2]); \
a[n + 1] = b[1] ^ ((~b[2]) & b[3]); \
a[n + 2] = b[2] ^ ((~b[3]) & b[4]); \
a[n + 3] = b[3] ^ ((~b[4]) & b[0]); \
a[n + 4] = b[4] ^ ((~b[0]) & b[1]);

#define chi() chi_(0); chi_(5); chi_(10); chi_(15); chi_(20);

#define iota(x) a[0] ^= x;

#define iteration(x) theta(); rhoPi(); chi(); iota(x);

static inline void keccakf(ulong *a)
{
  ulong b[5];
  ulong t;

  iteration(0x0000000000000001); // iteration 1
  iteration(0x0000000000008082); // iteration 2
  iteration(0x800000000000808a); // iteration 3
  iteration(0x8000000080008000); // iteration 4
  iteration(0x000000000000808b); // iteration 5
  iteration(0x0000000080000001); // iteration 6
  iteration(0x8000000080008081); // iteration 7
  iteration(0x8000000000008009); // iteration 8
  iteration(0x000000000000008a); // iteration 9
  iteration(0x0000000000000088); // iteration 10
  iteration(0x0000000080008009); // iteration 11
  iteration(0x000000008000000a); // iteration 12
  iteration(0x000000008000808b); // iteration 13
  iteration(0x800000000000008b); // iteration 14
  iteration(0x8000000000008089); // iteration 15
  iteration(0x8000000000008003); // iteration 16
  iteration(0x8000000000008002); // iteration 17
  iteration(0x8000000000000080); // iteration 18
  iteration(0x000000000000800a); // iteration 19
  iteration(0x800000008000000a); // iteration 20
  iteration(0x8000000080008081); // iteration 21
  iteration(0x8000000000008080); // iteration 22
  iteration(0x0000000080000001); // iteration 23

  // iteration 24 (partial)

#define o ((uint *)(a))
  // Theta (partial)
  b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
  b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
  b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
  b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
  b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

  a[0] ^= b[4] ^ rol1(b[1]);
  a[6] ^= b[0] ^ rol1(b[2]);
  a[12] ^= b[1] ^ rol1(b[3]);
  a[18] ^= b[2] ^ rol1(b[4]);
  a[24] ^= b[3] ^ rol1(b[0]);

  // Rho Pi (partial)
  o[3] = (o[13] >> 20) | (o[12] << 12);
  a[2] = rol(a[12], 43);
  a[3] = rol(a[18], 21);
  a[4] = rol(a[24], 14);

  // Chi (partial)
  o[3] ^= ((~o[5]) & o[7]);
  o[4] ^= ((~o[6]) & o[8]);
  o[5] ^= ((~o[7]) & o[9]);
  o[6] ^= ((~o[8]) & o[0]);
  o[7] ^= ((~o[9]) & o[1]);
#undef o
}

#define hasTotal(d, S) ( \
  ((d[0] == S)) + ((d[1] == S)) + ((d[2] == S)) + ((d[3] == S)) + \
  ((d[4] == S)) + (!(d[5] == S)) + (!(d[6] == S)) + (!(d[7] == S)) + \
  ((d[8] == S)) + (!(d[9] == S)) + (!(d[10] == S)) + (!(d[11] == S)) + \
  ((d[12] == S)) + (!(d[13] == S)) + (!(d[14] == S)) + (!(d[15] == S)) + \
  ((d[16] == S)) + (!(d[17] == S)) + (!(d[18] == S)) + (!(d[19] == S)) \
>= TOTAL_ZEROES)

#if LEADING_ZEROES == 8
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1]))
// #elif LEADING_ZEROES == 7
// #define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x00ffffffu))
// #elif LEADING_ZEROES == 6
// #define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x0000ffffu))
// #elif LEADING_ZEROES == 5
// #define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x000000ffu))
// #elif LEADING_ZEROES == 4
// #define hasLeading(d) ((((uint*)d)[0] == 0x61616161u))
// #elif LEADING_ZEROES == 3
// #define hasLeading(d) (!(((uint*)d)[0] & 0x00ffffffu))
// #elif LEADING_ZEROES == 2
// #define hasLeading(d) (!(((uint*)d)[0] & 0x0000ffffu))
// #elif LEADING_ZEROES == 1
// #define hasLeading(d) (!(((uint*)d)[0] & 0x000000ffu))
#else
static inline bool hasLeading(uchar const *d)
{
  // For an address to start with "11" in hex:
  // We need to check if the first byte is 0x11 and the high nibble of the second byte is 0x1
  
  // Check if the first byte is 0x11
  if (d[0] != 0x11)
    return false;
    
  // Check if the high nibble of the second byte is 0x1
  if ((d[1] & 0xF0) != 0x10)
    return false;
    
  return true;
}
#endif

// Helper function to convert a nibble to its hex character
static inline char nibbleToHexChar(uchar nibble) {
  return nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
}

// Function to check if the address starts with a specific prefix and/or ends with a specific suffix
static inline bool matchesAddressCriteria(uchar const *d, __constant uchar const *prefix, int prefixLen, 
                                         __constant uchar const *suffix, int suffixLen) {
  // Check prefix if specified
  if (prefixLen > 0) {
    for (int i = 0; i < prefixLen; i++) {
      // Calculate which byte and nibble we need from d
      int byteIndex = i / 2;
      bool isHighNibble = (i % 2 == 0);
      
      // Extract the nibble from d
      uchar byte = d[byteIndex];
      uchar nibble = isHighNibble ? ((byte >> 4) & 0xF) : (byte & 0xF);
      
      // Convert nibble to hex character (always lowercase)
      char hexChar = nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
      
      // Compare with the prefix character
      if (hexChar != prefix[i]) {
        return false;
      }
    }
  }
  
  // Check suffix if specified
  if (suffixLen > 0) {
    for (int i = 0; i < suffixLen; i++) {
      // Calculate which byte and nibble we need from d
      // For suffix, we start from the end
      int addressLen = 40; // Length of address in hex chars
      int pos = addressLen - suffixLen + i;
      int byteIndex = pos / 2;
      bool isHighNibble = (pos % 2 == 0);
      
      // Extract the nibble from d
      uchar byte = d[byteIndex];
      uchar nibble = isHighNibble ? ((byte >> 4) & 0xF) : (byte & 0xF);
      
      // Convert nibble to hex character (always lowercase)
      char hexChar = nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
      
      // Compare with the suffix character
      if (hexChar != suffix[i]) {
        return false;
      }
    }
  }
  
  return true;
}

// Function to count leading 1s in an address
static inline int countLeadingOnes(uchar const *d) {
  int count = 0;
  
  // Check each nibble from the start
  for (int i = 0; i < 40; i++) { // 40 hex chars in an address
    // Calculate which byte and nibble we need
    int byteIndex = i / 2;
    bool isHighNibble = (i % 2 == 0);
    
    // Extract the nibble
    uchar byte = d[byteIndex];
    uchar nibble = isHighNibble ? ((byte >> 4) & 0xF) : (byte & 0xF);
    
    // Convert nibble to hex character
    char hexChar = nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
    
    // Check if it's a '1'
    if (hexChar == '1') {
      count++;
    } else {
      break; // Stop counting when we hit a non-1
    }
  }
  
  return count;
}

// Function to count trailing 1s in an address
static inline int countTrailingOnes(uchar const *d) {
  int count = 0;
  
  // Check each nibble from the end
  for (int i = 39; i >= 0; i--) { // 40 hex chars in an address
    // Calculate which byte and nibble we need
    int byteIndex = i / 2;
    bool isHighNibble = (i % 2 == 0);
    
    // Extract the nibble
    uchar byte = d[byteIndex];
    uchar nibble = isHighNibble ? ((byte >> 4) & 0xF) : (byte & 0xF);
    
    // Convert nibble to hex character
    char hexChar = nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
    
    // Check if it's a '1'
    if (hexChar == '1') {
      count++;
    } else {
      break; // Stop counting when we hit a non-1
    }
  }
  
  return count;
}

__kernel void hashMessage(
  __constant uchar const *d_message,
  __constant uint const *d_nonce,
  __global volatile ulong *restrict solutions,
  __global volatile uint *restrict has_solution,
  __global volatile uchar *restrict digest_output
) {
  ulong spongeBuffer[25];

#define sponge ((uchar *) spongeBuffer)
#define digest (sponge + 12)

  nonce_t nonce;

  // Initialize sponge with zeros
  for (int i = 0; i < 200; i++) {
    sponge[i] = 0;
  }

  // Start with 0xff prefix for CREATE2
  sponge[0] = 0xff;

  // Copy deployer address (factory address)
  for (int i = 0; i < 20; i++) {
    sponge[i + 1] = d_message[i];
  }

  // populate the nonce for the salt
  nonce.uint32_t[0] = get_global_id(0);
  nonce.uint32_t[1] = d_nonce[0];

  // Copy the salt (32 bytes, with the nonce at the end)
  for (int i = 0; i < 24; i++) {
    sponge[i + 21] = 0; // First 24 bytes of salt are zeros
  }
  
  // Last 8 bytes of salt are the nonce
  sponge[21 + 24] = nonce.uint8_t[0];
  sponge[21 + 25] = nonce.uint8_t[1];
  sponge[21 + 26] = nonce.uint8_t[2];
  sponge[21 + 27] = nonce.uint8_t[3];
  sponge[21 + 28] = nonce.uint8_t[4];
  sponge[21 + 29] = nonce.uint8_t[5];
  sponge[21 + 30] = nonce.uint8_t[6];
  sponge[21 + 31] = nonce.uint8_t[7];

  // Copy the init code hash (32 bytes)
  for (int i = 0; i < 32; i++) {
    sponge[i + 53] = d_message[i + 20];
  }

  // Get the prefix and suffix to check for and their lengths
  int prefixLen = d_message[52];
  __constant uchar const *prefix = &d_message[53];
  int suffixLen = d_message[53 + prefixLen];
  __constant uchar const *suffix = &d_message[54 + prefixLen];

  // begin padding based on message length (0xff + 20 bytes + 32 bytes + 32 bytes = 85 bytes)
  sponge[85] = 0x01;

  // fill padding
  for (int i = 86; i < 135; i++) {
    sponge[i] = 0;
  }

  // end padding
  sponge[135] = 0x80;

  // Apply keccakf
  keccakf(spongeBuffer);

  // Get the minimum required leading and trailing 1s
  int minLeadingOnes = d_message[52];
  int minTrailingOnes = d_message[53];
  int bestScore = d_message[54]; // Get the current best score

  // Count leading and trailing 1s in the address
  int leadingOnes = countLeadingOnes(digest);
  int trailingOnes = countTrailingOnes(digest);
  int totalOnes = leadingOnes + trailingOnes;

  // Check if this address meets our criteria and is BETTER than the best score (not just equal)
  if (leadingOnes >= minLeadingOnes && trailingOnes >= minTrailingOnes && totalOnes > bestScore) {
    // Found a solution
    solutions[0] = nonce.uint64_t;
    solutions[1] = leadingOnes;
    solutions[2] = trailingOnes;
    has_solution[0] = 1;
    
    // Copy the digest to the output buffer
    for (int i = 0; i < 200; i++) {
      digest_output[i] = sponge[i];
    }
  }
}

// Helper function implementation at the end of the file
char get_hex_char(uchar byte, bool high_nibble) {
    uchar nibble = high_nibble ? (byte >> 4) : (byte & 0xF);
    return nibble < 10 ? '0' + nibble : 'a' + (nibble - 10);
}
