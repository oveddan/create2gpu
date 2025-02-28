mod gpu;

extern crate byteorder;
extern crate console;
extern crate fs2;
extern crate hex;
extern crate itertools;
extern crate ocl;
extern crate ocl_extras;
extern crate rand;
extern crate rayon;
extern crate separator;
extern crate terminal_size;
extern crate tiny_keccak;

use hex::FromHex;

// Export only the gpu function
pub use gpu::gpu;

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

/// Requires three hex-encoded arguments: the address of the contract that will
/// be calling CREATE2, the address of the caller of said contract *(assuming
/// the contract calling CREATE2 has frontrunning protection in place - if not
/// applicable to your use-case you can set it to the null address)*, and the
/// keccak-256 hash of the bytecode that is provided by the contract calling
/// CREATE2 that will be used to initialize the new contract. An additional set
/// of three optional values may be provided: a device to target for OpenCL GPU
/// search, a threshold for leading zeroes to search for, and a threshold for
/// total zeroes to search for.
#[derive(Clone)]
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u32,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
    pub prefix: Option<String>,
    pub starts_with: String,  // Field for the prefix to search for
    pub ends_with: String,    // New field for the suffix to search for
    pub case_sensitive: bool, // Field for case-sensitive matching
}

/// Validate the provided arguments and construct the Config struct.
impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Self, &'static str> {
        // get args, skipping first arg (program name)
        args.next();

        let mut factory_address_string = match args.next() {
            Some(arg) => arg,
            None => return Err("didn't get a factory_address argument."),
        };

        let mut calling_address_string = match args.next() {
            Some(arg) => arg,
            None => return Err("didn't get a calling_address argument."),
        };

        let mut init_code_hash_string = match args.next() {
            Some(arg) => arg,
            None => return Err("didn't get an init_code_hash argument."),
        };

        let gpu_device_string = match args.next() {
            Some(arg) => arg,
            None => String::from("255"), // indicates that CPU will be used.
        };

        // If we have a prefix, we don't need these thresholds
        let prefix_string = match args.next() {
            Some(arg) => {
                if arg.starts_with("0x") {
                    Some(without_prefix(arg))
                } else {
                    Some(arg)
                }
            },
            None => None,
        };

        // Only parse these if we don't have a prefix
        let (leading_zeroes_threshold, total_zeroes_threshold) = if prefix_string.is_some() {
            (0, 0) // Default to 0 when using prefix
        } else {
            let leading = match args.next() {
                Some(arg) => arg,
                None => String::from("7"),
            };

            let total = match args.next() {
                Some(arg) => arg,
                None => String::from("5"),
            };

            // Convert to u8
            let leading_parsed = match leading.parse::<u8>() {
                Ok(t) => t,
                Err(_) => return Err("invalid leading zeroes threshold value supplied.")
            };

            let total_parsed = match total.parse::<u8>() {
                Ok(t) => t,
                Err(_) => return Err("invalid total zeroes threshold value supplied.")
            };

            (leading_parsed, total_parsed)
        };

        // strip 0x from args if applicable
        if factory_address_string.starts_with("0x") {
            factory_address_string = without_prefix(factory_address_string)
        }

        if calling_address_string.starts_with("0x") {
            calling_address_string = without_prefix(calling_address_string)
        }

        if init_code_hash_string.starts_with("0x") {
            init_code_hash_string = without_prefix(init_code_hash_string)
        }

        // convert main arguments from hex string to vector of bytes
        let factory_address_vec: Vec<u8> = match Vec::from_hex(
            &factory_address_string
        ) {
            Ok(t) => t,
            Err(_) => {
                return Err("could not decode factory address argument.")
            }
        };

        let calling_address_vec: Vec<u8> = match Vec::from_hex(
            &calling_address_string
        ) {
            Ok(t) => t,
            Err(_) => {
                return Err("could not decode calling address argument.")
            }
        };

        let init_code_hash_vec: Vec<u8> = match Vec::from_hex(
            &init_code_hash_string
        ) {
            Ok(t) => t,
            Err(_) => {
                return Err(
                    "could not decode initialization code hash argument."
                )
            }
        };

        // validate length of each argument (20, 20, 32)
        if factory_address_vec.len() != 20 {
            return Err("invalid length for factory address argument.")
        }

        if calling_address_vec.len() != 20 {
            return Err("invalid length for calling address argument.")
        }

        if init_code_hash_vec.len() != 32 {
            return Err("invalid length for initialization code hash argument.")
        }

        // convert from vector to fixed array
        let factory_address = to_fixed_20(factory_address_vec);
        let calling_address = to_fixed_20(calling_address_vec);
        let init_code_hash = to_fixed_32(init_code_hash_vec);

        // convert gpu arguments to u8 values
        let gpu_device: u32 = match gpu_device_string
                                               .parse::<u32>() {
            Ok(t) => t,
            Err(_) => {
                return Err(
                    "invalid gpu device value."
                )
            }
        };

        // Validate prefix if provided
        let prefix = if let Some(prefix_str) = prefix_string {
            // Validate that the prefix contains only valid hex characters
            if !prefix_str.chars().all(|c| c.is_digit(16)) {
                return Err("prefix must contain only valid hexadecimal characters");
            }
            
            Some(prefix_str)
        } else {
            None
        };

        // return the config object
        Ok(
          Self {
            factory_address,
            calling_address,
            init_code_hash,
            gpu_device,
            leading_zeroes_threshold,
            total_zeroes_threshold,
            prefix,
            starts_with: String::new(),
            ends_with: String::new(),
            case_sensitive: false,
          }
        )
    }
}

/// Remove the `0x` prefix from a hex string.
fn without_prefix(string: String) -> String {
    string
      .char_indices()
      .nth(2)
      .and_then(|(i, _)| string.get(i..))
      .unwrap()
      .to_string()
}

/// Convert a properly-sized vector to a fixed array of 20 bytes.
fn to_fixed_20(bytes: std::vec::Vec<u8>) -> [u8; 20] {
    let mut array = [0; 20];
    let bytes = &bytes[..array.len()];
    array.copy_from_slice(bytes);
    array
}

/// Convert a properly-sized vector to a fixed array of 32 bytes.
fn to_fixed_32(bytes: std::vec::Vec<u8>) -> [u8; 32] {
    let mut array = [0; 32];
    let bytes = &bytes[..array.len()];
    array.copy_from_slice(bytes);
    array
}

/// Convert 64-bit unsigned integer to little-endian fixed array of eight bytes.
pub fn u64_to_le_fixed_8(x: &u64) -> [u8; 8] {
    let mask: u64 = 0xff;
    let b1: u8 = ((x >> 56) & mask) as u8;
    let b2: u8 = ((x >> 48) & mask) as u8;
    let b3: u8 = ((x >> 40) & mask) as u8;
    let b4: u8 = ((x >> 32) & mask) as u8;
    let b5: u8 = ((x >> 24) & mask) as u8;
    let b6: u8 = ((x >> 16) & mask) as u8;
    let b7: u8 = ((x >> 8) & mask) as u8;
    let b8: u8 = (x & mask) as u8;
    [b8, b7, b6, b5, b4, b3, b2, b1]
}
