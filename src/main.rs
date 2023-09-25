extern crate create2gpu;
extern crate clap;

use std::process;
use std::error::Error;
use clap::Parser;

use create2gpu::{Config, gpu};

/// A tool for finding CREATE2 salts that generate addresses with specific prefixes using GPU acceleration
#[derive(Parser, Debug)]
#[command(name = "create2gpu", author, version, about, long_about = None)]
struct Args {
    /// Prefix for the contract address (e.g., "dead", "cafe", etc.)
    #[arg(long, short, value_name = "HEX")]
    starts_with: String,

    /// Address of the contract deployer that will call CREATE2
    #[arg(long, value_name = "ADDRESS")]
    deployer: String,

    /// Address of the caller (for factory addresses with frontrunning protection)
    #[arg(long, short, value_name = "ADDRESS")]
    caller: String,

    /// Keccak-256 hash of the initialization code
    #[arg(long, value_name = "HASH")]
    init_code_hash: String,

    /// GPU device to use (0 for default GPU)
    #[arg(long, short, value_name = "DEVICE", default_value = "0")]
    gpu: u32,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Parse the addresses and hash
    let factory_address = parse_address(&args.deployer)?;
    let calling_address = parse_address(&args.caller)?;
    let init_code_hash = parse_hash(&args.init_code_hash)?;

    // Create the configuration
    let config = Config {
        factory_address,
        calling_address,
        init_code_hash,
        gpu_device: args.gpu,
        leading_zeroes_threshold: 0,
        total_zeroes_threshold: 0,
        prefix: None,
        starts_with: args.starts_with,
    };

    // Run the GPU search
    println!("Using GPU device {}...", config.gpu_device);
    if let Err(e) = gpu(config) {
        eprintln!("GPU search failed: {}", e);
        process::exit(1);
    }

    Ok(())
}

// Helper function to parse an address from a hex string
fn parse_address(address_str: &str) -> Result<[u8; 20], Box<dyn Error>> {
    let address_str = if address_str.starts_with("0x") {
        &address_str[2..]
    } else {
        address_str
    };
    
    // Ensure the string has an even length
    let address_str = if address_str.len() % 2 != 0 {
        format!("0{}", address_str) // Pad with a leading zero if needed
    } else {
        address_str.to_string()
    };
    
    let bytes = hex::decode(&address_str)?;
    if bytes.len() != 20 {
        // If we don't have exactly 20 bytes, pad or truncate
        let mut result = [0u8; 20];
        let copy_len = std::cmp::min(bytes.len(), 20);
        result[20 - copy_len..].copy_from_slice(&bytes[..copy_len]);
        return Ok(result);
    }
    
    let mut result = [0u8; 20];
    result.copy_from_slice(&bytes);
    Ok(result)
}

// Helper function to parse a hash from a hex string
fn parse_hash(hash_str: &str) -> Result<[u8; 32], Box<dyn Error>> {
    let hash_str = if hash_str.starts_with("0x") {
        &hash_str[2..]
    } else {
        hash_str
    };
    
    // Ensure the string has an even length
    let hash_str = if hash_str.len() % 2 != 0 {
        format!("0{}", hash_str) // Pad with a leading zero if needed
    } else {
        hash_str.to_string()
    };
    
    let bytes = hex::decode(&hash_str)?;
    if bytes.len() != 32 {
        // If we don't have exactly 32 bytes, pad or truncate
        let mut result = [0u8; 32];
        let copy_len = std::cmp::min(bytes.len(), 32);
        result[32 - copy_len..].copy_from_slice(&bytes[..copy_len]);
        return Ok(result);
    }
    
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);
    Ok(result)
}
