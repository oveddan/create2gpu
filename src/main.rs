extern crate create2gpu;
extern crate clap;

use std::process;
use std::error::Error;
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};

use create2gpu::{Config, gpu};

/// A tool for finding CREATE2 salts that generate addresses with specific prefixes using GPU acceleration
#[derive(Parser, Debug)]
#[command(name = "create2gpu", author, version, about, long_about = None)]
struct Args {
    /// Prefix for the contract address (e.g., "dead", "cafe", etc.)
    #[arg(long, short, value_name = "HEX")]
    starts_with: Option<String>,

    /// Suffix for the contract address (e.g., "dead", "cafe", etc.)
    #[arg(long, short, value_name = "HEX")]
    ends_with: Option<String>,

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
    
    /// Use all available GPUs
    #[arg(long, short = 'a')]
    all_gpus: bool,

    /// Output file for successful finds (CSV format)
    #[arg(long, short, value_name = "FILE", default_value = "results.csv")]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Parse the addresses and hash
    let factory_address = parse_address(&args.deployer)?;
    let calling_address = parse_address(&args.caller)?;
    let init_code_hash = parse_hash(&args.init_code_hash)?;

    // Read the best score from the CSV file if it exists
    let _best_score = read_best_score_from_csv(&args.output);

    // Create the base configuration
    let base_config = Config {
        factory_address,
        calling_address,
        init_code_hash,
        gpu_device: args.gpu,
        platform_id: 0,
        leading_zeroes_threshold: 0,
        total_zeroes_threshold: 0,
        prefix: None,
        starts_with: String::new(),
        ends_with: String::new(),
        case_sensitive: false,
        min_leading_ones: 4,
        min_trailing_ones: 4,
        output_file: args.output.clone(),
    };

    if args.all_gpus {
        // Run on all available GPUs
        run_on_all_gpus(base_config)?;
    } else {
        // Original single-GPU code
        println!("Using GPU device {}...", base_config.gpu_device);
        if let Err(e) = gpu(base_config) {
            eprintln!("GPU search failed: {}", e);
            process::exit(1);
        }
    }

    Ok(())
}

// Helper function to run the search on all available GPUs
fn run_on_all_gpus(base_config: Config) -> Result<(), Box<dyn Error>> {
    // Get all available platforms and devices
    let platforms = ocl::Platform::list();
    
    if platforms.is_empty() {
        return Err("No OpenCL platforms found".into());
    }
    
    let mut gpu_configs = Vec::new();
    let mut total_gpus = 0;
    
    // Collect all available GPUs across all platforms
    for platform_id in 0..platforms.len() {
        // Get the platform
        let platform_id = platforms[platform_id];
        
        // Get devices for this platform
        let devices = match ocl::Device::list(platform_id, None) {
            Ok(devices) => devices,
            Err(e) => {
                println!("Warning: Failed to get devices for platform {}: {}", platform_id, e);
                continue;
            }
        };
        
        for (device_id, device) in devices.iter().enumerate() {
            // Check if this is a GPU device
            let device_type = match device.info(ocl::enums::DeviceInfo::Type) {
                Ok(t) => t,
                Err(e) => {
                    println!("Warning: Failed to get device type for device {}: {}", device_id, e);
                    continue;
                }
            };
            
            // Alternative approach using string representation
            if let ocl::enums::DeviceInfoResult::Type(device_type) = device_type {
                // Convert to string and check if it contains "GPU"
                let type_str = format!("{:?}", device_type);
                if type_str.contains("GPU") {
                    let mut config = base_config.clone();
                    config.gpu_device = device_id as u32;
                    gpu_configs.push((platform_id, device_id as u32, config));
                    total_gpus += 1;
                }
            }
        }
    }
    
    if total_gpus == 0 {
        return Err("No GPU devices found".into());
    }
    
    println!("Found {} GPU devices across {} platforms", total_gpus, platforms.len());
    
    // Create a channel for the first GPU to signal when a solution is found
    let (tx, rx) = std::sync::mpsc::channel();
    
    // Spawn threads for each GPU
    let _handles: Vec<_> = gpu_configs.into_iter().map(|(platform_id, device_id, cfg)| {
        let tx = tx.clone();
        let gpu_device = cfg.gpu_device; // Clone the GPU device ID before moving cfg
        std::thread::spawn(move || {
            println!("Starting search on platform {:?} GPU {}", platform_id, device_id);
            if let Err(e) = gpu(cfg) {
                eprintln!("GPU {} search failed: {}", gpu_device, e);
            }
            // Signal that we're done (either success or failure)
            let _ = tx.send(());
        })
    }).collect();
    
    // Wait for the first GPU to find a solution
    let _ = rx.recv();
    
    // All threads will exit when the main thread exits
    println!("Solution found! Exiting...");
    
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

// Add this function to read the best score from the CSV file
fn read_best_score_from_csv(file_path: &str) -> usize {
    // Default to 8 (4 leading + 4 trailing) if file doesn't exist or can't be read
    let mut best_score = 8;
    
    // Try to open the file
    if let Ok(file) = File::open(file_path) {
        let reader = BufReader::new(file);
        
        // Skip the header line
        for line in reader.lines().skip(1) {
            if let Ok(line) = line {
                // Parse the line (format: address,salt,score,leading_ones,trailing_ones)
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 3 {
                    if let Ok(score) = parts[2].parse::<usize>() {
                        if score > best_score {
                            best_score = score;
                        }
                    }
                }
            }
        }
    }
    
    best_score
}
