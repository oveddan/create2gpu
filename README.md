# create2gpu

Leverages GPU acceleration to mine a salt for a deterministic contract address matching certain criteria using CREATE2.

Fork of [create2crunch](https://github.com/0age/create2crunch).

For CPU based generation, see [use create2](https://book.getfoundry.sh/reference/cast/cast-create2)

## Usage

[Install rust and cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)

Clone the repository:

```sh
$ git clone https://github.com/oveddan/create2gpu
```

Generate a contract address with leading "dead" prefix:

```sh
$ cargo run --release -- --starts-with dead --deployer 0x0000000000FFe8B47B3e2130213B802212439497 --caller 0x0000000000000000000000000000000000000000 --init-code-hash 0c591f26891d6443cf08c5be3584c1e6ae10a4c2f07c5c53218741e9755fb9cd
```

### Options

- `--starts-with` hex: Prefix for the contract address.
- `--deployer` address: Address of the contract deployer
- `--caller` address: Address of the caller. Used for the first 20 bytes of the salt
- `--init-code-hash` hash: Init code hash of the contract to be deployed, without 0x prefix
- `--gpu` number: GPU device to use. Defaults to 0.
- `--help`: Print help information

### Output

When a matching address is found, the tool will output:

- The contract address (with checksum)
- The creation code hash used
- The salt value to use with CREATE2
- Verification that the address matches using the same algorithm as Foundry

You can then use this salt value in your contract deployment to get the desired address.
