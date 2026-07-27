// QSB Transaction Consensus Verifier v2
// Tests multiple flag combinations to find which one accepts our non-standard tx.

use bitcoinconsensus::{
    verify_with_flags, VERIFY_ALL_PRE_TAPROOT, VERIFY_CHECKLOCKTIMEVERIFY,
    VERIFY_CHECKSEQUENCEVERIFY, VERIFY_DERSIG, VERIFY_NONE, VERIFY_NULLDUMMY, VERIFY_P2SH,
    VERIFY_WITNESS,
};
use std::env;
use std::fs;
use std::process::exit;

fn read_hex_file(path: &str) -> Vec<u8> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: can't read {}: {}", path, e);
        exit(1);
    });
    let cleaned: String = content.chars().filter(|c| !c.is_whitespace()).collect();
    hex::decode(&cleaned).unwrap_or_else(|e| {
        eprintln!("ERROR: bad hex in {}: {}", path, e);
        exit(1);
    })
}

fn extract_vout(tx: &[u8], vout_idx: usize) -> (Vec<u8>, u64) {
    let mut p = 0;
    p += 4; // version
    if tx[p] == 0x00 && tx[p + 1] == 0x01 {
        p += 2;
    }
    let (in_count, n) = read_varint(&tx[p..]);
    p += n;
    for _ in 0..in_count {
        p += 32 + 4;
        let (sig_len, n) = read_varint(&tx[p..]);
        p += n;
        p += sig_len as usize;
        p += 4;
    }
    let (out_count, n) = read_varint(&tx[p..]);
    p += n;
    if vout_idx as u64 >= out_count {
        panic!("vout_idx {} out of range ({})", vout_idx, out_count);
    }
    for i in 0..out_count {
        let value = u64::from_le_bytes(tx[p..p + 8].try_into().unwrap());
        p += 8;
        let (spk_len, n) = read_varint(&tx[p..]);
        p += n;
        let spk = tx[p..p + spk_len as usize].to_vec();
        p += spk_len as usize;
        if i == vout_idx as u64 {
            return (spk, value);
        }
    }
    unreachable!()
}

fn read_varint(b: &[u8]) -> (u64, usize) {
    match b[0] {
        0xfd => (u16::from_le_bytes([b[1], b[2]]) as u64, 3),
        0xfe => (u32::from_le_bytes([b[1], b[2], b[3], b[4]]) as u64, 5),
        0xff => (
            u64::from_le_bytes([b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]]),
            9,
        ),
        n => (n as u64, 1),
    }
}

fn try_verify(
    label: &str,
    script_pubkey: &[u8],
    value: u64,
    spending_tx: &[u8],
    input_idx: usize,
    flags: u32,
) {
    print!("[flags = 0x{:04x}] {:45} ... ", flags, label);
    match verify_with_flags(script_pubkey, value, spending_tx, None, input_idx, flags) {
        Ok(()) => println!("✅ SUCCESS"),
        Err(e) => println!("❌ {:?}", e),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <spending_tx.hex> <funding_tx.hex> [vout_idx=0] [input_idx=0]",
            args[0]
        );
        exit(2);
    }
    let spending_path = &args[1];
    let funding_path = &args[2];
    let vout_idx: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(0);
    let input_idx: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(0);

    println!("bitcoinconsensus version: {}", bitcoinconsensus::version());
    println!("Spending tx: {}", spending_path);
    println!("Funding tx : {}  (vout {})\n", funding_path, vout_idx);

    let spending_tx = read_hex_file(spending_path);
    let funding_tx = read_hex_file(funding_path);

    let (script_pubkey, value) = extract_vout(&funding_tx, vout_idx);
    println!(
        "Funding vout {}: value = {} sats, scriptPubKey = {} bytes",
        vout_idx,
        value,
        script_pubkey.len()
    );
    println!("Spending tx size: {} bytes\n", spending_tx.len());

    println!("Available flag constants:");
    println!("  VERIFY_NONE                 = 0x{:04x}", VERIFY_NONE);
    println!("  VERIFY_P2SH                 = 0x{:04x}", VERIFY_P2SH);
    println!("  VERIFY_DERSIG               = 0x{:04x}", VERIFY_DERSIG);
    println!("  VERIFY_NULLDUMMY            = 0x{:04x}", VERIFY_NULLDUMMY);
    println!(
        "  VERIFY_CHECKLOCKTIMEVERIFY  = 0x{:04x}",
        VERIFY_CHECKLOCKTIMEVERIFY
    );
    println!(
        "  VERIFY_CHECKSEQUENCEVERIFY  = 0x{:04x}",
        VERIFY_CHECKSEQUENCEVERIFY
    );
    println!("  VERIFY_WITNESS              = 0x{:04x}", VERIFY_WITNESS);
    println!(
        "  VERIFY_ALL_PRE_TAPROOT      = 0x{:04x}\n",
        VERIFY_ALL_PRE_TAPROOT
    );

    println!("=== Testing with various flag combinations ===\n");

    try_verify(
        "NONE (no soft forks)",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_NONE,
    );
    try_verify(
        "P2SH only",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_P2SH,
    );
    try_verify(
        "P2SH + DERSIG",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_P2SH | VERIFY_DERSIG,
    );
    try_verify(
        "P2SH + DERSIG + CLTV + CSV",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_P2SH | VERIFY_DERSIG | VERIFY_CHECKLOCKTIMEVERIFY | VERIFY_CHECKSEQUENCEVERIFY,
    );
    try_verify(
        "P2SH + WITNESS",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_P2SH | VERIFY_WITNESS,
    );
    try_verify(
        "ALL_PRE_TAPROOT",
        &script_pubkey,
        value,
        &spending_tx,
        input_idx,
        VERIFY_ALL_PRE_TAPROOT,
    );

    println!("\n=== Interpretation ===");
    println!("SUCCESS with ALL_PRE_TAPROOT -> consensus-valid, ready for Slipstream.");
    println!("SUCCESS with fewer flags     -> consensus-valid under those rules.");
    println!("ERR_SCRIPT everywhere        -> script rejected at base consensus level.");
    println!("ERR_INVALID_FLAGS everywhere -> lib bug or flags not in valid combination.");
}
