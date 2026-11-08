// Copyright (c) 2024 Tencent Inc.
// SPDX-License-Identifier: Apache-2.0

//! Kernel parameter metadata extraction from CUDA cubin / fatbin images.
//!
//! For every kernel (or device function) we recover the byte size of each
//! parameter, in signature order. This is what the host daemon needs to
//! rebuild `kernelParams` for `cuLaunchKernel` replay.
//!
//! How it works:
//!
//! * A cubin is an ELF64 little-endian object with `e_machine == 190` (EM_CUDA).
//! * `.symtab` owns one `STT_FUNC`/`STT_NOTYPE` symbol per function; its
//!   `.text.<name>` and `.nv.info.<name>` sections use the symbol name as
//!   suffix.
//! * `.nv.info.<name>` is a stream of attributes, each starting with a 4-byte
//!   header `{ u8 format, u8 attribute, u16 value }`:
//!     - `EIFMT_NVAL` (1): no payload, `value` unused.
//!     - `EIFMT_BVAL` (2): 1-byte value carried in the header, no payload.
//!     - `EIFMT_HVAL` (3): `u16` value carried in the header, no payload.
//!     - `EIFMT_SVAL` (4): `value` is a byte count, payload follows.
//! * `EIATTR_KPARAM_INFO` (0x17) is an `EIFMT_SVAL` attribute whose 12-byte
//!   payload is:
//!     ```text
//!     u32 index;      // sub-index (0 in practice)
//!     u16 ordinal;    // 0-based parameter position in the signature
//!     u16 offset;     // byte offset in the kernel parameter bank
//!     u32 tail;       // bit  0..7  pointee log2 alignment
//!                     // bit  8..11 space
//!                     // bit 12..16 cbank
//!                     // bit 17     parameter space (0 = CBANK)
//!                     // bit 18..31 parameter size in bytes
//!     ```
//!   Only `ordinal` and the size field are needed here.
//! * A fatbin (`cuModuleLoadData` input) is a container (`0xBA55ED50` magic)
//!   with one or more embedded cubin ELF images. Host ELF objects carrying a
//!   `.nv_fatbin` section work the same way. We scan the image for ELF magic
//!   and merge the metadata of every embedded cubin that parses.
//!
//! All reads are bounds-checked; malformed input yields `None` (or the
//! offending cubin is skipped) and never panics.

use std::collections::HashMap;

/// ELF `e_machine` for NVIDIA CUDA binaries.
const EM_CUDA: u16 = 190;

/// ELF section type for `.symtab`.
const SHT_SYMTAB: u32 = 2;

const STT_NOTYPE: u8 = 0;
const STT_FUNC: u8 = 2;

/// `.nv.info` attribute holding per-parameter layout.
const EIATTR_KPARAM_INFO: u8 = 0x17;

const EIFMT_NVAL: u8 = 1;
const EIFMT_BVAL: u8 = 2;
const EIFMT_HVAL: u8 = 3;
const EIFMT_SVAL: u8 = 4;

/// Payload length of an `EIATTR_KPARAM_INFO` `EIFMT_SVAL` attribute.
const KPARAM_ENTRY_LEN: usize = 12;

/// Protocol limit: `module_get_func_resp.param_sizes` has 32 entries.
const MAX_PARAMS: usize = 32;

const ELF_HEADER_LEN: usize = 64;
const SECTION_HEADER_LEN: usize = 64;
const SYMBOL_ENTRY_LEN: usize = 24;

#[derive(Debug)]
struct Section {
    name: String,
    sh_type: u32,
    offset: usize,
    size: usize,
    link: usize,
    entsize: usize,
}

fn read_u16(data: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    let bytes = data.get(off..end)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let bytes = data.get(off..end)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(data: &[u8], off: usize) -> Option<u64> {
    let end = off.checked_add(8)?;
    let bytes = data.get(off..end)?;
    let mut value = [0u8; 8];
    value.copy_from_slice(bytes);
    Some(u64::from_le_bytes(value))
}

/// Read a NUL-terminated UTF-8 string starting at `off`.
fn read_cstr(data: &[u8], off: usize) -> Option<&str> {
    let tail = data.get(off..)?;
    let len = tail.iter().position(|&byte| byte == 0)?;
    std::str::from_utf8(&tail[..len]).ok()
}

fn section_bytes<'a>(image: &'a [u8], section: &Section) -> Option<&'a [u8]> {
    let end = section.offset.checked_add(section.size)?;
    image.get(section.offset..end)
}

fn find_section<'a>(sections: &'a [Section], name: &str) -> Option<&'a Section> {
    sections.iter().find(|section| section.name == name)
}

/// Parse the ELF64 section header table of a cubin at the start of `image`.
///
/// Returns `None` unless `image` begins with a little-endian ELF64 object for
/// `EM_CUDA` with an in-bounds, named section table.
fn parse_sections(image: &[u8]) -> Option<Vec<Section>> {
    if image.len() < ELF_HEADER_LEN || &image[0..4] != b"\x7fELF" {
        return None;
    }
    // ELFCLASS64, ELFDATA2LSB.
    if image[4] != 2 || image[5] != 1 {
        return None;
    }
    if read_u16(image, 0x12)? != EM_CUDA {
        return None;
    }

    let shoff = usize::try_from(read_u64(image, 0x28)?).ok()?;
    let shentsize = usize::try_from(read_u16(image, 0x3a)?).ok()?;
    let shnum = usize::try_from(read_u16(image, 0x3c)?).ok()?;
    let shstrndx = usize::try_from(read_u16(image, 0x3e)?).ok()?;

    if shnum == 0 || shentsize < SECTION_HEADER_LEN || shstrndx >= shnum {
        return None;
    }

    // (sh_name, sh_type, sh_offset, sh_size, sh_link, sh_entsize)
    let mut raw = Vec::with_capacity(shnum);
    for index in 0..shnum {
        let base = shoff.checked_add(index.checked_mul(shentsize)?)?;
        let header = image.get(base..base.checked_add(SECTION_HEADER_LEN)?)?;
        raw.push((
            read_u32(header, 0x00)?,
            read_u32(header, 0x04)?,
            usize::try_from(read_u64(header, 0x18)?).ok()?,
            usize::try_from(read_u64(header, 0x20)?).ok()?,
            usize::try_from(read_u32(header, 0x28)?).ok()?,
            usize::try_from(read_u64(header, 0x38)?).ok()?,
        ));
    }

    let shstrtab = &raw[shstrndx];
    let strtab_start = shstrtab.2;
    let strtab_end = strtab_start.checked_add(shstrtab.3)?;
    let strtab = image.get(strtab_start..strtab_end)?;

    let mut sections = Vec::with_capacity(raw.len());
    for (name_off, sh_type, offset, size, link, entsize) in raw {
        let name = read_cstr(strtab, usize::try_from(name_off).ok()?)
            .unwrap_or("")
            .to_string();
        sections.push(Section {
            name,
            sh_type,
            offset,
            size,
            link,
            entsize,
        });
    }
    Some(sections)
}

/// Names of `STT_FUNC` / `STT_NOTYPE` symbols defined by the cubin.
fn symbol_names(image: &[u8], sections: &[Section]) -> Vec<String> {
    let mut names = Vec::new();

    let Some(symtab) = sections.iter().find(|s| s.sh_type == SHT_SYMTAB) else {
        return names;
    };
    let Some(strtab) = sections.get(symtab.link) else {
        return names;
    };
    let (Some(table), Some(strings)) = (section_bytes(image, symtab), section_bytes(image, strtab))
    else {
        return names;
    };

    let entsize = symtab.entsize.max(SYMBOL_ENTRY_LEN);
    let mut off = 0usize;
    while off + SYMBOL_ENTRY_LEN <= table.len() {
        let entry = &table[off..off + SYMBOL_ENTRY_LEN];
        let st_name = read_u32(entry, 0).unwrap_or(0) as usize;
        let st_info = entry[4];
        let st_type = st_info & 0x0f;

        if (st_type == STT_FUNC || st_type == STT_NOTYPE) && st_name != 0 {
            if let Some(name) = read_cstr(strings, st_name) {
                if !name.is_empty() {
                    names.push(name.to_string());
                }
            }
        }

        match off.checked_add(entsize) {
            Some(next) if next > off => off = next,
            _ => break,
        }
    }
    names
}

/// Extract `(ordinal, size)` pairs from one `.nv.info.<name>` section.
///
/// `None` means the attribute stream is malformed (bad format tag, impossible
/// payload length, truncated KPARAM_INFO entry).
fn parse_kparam_records(data: &[u8]) -> Option<Vec<(u16, u16)>> {
    let mut params = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        // A zero byte cannot start an attribute header; treat the rest as
        // section padding.
        if data.len() - pos < 4 || data[pos] == 0 {
            break;
        }

        let format = data[pos];
        let attribute = data[pos + 1];
        let word = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
        pos += 4;

        match format {
            EIFMT_NVAL | EIFMT_BVAL | EIFMT_HVAL => {
                // Values are carried inside the header word.
            }
            EIFMT_SVAL => {
                let len = word as usize;
                let end = pos.checked_add(len)?;
                let payload = data.get(pos..end)?;

                if attribute == EIATTR_KPARAM_INFO {
                    if payload.len() < KPARAM_ENTRY_LEN {
                        return None;
                    }
                    let ordinal = u16::from_le_bytes([payload[4], payload[5]]);
                    let tail =
                        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
                    let size = ((tail >> 18) & 0x3fff) as u16;
                    params.push((ordinal, size));
                }
                pos = end;
            }
            _ => return None,
        }
    }

    Some(params)
}

/// Order parameters by ordinal and apply the protocol's 32-entry limit.
fn finalize_sizes(mut records: Vec<(u16, u16)>) -> Vec<u16> {
    records.sort_by_key(|&(ordinal, _)| ordinal);
    // Ordinal is unique per parameter; keep the first record defensively.
    records.dedup_by_key(|&mut (ordinal, _)| ordinal);
    records
        .into_iter()
        .take(MAX_PARAMS)
        .map(|(_, size)| size)
        .collect()
}

/// Parse one cubin that starts at `image[0]`.
fn parse_cubin(image: &[u8]) -> Option<HashMap<String, Vec<u16>>> {
    let sections = parse_sections(image)?;

    // Symbol-derived names first, then `.nv.info.<name>` sections as a
    // fallback for cubins with a missing/partial symbol table.
    let mut names = symbol_names(image, &sections);
    for section in &sections {
        if let Some(name) = section.name.strip_prefix(".nv.info.") {
            if !name.is_empty() && find_section(&sections, &format!(".text.{}", name)).is_some() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    names.dedup();

    let mut result = HashMap::new();
    for name in names {
        if find_section(&sections, &format!(".text.{}", name)).is_none() {
            continue;
        }
        let Some(info) = find_section(&sections, &format!(".nv.info.{}", name)) else {
            continue;
        };
        let Some(data) = section_bytes(image, info) else {
            continue;
        };
        let Some(records) = parse_kparam_records(data) else {
            continue;
        };
        result.insert(name, finalize_sizes(records));
    }
    Some(result)
}

/// Parse kernel parameter metadata from a cubin or fatbin image.
///
/// Returns kernel/function name -> ordered per-parameter byte sizes, where the
/// vector is indexed by the parameter's (0-based) ordinal in the signature and
/// is truncated to the first 32 parameters. A kernel without parameters maps
/// to an empty vector.
///
/// The image may be:
/// * a bare cubin (`nvcc -cubin`, or a cubin loaded via `cuModuleLoadData`),
/// * a fatbin container (`nvcc -fatbin`),
/// * an ELF object embedding a `.nv_fatbin` section.
///
/// Every embedded cubin is merged; when the same kernel appears more than once
/// the first parsed size vector wins.
///
/// Returns `None` when the image contains no parseable cubin metadata. Callers
/// should then report `has_meta = 0` / `NOT_SUPPORTED` for the module.
/// Parse a PTX text image: every `.entry <name>(.param ...)` declaration
/// yields the ordered parameter sizes. PTX is what `nvcc -ptx` produces and
/// what `cuModuleLoadData` accepts directly when JIT is used.
///
/// Parameters appear as e.g. `.param .u32 name`, `.param .f64 name`,
/// `.param .align 8 .b8 name[32]` (structs/vectors are flattened to byte
/// arrays) or `.param .v4 .f32 name` (vector types).
fn parse_ptx(image: &[u8]) -> Option<HashMap<String, Vec<u16>>> {
    let text = std::str::from_utf8(image).ok()?;
    let mut out: HashMap<String, Vec<u16>> = HashMap::new();
    let mut search_from = 0usize;

    while let Some(rel) = text[search_from..].find(".entry") {
        let entry_pos = search_from + rel;
        let after = entry_pos + ".entry".len();
        let rest = &text[after..];
        let rest_trimmed = rest.trim_start();
        let name_len = match rest_trimmed.find(|c: char| c.is_whitespace() || c == '(') {
            Some(n) if n > 0 => n,
            _ => {
                search_from = after;
                continue;
            }
        };
        let name = &rest_trimmed[..name_len];
        let after_name = after + (rest.len() - rest_trimmed.len()) + name_len;

        let open_rel = match text[after_name..].find('(') {
            Some(o) => o,
            None => break,
        };
        let open = after_name + open_rel;
        let mut depth = 0i32;
        let mut close = None;
        for (i, ch) in text[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = match close {
            Some(c) => c,
            None => break,
        };

        let params = &text[open + 1..close];
        let mut sizes = Vec::new();
        let mut ok = true;
        for segment in params.split(',') {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }
            if !segment.contains(".param") {
                ok = false;
                break;
            }
            match ptx_param_size(segment) {
                Some(size) => sizes.push(size),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            sizes.truncate(32);
            out.insert(name.to_string(), sizes);
        }
        search_from = close + 1;
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Byte size of a single PTX parameter declaration segment.
fn ptx_param_size(segment: &str) -> Option<u16> {
    // Aggregates and flattened vectors: `<name>[N]`.
    if let Some(open) = segment.rfind('[') {
        if let Some(close) = segment[open..].find(']') {
            if let Ok(n) = segment[open + 1..open + close].trim().parse::<u32>() {
                return u16::try_from(n).ok();
            }
        }
    }

    let mut vector: u16 = 1;
    for token in segment.split_whitespace() {
        if token == ".v2" {
            vector = 2;
            continue;
        }
        if token == ".v4" {
            vector = 4;
            continue;
        }
        let scalar = match token {
            ".u8" | ".s8" | ".b8" | ".pred" => 1u16,
            ".u16" | ".s16" | ".f16" | ".b16" => 2,
            ".u32" | ".s32" | ".f32" | ".b32" => 4,
            ".u64" | ".s64" | ".f64" | ".b64" => 8,
            _ => continue,
        };
        return scalar.checked_mul(vector);
    }
    None
}

pub fn parse_kernel_param_sizes(image: &[u8]) -> Option<HashMap<String, Vec<u16>>> {
    // PTX text goes through the text parser; anything else is scanned for
    // embedded cubin ELF images (bare cubin, fatbin, `.nv_fatbin`).
    if !image.starts_with(b"\x7fELF") {
        if let Ok(text) = std::str::from_utf8(image) {
            if text.contains(".version") && text.contains(".entry") {
                if let Some(kernels) = parse_ptx(image) {
                    return Some(kernels);
                }
            }
        }
    }

    let mut merged: HashMap<String, Vec<u16>> = HashMap::new();
    let mut found = false;

    let mut pos = 0usize;
    while pos + 4 <= image.len() {
        if &image[pos..pos + 4] == b"\x7fELF" {
            if let Some(kernels) = parse_cubin(&image[pos..]) {
                found = true;
                for (name, sizes) in kernels {
                    merged.entry(name).or_insert(sizes);
                }
            }
        }
        pos += 1;
    }

    if found {
        Some(merged)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUBIN: &[u8] = include_bytes!("../tests/fixtures/params.cubin");
    const FATBIN: &[u8] = include_bytes!("../tests/fixtures/params.fatbin");
    const PTX: &[u8] = include_bytes!("../tests/fixtures/params.ptx");

    fn expected() -> HashMap<String, Vec<u16>> {
        HashMap::from([
            ("noargs".to_string(), Vec::new()),
            ("mixed".to_string(), vec![4, 8, 8, 8]),
            ("big".to_string(), vec![32, 16]),
        ])
    }

    #[test]
    fn parses_cubin_parameter_sizes() {
        let parsed = parse_kernel_param_sizes(CUBIN).expect("cubin should parse");
        assert_eq!(parsed, expected());
    }

    #[test]
    fn parses_fatbin_parameter_sizes() {
        let parsed = parse_kernel_param_sizes(FATBIN).expect("fatbin should parse");
        assert_eq!(parsed, expected());
    }

    #[test]
    fn parses_ptx_parameter_sizes() {
        let parsed = parse_kernel_param_sizes(PTX).expect("ptx should parse");
        assert_eq!(parsed, expected());
    }

    #[test]
    fn finds_cubin_embedded_in_container() {
        let mut container = b"not a real fatbin header".to_vec();
        container.extend_from_slice(CUBIN);
        let parsed = parse_kernel_param_sizes(&container).expect("embedded cubin should parse");
        assert_eq!(parsed, expected());
    }

    #[test]
    fn rejects_non_cubin_images() {
        assert!(parse_kernel_param_sizes(b"").is_none());
        assert!(parse_kernel_param_sizes(b"\x7fELF").is_none());
        assert!(parse_kernel_param_sizes(&[0u8; 4096]).is_none());
        assert!(parse_kernel_param_sizes(&[0xffu8; 4096]).is_none());
        // ELF magic + garbage must not panic.
        let mut fake = b"\x7fELF\x02\x01\x01".to_vec();
        fake.extend_from_slice(&[0x41; 512]);
        assert!(parse_kernel_param_sizes(&fake).is_none());
    }

    #[test]
    fn survives_truncated_cubin() {
        for len in 0..CUBIN.len() {
            // Must not panic; every prefix either parses partially or is
            // rejected.
            let _ = parse_kernel_param_sizes(&CUBIN[..len]);
        }
        for len in 0..FATBIN.len() {
            let _ = parse_kernel_param_sizes(&FATBIN[..len]);
        }
    }

    #[test]
    fn caps_parameter_count_at_protocol_limit() {
        let mut records: Vec<(u16, u16)> = (0..40).map(|i| (i as u16, i + 1)).collect();
        records.reverse(); // out of order on purpose
        let sizes = finalize_sizes(records);
        assert_eq!(sizes.len(), MAX_PARAMS);
        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[31], 32);
    }
}
