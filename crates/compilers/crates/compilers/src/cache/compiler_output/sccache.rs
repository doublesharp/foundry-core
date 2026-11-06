//! Optional storage client for the private **sccache 0.19.0** IPC protocol.
//!
//! This is deliberately version-pinned, not a claim of a stable sccache API. Solidity input
//! eligibility, keys, compiler identity and process coordination remain owned by this crate.
//! See mozilla/sccache v0.19.0 `src/protocol.rs`, `src/client.rs` and `src/cache/cache_io.rs`.

use super::{
    CompilerOutputCache, CompilerOutputCacheKey, ENTRY_HEADER_LEN, ENTRY_MAGIC,
    LocalCompilerOutputCache,
};
use alloy_primitives::keccak256;
use std::{
    collections::HashSet,
    env,
    io::{self, Cursor, Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpStream},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

const TIMEOUT: Duration = Duration::from_secs(2);
const MAX_FRAME: usize = 8 * 1024 * 1024;
const MAX_DECODED: usize = 64 * 1024 * 1024;
const MAX_PROBE: u64 = 64 * 1024;
const MEMBER: &str = "forgeds-solidity-v1";
const NAMESPACE: &[u8] = b"forgeds-sccache-solidity-v1\0";

/// A local compiler-output cache backed by an optional sccache 0.19.0 daemon.
///
/// Local hits never contact the daemon. The first miss probes the running daemon's version.
/// Unavailable, incompatible or failed storage is bypassed for the lifetime of this instance.
/// Only loopback TCP is currently supported; configured Unix sockets are explicitly bypassed.
#[derive(Clone, Debug)]
pub struct SccacheCompilerOutputCache {
    local: LocalCompilerOutputCache,
    state: Arc<State>,
}

#[derive(Debug, Default)]
struct State {
    daemon: OnceLock<Result<Daemon, String>>,
    disabled: AtomicBool,
    invalidated: Mutex<HashSet<CompilerOutputCacheKey>>,
}

#[derive(Clone, Copy, Debug)]
struct Daemon {
    address: SocketAddr,
    writable: bool,
}

impl SccacheCompilerOutputCache {
    /// Uses `local` for fast hits and process coordination, and sccache for secondary storage.
    pub fn new(local: LocalCompilerOutputCache) -> Self {
        Self { local, state: Arc::default() }
    }

    fn daemon(&self, key: &CompilerOutputCacheKey) -> Option<Daemon> {
        if self.state.disabled.load(Ordering::Relaxed) {
            return None;
        }
        match self.state.daemon.get_or_init(|| probe_daemon().map_err(|err| err.to_string())) {
            Ok(daemon) => Some(*daemon),
            Err(reason) => {
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_bypass", %key, %reason);
                None
            }
        }
    }

    fn transport_error(&self, key: &CompilerOutputCacheKey, error: impl std::fmt::Display) {
        self.state.disabled.store(true, Ordering::Relaxed);
        trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_error", %key, %error);
    }

    fn suppress(&self, key: &CompilerOutputCacheKey) {
        if let Ok(mut invalidated) = self.state.invalidated.lock() {
            invalidated.insert(*key);
        } else {
            self.state.disabled.store(true, Ordering::Relaxed);
        }
    }
}

impl CompilerOutputCache for SccacheCompilerOutputCache {
    fn read(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
        if let Some(output) = self.local.read(key) {
            return Some(output);
        }
        if self.state.invalidated.lock().map_or(true, |keys| keys.contains(key)) {
            return None;
        }
        let daemon = self.daemon(key)?;
        let request = get_request(key);
        let result = rpc(daemon.address, &request, TIMEOUT).and_then(parse_get_response);
        let data = match result {
            Ok(Some(data)) => data,
            Ok(None) => {
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_miss", %key, bytes = 0);
                return None;
            }
            Err(error) => {
                self.transport_error(key, error);
                return None;
            }
        };
        match decode_entry(key, &data, MAX_DECODED) {
            Ok(output) => {
                self.local.write(key, &output);
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_hit", %key, bytes = output.len());
                Some(output)
            }
            Err(error) => {
                self.suppress(key);
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_error", %key, reason = "corrupt_entry", %error);
                None
            }
        }
    }

    fn read_local(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
        self.local.read(key)
    }

    fn write(&self, key: &CompilerOutputCacheKey, output: &[u8]) {
        self.local.write(key, output);
        let Some(daemon) = self.daemon(key) else { return };
        if !daemon.writable {
            trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_bypass", %key, reason = "read_only");
            return;
        }
        let request = match encode_entry(key, output).and_then(|data| put_request(key, &data)) {
            Ok(request) => request,
            Err(error) => {
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_bypass", %key, %error);
                return;
            }
        };
        match rpc(daemon.address, &request, TIMEOUT).and_then(|data| parse_put_response(&data)) {
            Ok(()) => {
                if let Ok(mut invalidated) = self.state.invalidated.lock() {
                    invalidated.remove(key);
                }
                trace!(target: "foundry_compilers::compiler_cache", outcome = "sccache_write", %key, bytes = output.len());
            }
            Err(error) => self.transport_error(key, error),
        }
    }

    fn invalidate(&self, key: &CompilerOutputCacheKey) {
        self.local.invalidate(key);
        // The pinned protocol has no delete RPC. Do not fetch the same bad entry repeatedly.
        self.suppress(key);
    }

    fn try_acquire(&self, key: &CompilerOutputCacheKey) -> io::Result<Option<Box<dyn Send>>> {
        self.local.try_acquire(key)
    }

    fn requires_blocking_io(&self) -> bool {
        true
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "sccache request deadline exceeded"))
}

fn probe_daemon() -> io::Result<Daemon> {
    // Unlike TCP connect_timeout, std's blocking UnixStream::connect has no deadline.
    // Bypass unsupported transport explicitly instead of introducing an unbounded operation.
    #[cfg(unix)]
    if env::var_os("SCCACHE_SERVER_UDS").is_some_and(|value| !value.is_empty()) {
        return Err(invalid("sccache Unix socket transport is not supported by this backend"));
    }
    let port = env::var("SCCACHE_SERVER_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(4226);
    let deadline = Instant::now() + TIMEOUT;
    let mut child = Command::new("sccache")
        .args(["--show-stats", "--stats-format", "json"])
        .env_remove("SCCACHE_START_SERVER")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout = child.stdout.take().ok_or_else(|| invalid("missing sccache probe stdout"))?;
    let (sender, receiver) = mpsc::sync_channel(1);
    // Drain stdout concurrently, with a hard cap, so even excessive daemon statistics cannot
    // fill the pipe and deadlock process completion. No stderr content is retained.
    let reader = thread::Builder::new().name("sccache-version".into()).spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take(MAX_PROBE + 1).read_to_end(&mut bytes).map(|_| bytes);
        let _ = sender.send(result);
    });
    if let Err(error) = reader {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    let result = (|| {
        let status = loop {
            remaining(deadline)?;
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(5).min(remaining(deadline)?));
        };
        if !status.success() {
            return Err(invalid("sccache daemon version probe failed"));
        }
        let bytes = receiver
            .recv_timeout(remaining(deadline)?)
            .map_err(|_| invalid("sccache daemon version probe output timed out"))??;
        let writable = validate_probe(&bytes)?;
        Ok(Daemon { address: (Ipv4Addr::LOCALHOST, port).into(), writable })
    })();
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

fn validate_probe(bytes: &[u8]) -> io::Result<bool> {
    if bytes.len() as u64 > MAX_PROBE {
        return Err(invalid("sccache daemon version probe output too large"));
    }
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(io::Error::other)?;
    if value["version"] != "0.19.0" {
        return Err(invalid("unsupported sccache daemon version; expected 0.19.0"));
    }
    // With no reachable daemon, --show-stats fabricates empty statistics with the CLI version.
    // Only a successful daemon handshake populates effective_cache_mode.
    match value["effective_cache_mode"].as_str() {
        Some("ReadWrite") => Ok(true),
        Some("ReadOnly") => Ok(false),
        _ => Err(invalid("no compatible running sccache daemon")),
    }
}

fn storage_key(key: &CompilerOutputCacheKey) -> String {
    let mut namespaced = NAMESPACE.to_vec();
    namespaced.extend_from_slice(key.as_slice());
    format!("{:x}", keccak256(namespaced))
}

fn get_request(key: &CompilerOutputCacheKey) -> Vec<u8> {
    let mut data = 7_u32.to_le_bytes().to_vec();
    let key = storage_key(key);
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data
}

fn put_request(key: &CompilerOutputCacheKey, entry: &[u8]) -> io::Result<Vec<u8>> {
    if entry.len() > MAX_FRAME - 84 {
        return Err(invalid("sccache entry exceeds the 8 MiB frame limit"));
    }
    let mut data = get_request(key);
    data[..4].copy_from_slice(&8_u32.to_le_bytes());
    data.extend_from_slice(&(entry.len() as u64).to_le_bytes());
    data.extend_from_slice(entry);
    Ok(data)
}

fn rpc(address: SocketAddr, request: &[u8], timeout: Duration) -> io::Result<Vec<u8>> {
    if request.len() > MAX_FRAME {
        return Err(invalid("sccache request exceeds frame limit"));
    }
    let deadline = Instant::now() + timeout;
    let mut stream = TcpStream::connect_timeout(&address, remaining(deadline)?)?;
    stream.set_nodelay(true)?;
    let length = (request.len() as u32).to_be_bytes();
    for mut bytes in [&length[..], request] {
        while !bytes.is_empty() {
            stream.set_write_timeout(Some(remaining(deadline)?))?;
            match stream.write(bytes) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(written) => bytes = &bytes[written..],
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
    }
    let mut length = [0; 4];
    read_exact_before(&mut stream, &mut length, deadline)?;
    let length = u32::from_be_bytes(length) as usize;
    if !(4..=MAX_FRAME).contains(&length) {
        return Err(invalid("invalid sccache response frame length"));
    }
    let mut response = vec![0; length];
    read_exact_before(&mut stream, &mut response, deadline)?;
    Ok(response)
}

fn read_exact_before(
    stream: &mut TcpStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    while !bytes.is_empty() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        match stream.read(bytes) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(count) => bytes = &mut bytes[count..],
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

fn parse_get_response(mut response: Vec<u8>) -> io::Result<Option<Vec<u8>>> {
    if response.len() < 5 || response[..4] != 8_u32.to_le_bytes() {
        return Err(invalid("unexpected sccache get response"));
    }
    match response[4] {
        0 if response.len() == 5 => Ok(None),
        1 if response.len() >= 13 => {
            let len = u64::from_le_bytes(response[5..13].try_into().unwrap());
            if len != (response.len() - 13) as u64 {
                return Err(invalid("invalid sccache get payload length"));
            }
            response.copy_within(13.., 0);
            response.truncate(len as usize);
            Ok(Some(response))
        }
        _ => Err(invalid("invalid sccache get option")),
    }
}

fn parse_put_response(response: &[u8]) -> io::Result<()> {
    if response == [9, 0, 0, 0, 0, 0, 0, 0] {
        return Ok(());
    }
    // Any error or unsupported response fails open, without treating daemon text as trusted data.
    Err(invalid("sccache rejected the cache write or returned an invalid response"))
}

fn encode_entry(key: &CompilerOutputCacheKey, output: &[u8]) -> io::Result<Vec<u8>> {
    if output.len() > MAX_DECODED - ENTRY_HEADER_LEN {
        return Err(invalid("compiler output exceeds sccache decoded size limit"));
    }
    let mut envelope = Vec::with_capacity(ENTRY_HEADER_LEN + output.len());
    envelope.extend_from_slice(ENTRY_MAGIC);
    envelope.extend_from_slice(key.as_slice());
    envelope.extend_from_slice(keccak256(output).as_slice());
    envelope.extend_from_slice(&(output.len() as u64).to_be_bytes());
    envelope.extend_from_slice(output);
    let compressed = zstd::stream::encode_all(Cursor::new(envelope), 3)?;
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    zip.start_file(
        MEMBER,
        SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
    )
    .map_err(io::Error::other)?;
    zip.write_all(&compressed)?;
    Ok(zip.finish().map_err(io::Error::other)?.into_inner())
}

fn decode_entry(
    key: &CompilerOutputCacheKey,
    bytes: &[u8],
    max_decoded: usize,
) -> io::Result<Vec<u8>> {
    if bytes.len() > MAX_FRAME {
        return Err(invalid("sccache ZIP exceeds frame limit"));
    }
    let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(io::Error::other)?;
    if zip.len() != 1 {
        return Err(invalid("unexpected sccache ZIP members"));
    }
    let file = zip.by_name(MEMBER).map_err(io::Error::other)?;
    if file.compression() != CompressionMethod::Stored || file.size() > MAX_FRAME as u64 {
        return Err(invalid("unexpected sccache ZIP compression or size"));
    }
    let mut decoded = Vec::new();
    let mut decoder = zstd::stream::read::Decoder::new(file)?;
    // Bound decoder working memory as well as the resulting output allocation.
    decoder.window_log_max(26)?;
    decoder.take(max_decoded as u64 + 1).read_to_end(&mut decoded)?;
    if decoded.len() > max_decoded
        || decoded.len() < ENTRY_HEADER_LEN
        || !decoded.starts_with(ENTRY_MAGIC)
    {
        return Err(invalid("invalid sccache compiler envelope size or magic"));
    }
    let key_end = ENTRY_MAGIC.len() + 32;
    let checksum_end = key_end + 32;
    if &decoded[ENTRY_MAGIC.len()..key_end] != key.as_slice() {
        return Err(invalid("sccache compiler envelope key mismatch"));
    }
    let length = u64::from_be_bytes(decoded[checksum_end..ENTRY_HEADER_LEN].try_into().unwrap());
    let output = &decoded[ENTRY_HEADER_LEN..];
    if length != output.len() as u64
        || keccak256(output).as_slice() != &decoded[key_end..checksum_end]
    {
        return Err(invalid("sccache compiler envelope length or checksum mismatch"));
    }
    decoded.copy_within(ENTRY_HEADER_LEN.., 0);
    decoded.truncate(length as usize);
    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::B256;
    use std::net::TcpListener;

    fn key(byte: u8) -> CompilerOutputCacheKey {
        CompilerOutputCacheKey(B256::repeat_byte(byte))
    }

    #[test]
    fn sccache_wire_request_layout_and_namespace() {
        let key = key(1);
        let storage = storage_key(&key);
        assert_eq!(storage.len(), 64);
        assert!(storage.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(storage, key.to_string());
        let get = get_request(&key);
        assert_eq!(&get[..12], &[7, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&get[12..], storage.as_bytes());
        let put = put_request(&key, b"zip").unwrap();
        assert_eq!(&put[..4], &[8, 0, 0, 0]);
        assert_eq!(&put[76..84], &3_u64.to_le_bytes());
        assert_eq!(&put[84..], b"zip");
    }

    #[test]
    fn sccache_wire_responses_reject_truncation_tags_and_trailing_data() {
        assert!(parse_get_response(vec![8, 0, 0, 0, 0]).unwrap().is_none());
        let mut hit = vec![8, 0, 0, 0, 1];
        hit.extend_from_slice(&3_u64.to_le_bytes());
        hit.extend_from_slice(b"zip");
        assert_eq!(parse_get_response(hit.clone()).unwrap(), Some(b"zip".to_vec()));
        for invalid in [vec![], vec![8, 0, 0, 0, 0, 1], vec![8, 0, 0, 0, 2], hit[..15].to_vec()] {
            assert!(parse_get_response(invalid).is_err());
        }
        hit[5..13].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse_get_response(hit).is_err());
        assert!(parse_put_response(&[9, 0, 0, 0, 0, 0, 0, 0]).is_ok());
        assert!(parse_put_response(&[9, 0, 0, 0, 1, 0, 0, 0]).is_err());
        assert!(parse_put_response(&[9, 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn sccache_daemon_probe_requires_running_exact_version() {
        assert!(
            validate_probe(br#"{"version":"0.19.0","effective_cache_mode":"ReadWrite"}"#).unwrap()
        );
        assert!(
            !validate_probe(br#"{"version":"0.19.0","effective_cache_mode":"ReadOnly"}"#).unwrap()
        );
        for bytes in [
            br#"{"version":"0.19.0","effective_cache_mode":null}"#.as_slice(),
            br#"{"version":"0.19.0"}"#,
            br#"{"version":"0.18.0","effective_cache_mode":"ReadWrite"}"#,
            br#"{"version":"0.20.0","effective_cache_mode":"ReadWrite"}"#,
            b"not json",
        ] {
            assert!(validate_probe(bytes).is_err());
        }
        assert!(validate_probe(&vec![b' '; MAX_PROBE as usize + 1]).is_err());
    }

    #[test]
    fn sccache_envelope_roundtrip_wrong_key_corruption_and_size_limit() {
        let key = key(2);
        let output = br#"{"contracts":{"src/A.sol":{"A":{"metadata":"exact","evm":{"bytecode":{"object":"6000","sourceMap":"0:1:0"}}}}}}"#;
        let encoded = encode_entry(&key, output).unwrap();
        assert_eq!(decode_entry(&key, &encoded, MAX_DECODED).unwrap(), output);
        assert!(decode_entry(&super::tests::key(3), &encoded, MAX_DECODED).is_err());
        assert!(decode_entry(&key, &encoded, ENTRY_HEADER_LEN).is_err());
        assert!(decode_entry(&key, &encoded[..encoded.len() / 2], MAX_DECODED).is_err());
        let mut zip = ZipArchive::new(Cursor::new(&encoded)).unwrap();
        let mut decoded = zstd::stream::decode_all(zip.by_name(MEMBER).unwrap()).unwrap();
        *decoded.last_mut().unwrap() ^= 1;
        let compressed = zstd::stream::encode_all(Cursor::new(decoded), 3).unwrap();
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        zip.start_file(
            MEMBER,
            SimpleFileOptions::default().compression_method(CompressionMethod::Stored),
        )
        .unwrap();
        zip.write_all(&compressed).unwrap();
        let corrupted = zip.finish().unwrap().into_inner();
        assert!(decode_entry(&key, &corrupted, MAX_DECODED).is_err());
    }

    #[test]
    fn sccache_local_hits_do_not_probe_and_invalidated_keys_are_not_refetched() {
        let directory = tempfile::tempdir().unwrap();
        let local = LocalCompilerOutputCache::new(directory.path());
        let key = key(4);
        local.write(&key, b"{}");
        let cache = SccacheCompilerOutputCache::new(local);
        assert_eq!(cache.read(&key), Some(b"{}".to_vec()));
        assert!(cache.state.daemon.get().is_none());
        cache.invalidate(&key);
        assert!(cache.read(&key).is_none());
        assert!(cache.state.daemon.get().is_none());
        cache.state.disabled.store(true, Ordering::Relaxed);
        cache.write(&key, b"{\"contracts\":{}}");
        assert_eq!(cache.read(&key), Some(b"{\"contracts\":{}}".to_vec()));
    }

    #[test]
    fn sccache_transport_rejects_oversized_frames_before_allocating() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 80];
            stream.read_exact(&mut request).unwrap();
            stream.write_all(&((MAX_FRAME + 1) as u32).to_be_bytes()).unwrap();
        });
        assert!(rpc(address, &get_request(&key(5)), TIMEOUT).is_err());
        server.join().unwrap();
        assert!(put_request(&key(5), &vec![0; MAX_FRAME - 83]).is_err());
    }

    #[test]
    fn sccache_transport_times_out_on_unresponsive_daemon() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_millis(100));
        });
        let start = Instant::now();
        assert!(rpc(address, &get_request(&key(6)), Duration::from_millis(20)).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        server.join().unwrap();
    }
}
