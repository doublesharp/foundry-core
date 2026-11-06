use alloy_primitives::{B256, Keccak256, keccak256};
use semver::Version;
use std::{
    collections::HashSet,
    fmt,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const KEY_SCHEMA: &[u8] = b"foundry-compiler-output-cache-v2";
const ENTRY_MAGIC: &[u8; 8] = b"FOCC\0\0\0\x01";
const ENTRY_HEADER_LEN: usize = ENTRY_MAGIC.len() + 32 + 32 + 8;
static TEMP_FILE_ID: AtomicU64 = AtomicU64::new(0);
const DEFAULT_MAX_BYTES: u64 = 20 * 1024 * 1024 * 1024;
const DEFAULT_TARGET_BYTES: u64 = 18 * 1024 * 1024 * 1024;
const DEFAULT_MAX_IDLE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
const DEFAULT_PRUNE_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const ACCESS_TOUCH_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const STALE_TEMP_AGE: Duration = Duration::from_secs(60 * 60);
const PRUNE_MARKER: &str = ".last-prune-v1";
const PRUNE_LOCK: &str = ".prune-lock-v1";

/// Retention policy for the local compiler-output cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompilerCachePolicy {
    /// Size that triggers least-recently-used eviction.
    pub max_bytes: u64,
    /// Size to retain after the cache crosses `max_bytes`.
    pub target_bytes: u64,
    /// Maximum time an unused entry is retained.
    /// Access timestamps have a granularity of at most one day, capped at half this duration.
    pub max_idle: Duration,
    /// Minimum interval between automatic retention scans.
    pub prune_interval: Duration,
}

impl Default for CompilerCachePolicy {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_BYTES,
            target_bytes: DEFAULT_TARGET_BYTES,
            max_idle: DEFAULT_MAX_IDLE,
            prune_interval: DEFAULT_PRUNE_INTERVAL,
        }
    }
}

/// Inventory of the local compiler-output cache.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompilerCacheStats {
    /// Number of immutable compiler-output entries.
    pub entry_count: usize,
    /// Total bytes occupied by compiler-output entries.
    pub total_bytes: u64,
    /// Least-recently accessed entry.
    pub oldest_access: Option<SystemTime>,
    /// Most-recently accessed entry.
    pub newest_access: Option<SystemTime>,
    /// Non-fatal filesystem errors encountered while scanning.
    pub warnings: Vec<String>,
}

/// Result of a compiler-output cache cleanup operation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompilerCachePruneReport {
    /// Entries considered by the operation.
    pub scanned_entries: usize,
    /// Entries removed by the operation.
    pub removed_entries: usize,
    /// Bytes removed by the operation.
    pub removed_bytes: u64,
    /// Entries remaining after the operation.
    pub remaining_entries: usize,
    /// Entry bytes remaining after the operation.
    pub remaining_bytes: u64,
    /// Non-fatal filesystem errors encountered by the operation.
    pub warnings: Vec<String>,
}

/// Content-addressed identifier for a compiler invocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompilerOutputCacheKey(B256);

impl CompilerOutputCacheKey {
    /// Creates a key from the full compiler version, normalized standard-json input, and compiler
    /// invocation.
    pub fn new<'a>(
        version: &Version,
        executable: &[u8],
        input: &[u8],
        base_path: Option<&Path>,
        allow_paths: impl IntoIterator<Item = &'a Path>,
        include_paths: impl IntoIterator<Item = &'a Path>,
        extra_args: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        Self::new_with_executable_hash(
            version,
            keccak256(executable),
            input,
            base_path,
            allow_paths,
            include_paths,
            extra_args,
        )
    }

    pub(crate) fn new_with_executable_hash<'a>(
        version: &Version,
        executable_hash: B256,
        input: &[u8],
        base_path: Option<&Path>,
        allow_paths: impl IntoIterator<Item = &'a Path>,
        include_paths: impl IntoIterator<Item = &'a Path>,
        extra_args: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        let mut key = Keccak256::new();
        append_part(&mut key, KEY_SCHEMA);
        append_part(&mut key, version.to_string().as_bytes());
        append_part(&mut key, executable_hash.as_slice());
        append_part(&mut key, input);
        append_optional_path(&mut key, base_path, base_path);
        append_paths(&mut key, allow_paths, base_path);
        append_paths(&mut key, include_paths, base_path);
        for arg in normalized_extra_args(extra_args, base_path) {
            append_part(&mut key, arg.as_bytes());
        }
        Self(key.finalize())
    }

    const fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl fmt::Debug for CompilerOutputCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompilerOutputCacheKey").field(&self.to_string()).finish()
    }
}

impl fmt::Display for CompilerOutputCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self.0)
    }
}

/// Storage for raw compiler standard-json output.
///
/// Implementations must fail open: cache misses, storage errors, and corrupt entries all return
/// `None` or otherwise leave compilation unaffected.
pub trait CompilerOutputCache: fmt::Debug + Send + Sync {
    /// Reads a raw standard-json compiler output.
    fn read(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>>;

    /// Stores a raw standard-json compiler output.
    fn write(&self, key: &CompilerOutputCacheKey, output: &[u8]);

    /// Invalidates an entry that the compiler could not deserialize.
    fn invalidate(&self, _key: &CompilerOutputCacheKey) {}

    /// Tries to coordinate a cache miss with other invocations of the same compiler input.
    ///
    /// The caller must read again after acquiring the guard, and retain it until publication.
    /// A held lock returns [`std::io::ErrorKind::WouldBlock`], allowing cancellable retries.
    /// Unsupported backends return `Ok(None)`; other errors should bypass coordination.
    fn try_acquire(&self, _key: &CompilerOutputCacheKey) -> std::io::Result<Option<Box<dyn Send>>> {
        Ok(None)
    }
}

/// Local filesystem implementation of [`CompilerOutputCache`].
#[derive(Clone, Debug)]
pub struct LocalCompilerOutputCache {
    root: PathBuf,
    policy: CompilerCachePolicy,
}

impl LocalCompilerOutputCache {
    /// Creates a cache rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self::with_policy(root, CompilerCachePolicy::default())
    }

    /// Creates a cache rooted at `root` with an explicit retention policy.
    pub fn with_policy(root: impl Into<PathBuf>, policy: CompilerCachePolicy) -> Self {
        Self { root: root.into(), policy }
    }

    /// Returns the cache root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn entry_path(&self, key: &CompilerOutputCacheKey) -> PathBuf {
        let key = key.to_string();
        self.root.join(&key[..2]).join(key)
    }

    /// Returns a best-effort inventory of immutable compiler-output entries.
    pub fn stats(&self) -> CompilerCacheStats {
        let (entries, warnings) = self.entries();
        CompilerCacheStats {
            entry_count: entries.len(),
            total_bytes: entries.iter().map(|entry| entry.bytes).sum(),
            oldest_access: entries.iter().map(|entry| entry.accessed).min(),
            newest_access: entries.iter().map(|entry| entry.accessed).max(),
            warnings,
        }
    }

    /// Removes all recognized compiler-output entries while preserving unknown files.
    pub fn clean(&self) -> CompilerCachePruneReport {
        let (mut entries, mut warnings) = self.entries();
        entries.extend(self.stale_temporary_entries(&mut warnings));
        self.remove_entries(entries, warnings)
    }

    /// Applies idle-time and least-recently-used size retention.
    pub fn prune(&self, policy: CompilerCachePolicy) -> CompilerCachePruneReport {
        let (mut entries, mut warnings) = self.entries();
        let stale_temporary_entries = self.stale_temporary_entries(&mut warnings);
        let scanned_entries = entries.len() + stale_temporary_entries.len();
        entries.sort_unstable_by_key(|entry| (entry.accessed, entry.path.clone()));
        let now = SystemTime::now();
        let total_bytes =
            entries.iter().chain(&stale_temporary_entries).map(|entry| entry.bytes).sum::<u64>();
        let mut selected = stale_temporary_entries;
        let mut selected_bytes = selected.iter().map(|entry| entry.bytes).sum::<u64>();
        let mut selected_paths =
            selected.iter().map(|entry| entry.path.clone()).collect::<HashSet<_>>();

        for entry in &entries {
            if now.duration_since(entry.accessed).unwrap_or_default() >= policy.max_idle {
                selected_bytes = selected_bytes.saturating_add(entry.bytes);
                selected_paths.insert(entry.path.clone());
                selected.push(entry.clone());
            }
        }

        let after_idle = total_bytes.saturating_sub(selected_bytes);
        if after_idle > policy.max_bytes {
            let target = policy.target_bytes.min(policy.max_bytes);
            for entry in &entries {
                if selected_paths.contains(&entry.path) {
                    continue;
                }
                if total_bytes.saturating_sub(selected_bytes) <= target {
                    break;
                }
                selected_bytes = selected_bytes.saturating_add(entry.bytes);
                selected_paths.insert(entry.path.clone());
                selected.push(entry.clone());
            }
        }

        let mut report = self.remove_entries(selected, warnings);
        report.scanned_entries = scanned_entries;
        report.remaining_entries = report.scanned_entries.saturating_sub(report.removed_entries);
        report.remaining_bytes = total_bytes.saturating_sub(report.removed_bytes);
        report
    }

    fn entries(&self) -> (Vec<CacheEntry>, Vec<String>) {
        let mut entries = Vec::new();
        let mut warnings = Vec::new();
        let shards = match fs::read_dir(&self.root) {
            Ok(shards) => shards,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return (entries, warnings),
            Err(err) => {
                warnings
                    .push(format!("could not read compiler cache {}: {err}", self.root.display()));
                return (entries, warnings);
            }
        };
        for shard in shards {
            let shard = match shard {
                Ok(shard) => shard,
                Err(err) => {
                    warnings.push(format!("could not inspect compiler cache shard: {err}"));
                    continue;
                }
            };
            let shard_name = shard.file_name();
            let shard_name = shard_name.to_string_lossy();
            let file_type = match shard.file_type() {
                Ok(file_type) => file_type,
                Err(err) => {
                    warnings.push(format!(
                        "could not inspect cache shard {}: {err}",
                        shard.path().display()
                    ));
                    continue;
                }
            };
            if !file_type.is_dir() || !is_lower_hex(&shard_name, 2) {
                continue;
            }
            let Ok(files) = fs::read_dir(shard.path()) else {
                warnings.push(format!("could not read cache shard {}", shard.path().display()));
                continue;
            };
            for file in files {
                let file = match file {
                    Ok(file) => file,
                    Err(err) => {
                        warnings.push(format!("could not inspect compiler cache entry: {err}"));
                        continue;
                    }
                };
                let name = file.file_name();
                let name = name.to_string_lossy();
                let file_type = match file.file_type() {
                    Ok(file_type) => file_type,
                    Err(err) => {
                        warnings.push(format!(
                            "could not inspect cache entry {}: {err}",
                            file.path().display()
                        ));
                        continue;
                    }
                };
                if !file_type.is_file()
                    || !is_lower_hex(&name, 64)
                    || !name.starts_with(shard_name.as_ref())
                {
                    continue;
                }
                let mut header = [0_u8; ENTRY_MAGIC.len() + 32];
                match fs::File::open(file.path()).and_then(|mut file| file.read_exact(&mut header))
                {
                    Ok(())
                        if header.starts_with(ENTRY_MAGIC)
                            && format!("{:x}", B256::from_slice(&header[ENTRY_MAGIC.len()..]))
                                == name => {}
                    Ok(()) => continue,
                    Err(err) => {
                        warnings.push(format!(
                            "could not validate cache entry {}: {err}",
                            file.path().display()
                        ));
                        continue;
                    }
                }
                match file.metadata() {
                    Ok(metadata) => entries.push(CacheEntry {
                        path: file.path(),
                        bytes: metadata.len(),
                        accessed: metadata.modified().unwrap_or(UNIX_EPOCH),
                    }),
                    Err(err) => warnings.push(format!(
                        "could not inspect cache entry {}: {err}",
                        file.path().display()
                    )),
                }
            }
        }
        (entries, warnings)
    }

    fn stale_temporary_entries(&self, warnings: &mut Vec<String>) -> Vec<CacheEntry> {
        let mut entries = Vec::new();
        let Ok(shards) = fs::read_dir(&self.root) else { return entries };
        let now = SystemTime::now();
        for shard in shards.flatten() {
            let shard_name = shard.file_name().to_string_lossy().into_owned();
            if !shard.file_type().is_ok_and(|kind| kind.is_dir()) || !is_lower_hex(&shard_name, 2) {
                continue;
            }
            let Ok(files) = fs::read_dir(shard.path()) else { continue };
            for file in files.flatten() {
                let name = file.file_name().to_string_lossy().into_owned();
                let parts = name.split('.').collect::<Vec<_>>();
                if parts.len() != 5
                    || !is_lower_hex(parts[0], 64)
                    || !parts[0].starts_with(&shard_name)
                    || parts[1..4].iter().any(|part| {
                        part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit())
                    })
                    || parts[4] != "tmp"
                    || !file.file_type().is_ok_and(|kind| kind.is_file())
                {
                    continue;
                }
                match file.metadata() {
                    Ok(metadata)
                        if metadata
                            .modified()
                            .ok()
                            .and_then(|modified| now.duration_since(modified).ok())
                            .is_some_and(|age| age >= STALE_TEMP_AGE) =>
                    {
                        entries.push(CacheEntry {
                            path: file.path(),
                            bytes: metadata.len(),
                            accessed: metadata.modified().unwrap_or(UNIX_EPOCH),
                        });
                    }
                    Ok(_) => {}
                    Err(err) => warnings.push(format!(
                        "could not inspect temporary cache entry {}: {err}",
                        file.path().display()
                    )),
                }
            }
        }
        entries
    }

    fn remove_entries(
        &self,
        entries: Vec<CacheEntry>,
        mut warnings: Vec<String>,
    ) -> CompilerCachePruneReport {
        let scanned_entries = entries.len();
        let scanned_bytes = entries.iter().map(|entry| entry.bytes).sum::<u64>();
        let mut removed_entries = 0;
        let mut removed_bytes = 0_u64;
        for entry in entries {
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    removed_entries += 1;
                    removed_bytes = removed_bytes.saturating_add(entry.bytes);
                    if let Some(parent) = entry.path.parent() {
                        let _ = fs::remove_dir(parent);
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => warnings
                    .push(format!("could not remove cache entry {}: {err}", entry.path.display())),
            }
        }
        CompilerCachePruneReport {
            scanned_entries,
            removed_entries,
            removed_bytes,
            remaining_entries: scanned_entries.saturating_sub(removed_entries),
            remaining_bytes: scanned_bytes.saturating_sub(removed_bytes),
            warnings,
        }
    }

    fn maybe_prune(&self) {
        let marker = self.root.join(PRUNE_MARKER);
        if self.policy.prune_interval != Duration::ZERO
            && fs::symlink_metadata(&marker)
                .ok()
                .filter(|metadata| metadata.file_type().is_dir())
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                .is_some_and(|elapsed| elapsed < self.policy.prune_interval)
        {
            return;
        }
        if fs::create_dir_all(&self.root).is_err() {
            return;
        }
        let lock_path = self.root.join(PRUNE_LOCK);
        let Ok(lock_file) =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).open(lock_path)
        else {
            return;
        };
        let mut lock = fd_lock::RwLock::new(lock_file);
        let Ok(_guard) = lock.try_write() else { return };
        let report = self.prune(self.policy);
        trace!(
            target: "foundry_compilers::compiler_cache",
            outcome = "pruned",
            scanned_entries = report.scanned_entries,
            removed_entries = report.removed_entries,
            removed_bytes = report.removed_bytes,
            remaining_entries = report.remaining_entries,
            remaining_bytes = report.remaining_bytes,
        );
        match fs::symlink_metadata(&marker) {
            Ok(metadata) if metadata.file_type().is_dir() => {
                let _ = fs::remove_dir(&marker);
            }
            Ok(_) => {
                let _ = fs::remove_file(&marker);
            }
            Err(_) => {}
        }
        let _ = fs::create_dir(marker);
    }

    fn touch_access_if_stale(&self, path: &Path) {
        let now = SystemTime::now();
        let touch_interval = ACCESS_TOUCH_INTERVAL.min(self.policy.max_idle / 2);
        let stale = fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_none_or(|elapsed| elapsed >= touch_interval);
        if stale && let Ok(file) = fs::File::open(path) {
            let _ = file.set_times(fs::FileTimes::new().set_modified(now));
        }
    }
}

#[derive(Clone, Debug)]
struct CacheEntry {
    path: PathBuf,
    bytes: u64,
    accessed: SystemTime,
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl CompilerOutputCache for LocalCompilerOutputCache {
    fn read(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
        let entry_path = self.entry_path(key);
        let mut entry = match fs::read(&entry_path) {
            Ok(entry) => entry,
            Err(err) => {
                trace!(
                    target: "foundry_compilers::compiler_cache",
                    outcome = "miss",
                    key = %key,
                    reason = if err.kind() == std::io::ErrorKind::NotFound {
                        "not_found"
                    } else {
                        "read_error"
                    },
                );
                self.maybe_prune();
                return None;
            }
        };
        if entry.len() < ENTRY_HEADER_LEN || !entry.starts_with(ENTRY_MAGIC) {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "corrupt",
                key = %key,
                reason = "header",
            );
            let _ = fs::remove_file(&entry_path);
            self.maybe_prune();
            return None;
        }

        let key_start = ENTRY_MAGIC.len();
        let checksum_start = key_start + 32;
        let length_start = checksum_start + 32;
        if &entry[key_start..checksum_start] != key.as_slice() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "corrupt",
                key = %key,
                reason = "key",
            );
            let _ = fs::remove_file(&entry_path);
            self.maybe_prune();
            return None;
        }

        let output_len = u64::from_be_bytes(entry[length_start..ENTRY_HEADER_LEN].try_into().ok()?);
        let output = &entry[ENTRY_HEADER_LEN..];
        if usize::try_from(output_len).ok() != Some(output.len()) {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "corrupt",
                key = %key,
                reason = "length",
            );
            let _ = fs::remove_file(&entry_path);
            self.maybe_prune();
            return None;
        }
        if keccak256(output).as_slice() != &entry[checksum_start..length_start] {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "corrupt",
                key = %key,
                reason = "checksum",
            );
            let _ = fs::remove_file(&entry_path);
            self.maybe_prune();
            return None;
        }
        self.touch_access_if_stale(&entry_path);
        trace!(
            target: "foundry_compilers::compiler_cache",
            outcome = "hit",
            key = %key,
            bytes = output.len(),
        );
        // Keep the validated allocation instead of allocating a second payload-sized buffer.
        entry.copy_within(ENTRY_HEADER_LEN.., 0);
        entry.truncate(entry.len() - ENTRY_HEADER_LEN);
        self.maybe_prune();
        Some(entry)
    }

    fn write(&self, key: &CompilerOutputCacheKey, output: &[u8]) {
        let entry_path = self.entry_path(key);
        if entry_path.exists() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "write_skipped",
                key = %key,
                reason = "exists",
            );
            return;
        }
        if fs::create_dir_all(entry_path.parent().unwrap()).is_err() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "write_error",
                key = %key,
                reason = "create_directory",
            );
            return;
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let temp_path = entry_path.with_extension(format!(
            "{}.{}.{}.tmp",
            std::process::id(),
            timestamp,
            TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(&temp_path) else {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "write_error",
                key = %key,
                reason = "create_temporary",
            );
            return;
        };

        let result = (|| -> std::io::Result<()> {
            file.write_all(ENTRY_MAGIC)?;
            file.write_all(key.as_slice())?;
            file.write_all(keccak256(output).as_slice())?;
            file.write_all(&(output.len() as u64).to_be_bytes())?;
            file.write_all(output)?;
            file.sync_all()
        })();
        drop(file);

        if result.is_ok() {
            // Creating a hard link only succeeds if the immutable destination does not exist.
            // This makes completed entries appear atomically without replacing a racing writer.
            match fs::hard_link(&temp_path, &entry_path) {
                Ok(()) => trace!(
                    target: "foundry_compilers::compiler_cache",
                    outcome = "published",
                    key = %key,
                    bytes = output.len(),
                ),
                Err(err) => trace!(
                    target: "foundry_compilers::compiler_cache",
                    outcome = if err.kind() == std::io::ErrorKind::AlreadyExists {
                        "write_skipped"
                    } else {
                        "write_error"
                    },
                    key = %key,
                    reason = "publish",
                ),
            }
        } else {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "write_error",
                key = %key,
                reason = "write_temporary",
            );
        }
        let _ = fs::remove_file(temp_path);
        self.maybe_prune();
    }

    fn invalidate(&self, key: &CompilerOutputCacheKey) {
        let path = self.entry_path(key);
        if fs::remove_file(path).is_ok() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "invalidated",
                key = %key,
            );
        }
    }

    fn try_acquire(&self, key: &CompilerOutputCacheKey) -> std::io::Result<Option<Box<dyn Send>>> {
        let path = self.entry_path(key).with_extension("lock");
        let acquire = || -> std::io::Result<_> {
            fs::create_dir_all(path.parent().unwrap())?;
            // Never unlink this file: waiters must always lock the same inode, even after clean.
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            file.try_lock()?;
            Ok(CompilerOutputCacheLock(file))
        };
        match acquire() {
            Ok(guard) => Ok(Some(Box::new(guard))),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Err(error),
            Err(error) => {
                trace!(
                    target: "foundry_compilers::compiler_cache",
                    outcome = "lock_error",
                    key = %key,
                    %error,
                );
                Ok(None)
            }
        }
    }
}

/// The OS releases this advisory lock if the process exits without dropping the guard.
struct CompilerOutputCacheLock(fs::File);

impl Drop for CompilerOutputCacheLock {
    fn drop(&mut self) {
        // Explicitly unlock in case a concurrent spawn inherited the descriptor before exec.
        let _ = self.0.unlock();
    }
}

fn append_part(buffer: &mut Keccak256, part: &[u8]) {
    buffer.update((part.len() as u64).to_be_bytes());
    buffer.update(part);
}

fn append_optional_path(buffer: &mut Keccak256, path: Option<&Path>, base_path: Option<&Path>) {
    match path {
        Some(path) => append_part(buffer, normalized_path(path, base_path).as_bytes()),
        None => append_part(buffer, b"<none>"),
    }
}

fn append_paths<'a>(
    buffer: &mut Keccak256,
    paths: impl IntoIterator<Item = &'a Path>,
    base_path: Option<&Path>,
) {
    let paths = paths.into_iter().map(|path| normalized_path(path, base_path)).collect::<Vec<_>>();
    append_part(buffer, &(paths.len() as u64).to_be_bytes());
    for path in paths {
        append_part(buffer, path.as_bytes());
    }
}

fn normalized_path(path: &Path, base_path: Option<&Path>) -> String {
    if base_path.is_some_and(|base| path == base) {
        return ".".to_string();
    }
    path.strip_prefix(base_path.unwrap_or_else(|| Path::new("")))
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn normalized_extra_args(
    extra_args: impl IntoIterator<Item = impl AsRef<str>>,
    base_path: Option<&Path>,
) -> Vec<String> {
    let mut normalized = Vec::new();
    let mut extra_args = extra_args.into_iter();
    while let Some(arg) = extra_args.next() {
        let arg = arg.as_ref();
        if let Some((flag, value)) = arg.split_once('=')
            && is_path_flag(flag)
        {
            normalized.push(format!("{flag}={}", normalize_path_arg(flag, value, base_path)));
        } else if is_path_flag(arg) {
            normalized.push(arg.to_string());
            if let Some(value) = extra_args.next() {
                normalized.push(normalize_path_arg(arg, value.as_ref(), base_path));
            }
        } else {
            normalized.push(arg.to_string());
        }
    }
    normalized
}

fn is_path_flag(arg: &str) -> bool {
    matches!(arg, "--base-path" | "--include-path" | "--allow-paths")
}

fn normalize_path_arg(flag: &str, value: &str, base_path: Option<&Path>) -> String {
    if flag == "--allow-paths" {
        return value
            .split(',')
            .map(|path| normalized_path(Path::new(path), base_path))
            .collect::<Vec<_>>()
            .join(",");
    }
    normalized_path(Path::new(value), base_path)
}

#[cfg(test)]
mod tests {
    use super::{
        CompilerCachePolicy, CompilerOutputCache, CompilerOutputCacheKey, LocalCompilerOutputCache,
        PRUNE_LOCK, PRUNE_MARKER,
    };
    use semver::Version;
    use std::{
        path::Path,
        sync::{Arc, Barrier},
        time::{Duration, SystemTime},
    };

    fn key(marker: &str) -> CompilerOutputCacheKey {
        CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc executable",
            marker.as_bytes(),
            None,
            std::iter::empty::<&Path>(),
            std::iter::empty::<&Path>(),
            std::iter::empty::<&str>(),
        )
    }

    #[test]
    fn key_matches_v2_framing() {
        for size in [0, 1, 135, 136, 137, 1024 * 1024] {
            let input = vec![0xab; size];
            let version = Version::parse("0.8.35+commit.12345678").unwrap();
            let executable = b"solc executable";
            let mut framed = Vec::new();
            for part in [
                super::KEY_SCHEMA,
                version.to_string().as_bytes(),
                alloy_primitives::keccak256(executable).as_slice(),
                input.as_slice(),
                b".",
                &2_u64.to_be_bytes(),
                b".",
                b"/external",
                &1_u64.to_be_bytes(),
                b"lib",
                b"--include-path=lib",
                b"--allow-paths",
                b".,/external",
                b"--via-ir",
            ] {
                framed.extend_from_slice(&(part.len() as u64).to_be_bytes());
                framed.extend_from_slice(part);
            }
            let actual = CompilerOutputCacheKey::new(
                &version,
                executable,
                &input,
                Some(Path::new("/project")),
                [Path::new("/project"), Path::new("/external")],
                [Path::new("/project/lib")],
                ["--include-path=/project/lib", "--allow-paths", "/project,/external", "--via-ir"],
            );
            assert_eq!(actual.0, alloy_primitives::keccak256(framed));
        }
    }

    #[test]
    fn local_store_reuses_the_entry_allocation() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = key("allocation reuse");
        let payload = vec![42; 1024 * 1024];
        store.write(&key, &payload);
        let output = store.read(&key).unwrap();
        assert_eq!(output, payload);
        assert!(output.capacity() >= output.len() + super::ENTRY_HEADER_LEN);
    }

    #[test]
    fn local_store_rejects_every_invalid_entry_region() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = key("entry regions");
        store.write(&key, b"payload");
        let path = store.entry_path(&key);
        let valid = std::fs::read(&path).unwrap();
        let mut corruptions = Vec::new();
        for offset in [0, 8, 40, 72, super::ENTRY_HEADER_LEN] {
            let mut corrupt = valid.clone();
            corrupt[offset] ^= 1;
            corruptions.push(corrupt);
        }
        for length in [0, 7, super::ENTRY_HEADER_LEN - 1, valid.len() - 1] {
            corruptions.push(valid[..length].to_vec());
        }
        let mut trailing = valid.clone();
        trailing.push(0);
        corruptions.push(trailing);
        let mut oversized = valid;
        oversized[72..super::ENTRY_HEADER_LEN].copy_from_slice(&u64::MAX.to_be_bytes());
        corruptions.push(oversized);
        for corrupted in corruptions {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, corrupted).unwrap();
            assert_eq!(store.read(&key), None);
            assert!(!path.exists());
        }
        store.write(&key, b"");
        assert_eq!(store.read(&key), Some(Vec::new()));
    }

    #[test]
    #[ignore = "manual compiler cache microbenchmark"]
    fn compiler_cache_io_benchmark() {
        let input = vec![b'x'; 8 * 1024 * 1024];
        let start = std::time::Instant::now();
        for _ in 0..64 {
            std::hint::black_box(CompilerOutputCacheKey::new(
                &Version::new(0, 8, 35),
                b"solc",
                std::hint::black_box(&input),
                None,
                [],
                [],
                std::iter::empty::<&str>(),
            ));
        }
        eprintln!("cache key: 64 x 8 MiB {:?}", start.elapsed());
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = key("benchmark");
        store.write(&key, &input);
        assert_eq!(store.read(&key).unwrap(), input);
        let start = std::time::Instant::now();
        for _ in 0..64 {
            std::hint::black_box(store.read(&key).unwrap());
        }
        eprintln!("cache read: 64 x 8 MiB {:?}", start.elapsed());
    }

    #[test]
    fn key_is_stable_across_project_roots() {
        let input =
            br#"{"language":"Solidity","sources":{"src/A.sol":{"content":"contract A {}"}}}"#;
        let executable = b"solc executable";
        let first = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            executable,
            input,
            Some(Path::new("/tmp/first-worktree")),
            [Path::new("/tmp/first-worktree")],
            [Path::new("/tmp/first-worktree/lib")],
            ["--via-ir"],
        );
        let second = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            executable,
            input,
            Some(Path::new("/tmp/second-worktree")),
            [Path::new("/tmp/second-worktree")],
            [Path::new("/tmp/second-worktree/lib")],
            ["--via-ir"],
        );

        assert_eq!(first, second);
    }

    #[test]
    fn key_normalizes_path_arguments_across_project_roots() {
        let input = br#"{"language":"Solidity","sources":{}}"#;
        let first = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc executable",
            input,
            Some(Path::new("/tmp/first-worktree")),
            std::iter::empty::<&Path>(),
            std::iter::empty::<&Path>(),
            ["--include-path=/tmp/first-worktree/lib", "--via-ir"],
        );
        let second = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc executable",
            input,
            Some(Path::new("/tmp/second-worktree")),
            std::iter::empty::<&Path>(),
            std::iter::empty::<&Path>(),
            ["--include-path=/tmp/second-worktree/lib", "--via-ir"],
        );

        assert_eq!(first, second);
    }

    #[test]
    fn key_changes_with_input_compiler_and_arguments() {
        let version = Version::parse("0.8.35+first-wrapper-target").unwrap();
        let base = || {
            CompilerOutputCacheKey::new(
                &version,
                b"solc executable",
                br#"{"sources":{"src/A.sol":{"content":"contract A {}"}}}"#,
                Some(Path::new("/project")),
                std::iter::empty::<&Path>(),
                std::iter::empty::<&Path>(),
                ["--via-ir"],
            )
        };
        let original = base();
        assert_ne!(
            original,
            CompilerOutputCacheKey::new(
                &Version::parse("0.8.35+second-wrapper-target").unwrap(),
                b"solc executable",
                br#"{"sources":{"src/A.sol":{"content":"contract A {}"}}}"#,
                Some(Path::new("/project")),
                std::iter::empty::<&Path>(),
                std::iter::empty::<&Path>(),
                ["--via-ir"],
            )
        );
        assert_ne!(
            original,
            CompilerOutputCacheKey::new(
                &version,
                b"solc executable",
                br#"{"sources":{"src/A.sol":{"content":"contract B {}"}}}"#,
                Some(Path::new("/project")),
                std::iter::empty::<&Path>(),
                std::iter::empty::<&Path>(),
                ["--via-ir"],
            )
        );
        assert_ne!(
            original,
            CompilerOutputCacheKey::new(
                &version,
                b"different solc executable",
                br#"{"sources":{"src/A.sol":{"content":"contract A {}"}}}"#,
                Some(Path::new("/project")),
                std::iter::empty::<&Path>(),
                std::iter::empty::<&Path>(),
                ["--via-ir"],
            )
        );
        assert_ne!(
            original,
            CompilerOutputCacheKey::new(
                &version,
                b"solc executable",
                br#"{"sources":{"src/A.sol":{"content":"contract A {}"}}}"#,
                Some(Path::new("/project")),
                std::iter::empty(),
                std::iter::empty(),
                ["--optimize"],
            )
        );
    }

    #[test]
    fn local_store_round_trips_atomically_and_ignores_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc executable",
            br#"{"sources":{}}"#,
            None,
            std::iter::empty::<&Path>(),
            std::iter::empty::<&Path>(),
            std::iter::empty::<&str>(),
        );

        store.write(&key, br#"{"contracts":{}}"#);
        assert_eq!(store.read(&key), Some(br#"{"contracts":{}}"#.to_vec()));

        std::fs::write(store.entry_path(&key), b"corrupt").unwrap();
        assert_eq!(store.read(&key), None);
        assert!(!store.entry_path(&key).exists());

        store.write(&key, br#"{"contracts":{"repaired":true}}"#);
        assert_eq!(store.read(&key), Some(br#"{"contracts":{"repaired":true}}"#.to_vec()));
    }

    #[test]
    fn local_store_keeps_the_first_concurrent_publication() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(LocalCompilerOutputCache::new(directory.path()));
        let key = CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc executable",
            br#"{"sources":{}}"#,
            None,
            std::iter::empty::<&Path>(),
            std::iter::empty::<&Path>(),
            std::iter::empty::<&str>(),
        );
        let barrier = Arc::new(Barrier::new(2));
        let first_store = store.clone();
        let first_key = key;
        let first_barrier = barrier.clone();
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_store.write(&first_key, b"first");
        });
        let second_store = store.clone();
        let second_key = key;
        let second = std::thread::spawn(move || {
            barrier.wait();
            second_store.write(&second_key, b"second");
        });
        first.join().unwrap();
        second.join().unwrap();

        let output = store.read(&key).unwrap();
        assert!(matches!(output.as_slice(), b"first" | b"second"));
        store.write(&key, b"replacement");
        assert_eq!(store.read(&key), Some(output));
    }

    #[test]
    fn local_store_reports_entry_inventory_without_following_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        store.write(&key("first"), b"first");
        store.write(&key("second"), b"second payload");
        std::fs::write(directory.path().join("metadata"), b"preserve").unwrap();

        #[cfg(unix)]
        std::os::unix::fs::symlink(directory.path(), directory.path().join("loop")).unwrap();

        let stats = store.stats();
        assert_eq!(stats.entry_count, 2);
        assert!(stats.total_bytes > b"first".len() as u64 + b"second payload".len() as u64);
        assert!(stats.oldest_access.is_some());
        assert!(stats.newest_access.is_some());
    }

    #[test]
    fn local_store_prunes_idle_then_lru_entries_to_target() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let oldest = key("oldest");
        let middle = key("middle");
        let newest = key("newest");
        for key in [&oldest, &middle, &newest] {
            store.write(key, &vec![b'x'; 256]);
        }
        let now = SystemTime::now();
        for (key, age) in [(&oldest, 90), (&middle, 10), (&newest, 1)] {
            let file = std::fs::File::open(store.entry_path(key)).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(now - Duration::from_secs(age)))
                .unwrap();
        }
        let entry_size = std::fs::metadata(store.entry_path(&newest)).unwrap().len();

        let report = store.prune(CompilerCachePolicy {
            max_bytes: entry_size * 2 - 1,
            target_bytes: entry_size,
            max_idle: Duration::from_secs(60),
            prune_interval: Duration::ZERO,
        });

        assert_eq!(report.removed_entries, 2);
        assert_eq!(report.remaining_entries, 1);
        assert!(store.read(&oldest).is_none());
        assert!(store.read(&middle).is_none());
        assert_eq!(store.read(&newest), Some(vec![b'x'; 256]));
    }

    #[test]
    fn local_store_clean_preserves_unknown_files() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        store.write(&key("entry"), b"output");
        let unknown = directory.path().join("metadata");
        std::fs::write(&unknown, b"preserve").unwrap();

        let report = store.clean();

        assert_eq!(report.removed_entries, 1);
        assert_eq!(report.remaining_entries, 0);
        assert_eq!(std::fs::read(unknown).unwrap(), b"preserve");
    }

    #[test]
    fn local_store_preserves_hex_named_non_cache_files() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let shard = directory.path().join("aa");
        std::fs::create_dir(&shard).unwrap();
        let unknown = shard.join(format!("aa{}", "0".repeat(62)));
        std::fs::write(&unknown, b"not a compiler cache entry").unwrap();

        let stats = store.stats();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert!(!stats.warnings.is_empty());
        store.clean();
        assert_eq!(std::fs::read(unknown).unwrap(), b"not a compiler cache entry");
    }

    #[test]
    fn local_store_reports_unreadable_cache_roots() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("cache-file");
        std::fs::write(&root, b"not a directory").unwrap();

        let stats = LocalCompilerOutputCache::new(root).stats();

        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.warnings.len(), 1);
    }

    #[test]
    fn local_store_cleans_stale_temporary_entries() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = key("temporary");
        let entry_path = store.entry_path(&key);
        std::fs::create_dir_all(entry_path.parent().unwrap()).unwrap();
        let temporary = entry_path.with_extension("123.456.0.tmp");
        std::fs::write(&temporary, vec![b'x'; 256]).unwrap();
        let file = std::fs::File::open(&temporary).unwrap();
        file.set_times(
            std::fs::FileTimes::new()
                .set_modified(SystemTime::now() - Duration::from_secs(2 * 60 * 60)),
        )
        .unwrap();

        let report = store.prune(CompilerCachePolicy {
            max_bytes: u64::MAX,
            target_bytes: u64::MAX,
            max_idle: Duration::MAX,
            prune_interval: Duration::ZERO,
        });

        assert_eq!(report.removed_entries, 1);
        assert_eq!(report.removed_bytes, 256);
        assert!(!temporary.exists());
    }

    #[test]
    fn local_store_preserves_malformed_stale_temporary_names() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        let key = key("malformed-temporary");
        let entry_path = store.entry_path(&key);
        std::fs::create_dir_all(entry_path.parent().unwrap()).unwrap();
        let malformed = entry_path.with_extension("...tmp");
        std::fs::write(&malformed, b"unknown").unwrap();
        let file = std::fs::File::open(&malformed).unwrap();
        file.set_times(
            std::fs::FileTimes::new()
                .set_modified(SystemTime::now() - Duration::from_secs(2 * 60 * 60)),
        )
        .unwrap();

        store.clean();

        assert_eq!(std::fs::read(malformed).unwrap(), b"unknown");
    }

    #[test]
    fn local_store_ignores_stale_advisory_lock_files() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::new(directory.path());
        std::fs::create_dir_all(directory.path()).unwrap();
        let lock = directory.path().join(PRUNE_LOCK);
        std::fs::write(&lock, []).unwrap();
        let file = std::fs::File::open(&lock).unwrap();
        file.set_times(
            std::fs::FileTimes::new()
                .set_modified(SystemTime::now() - Duration::from_secs(2 * 60 * 60)),
        )
        .unwrap();

        store.write(&key("entry"), b"output");

        assert!(lock.exists());
        assert!(directory.path().join(PRUNE_MARKER).exists());
    }

    #[test]
    fn prune_lock_excludes_overlapping_owners() {
        let directory = tempfile::tempdir().unwrap();
        let lock = directory.path().join(PRUNE_LOCK);
        let first_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock)
            .unwrap();
        let second_file = std::fs::OpenOptions::new().read(true).write(true).open(&lock).unwrap();
        let mut first = fd_lock::RwLock::new(first_file);
        let mut second = fd_lock::RwLock::new(second_file);
        let first_guard = first.try_write().unwrap();

        assert!(second.try_write().is_err());
        drop(first_guard);
        assert!(second.try_write().is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn automatic_prune_does_not_follow_marker_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let sentinel = directory.path().join("sentinel");
        std::fs::write(&sentinel, b"preserve").unwrap();
        std::fs::File::open(&sentinel)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(SystemTime::now() - Duration::from_secs(2 * 24 * 60 * 60)),
            )
            .unwrap();
        let cache = directory.path().join("cache");
        std::fs::create_dir(&cache).unwrap();
        symlink(&sentinel, cache.join(PRUNE_MARKER)).unwrap();
        let store = LocalCompilerOutputCache::with_policy(
            &cache,
            CompilerCachePolicy {
                prune_interval: Duration::from_secs(1),
                ..CompilerCachePolicy::default()
            },
        );

        store.write(&key("entry"), b"output");

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"preserve");
        assert!(std::fs::symlink_metadata(cache.join(PRUNE_MARKER)).unwrap().file_type().is_dir());
    }

    #[test]
    fn local_store_prunes_idle_entries_during_hit_only_workloads() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::with_policy(
            directory.path(),
            CompilerCachePolicy {
                max_bytes: u64::MAX,
                target_bytes: u64::MAX,
                max_idle: Duration::from_secs(60),
                prune_interval: Duration::from_secs(1),
            },
        );
        let active = key("active");
        let idle = key("idle");
        store.write(&active, b"active");
        store.write(&idle, b"idle");
        let idle_file = std::fs::File::open(store.entry_path(&idle)).unwrap();
        idle_file
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(SystemTime::now() - Duration::from_secs(2 * 60 * 60)),
            )
            .unwrap();
        let marker = std::fs::File::open(directory.path().join(PRUNE_MARKER)).unwrap();
        marker
            .set_times(
                std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(2)),
            )
            .unwrap();

        assert_eq!(store.read(&active), Some(b"active".to_vec()));

        assert!(!store.entry_path(&idle).exists());
    }

    #[test]
    fn local_store_retains_active_entries_with_short_idle_policy() {
        let directory = tempfile::tempdir().unwrap();
        let store = LocalCompilerOutputCache::with_policy(
            directory.path(),
            CompilerCachePolicy {
                max_idle: Duration::from_secs(60),
                prune_interval: Duration::ZERO,
                ..CompilerCachePolicy::default()
            },
        );
        let active = key("active");
        store.write(&active, b"active");
        std::fs::File::open(store.entry_path(&active))
            .unwrap()
            .set_times(
                std::fs::FileTimes::new().set_modified(SystemTime::now() - Duration::from_secs(61)),
            )
            .unwrap();

        assert_eq!(store.read(&active), Some(b"active".to_vec()));
        assert_eq!(store.read(&active), Some(b"active".to_vec()));
    }

    #[test]
    fn local_store_applies_automatic_size_retention() {
        let directory = tempfile::tempdir().unwrap();
        let probe = LocalCompilerOutputCache::new(directory.path());
        probe.write(&key("probe"), &vec![b'x'; 256]);
        let entry_size = std::fs::metadata(probe.entry_path(&key("probe"))).unwrap().len();
        probe.clean();

        let store = LocalCompilerOutputCache::with_policy(
            directory.path(),
            CompilerCachePolicy {
                max_bytes: entry_size * 2,
                target_bytes: entry_size,
                max_idle: Duration::from_secs(365 * 24 * 60 * 60),
                prune_interval: Duration::ZERO,
            },
        );
        store.write(&key("first"), &vec![b'a'; 256]);
        store.write(&key("second"), &vec![b'b'; 256]);
        store.write(&key("third"), &vec![b'c'; 256]);

        let stats = store.stats();
        assert!(stats.total_bytes <= entry_size);
        assert_eq!(stats.entry_count, 1);
    }
}
