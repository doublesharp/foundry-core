use super::command::retry_spawn;
use crate::{
    cache::compiler_output::{CompilerOutputCache, CompilerOutputCacheKey},
    compilers::compiler_path::resolve,
    resolver::parse::SolData,
};
use alloy_primitives::{B256, Keccak256, keccak256};
use foundry_compilers_artifacts::{CompilerOutput, SolcInput, sources::Source};
use foundry_compilers_core::{
    error::{Result, SolcError},
    utils::{SUPPORTS_BASE_PATH, SUPPORTS_INCLUDE_PATH, canonicalize},
};
use itertools::Itertools;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    str::FromStr,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, SystemTime},
};

#[cfg(feature = "async")]
use super::command::async_retry_spawn;

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

/// Extensions acceptable by solc compiler.
pub const SOLC_EXTENSIONS: &[&str] = &["sol", "yul"];

const CACHE_LOCK_RETRY_DELAY: Duration = Duration::from_millis(10);

#[cfg(all(test, target_os = "linux"))]
#[path = "spawn_tests.rs"]
mod spawn_tests;

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedCompilerIdentity {
    fingerprint: CompilerIdentityFingerprint,
    digest: B256,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompilerIdentityFingerprint {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
}

static COMPILER_IDENTITIES: LazyLock<Mutex<HashMap<PathBuf, CachedCompilerIdentity>>> =
    LazyLock::new(Mutex::default);

#[cfg(test)]
static COMPILER_IDENTITY_READS: LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    LazyLock::new(Mutex::default);

/// take the lock in tests, we use this to enforce that
/// a test does not run while a compiler version is being installed
///
/// This ensures that only one thread installs a missing `solc` exe.
/// Instead of taking this lock in `Solc::blocking_install`, the lock should be taken before
/// installation is detected.
#[cfg(feature = "svm-solc")]
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! take_solc_installer_lock {
    ($lock:ident) => {
        let lock_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".lock");
        let lock_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .unwrap();
        let mut lock = fd_lock::RwLock::new(lock_file);
        let $lock = lock.write().unwrap();
    };
}

/// A list of upstream Solc releases, used to check which version
/// we should download.
/// The boolean value marks whether there was an error accessing the release list
#[cfg(feature = "svm-solc")]
pub static RELEASES: std::sync::LazyLock<(svm::Releases, Vec<Version>, bool)> =
    std::sync::LazyLock::new(|| {
        match serde_json::from_str::<svm::Releases>(svm_builds::RELEASE_LIST_JSON) {
            Ok(releases) => {
                let sorted_versions = releases.clone().into_versions();
                (releases, sorted_versions, true)
            }
            Err(err) => {
                error!("failed to deserialize SVM static RELEASES JSON: {err}");
                Default::default()
            }
        }
    });

/// Abstraction over `solc` command line utility
///
/// Supports sync and async functions.
///
/// By default the solc path is configured as follows, with descending priority:
///   1. `SOLC_PATH` environment variable
///   2. [svm](https://github.com/roynalnaruto/svm-rs)'s  `global_version` (set via `svm use
///      <version>`), stored at `<svm_home>/.global_version`
///   3. `solc` otherwise
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Solc {
    /// Path to the `solc` executable
    pub solc: PathBuf,
    /// Compiler version.
    pub version: Version,
    /// Value for --base-path arg.
    pub base_path: Option<PathBuf>,
    /// Value for --allow-paths arg.
    pub allow_paths: BTreeSet<PathBuf>,
    /// Value for --include-paths arg.
    pub include_paths: BTreeSet<PathBuf>,
    /// Additional arbitrary arguments.
    pub extra_args: Vec<String>,
    /// Optional cache for successful raw standard-json compiler output.
    #[serde(skip)]
    compiler_output_cache: Option<std::sync::Arc<dyn CompilerOutputCache>>,
}

impl PartialEq for Solc {
    fn eq(&self, other: &Self) -> bool {
        self.solc == other.solc
            && self.version == other.version
            && self.base_path == other.base_path
            && self.allow_paths == other.allow_paths
            && self.include_paths == other.include_paths
            && self.extra_args == other.extra_args
    }
}

impl Eq for Solc {}

impl PartialOrd for Solc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Solc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            &self.solc,
            &self.version,
            &self.base_path,
            &self.allow_paths,
            &self.include_paths,
            &self.extra_args,
        )
            .cmp(&(
                &other.solc,
                &other.version,
                &other.base_path,
                &other.allow_paths,
                &other.include_paths,
                &other.extra_args,
            ))
    }
}

impl Solc {
    /// A new instance which points to `solc`. Invokes `solc --version` to determine the version.
    ///
    /// Returns error if `solc` is not found in the system or if the version cannot be retrieved.
    #[instrument(name = "Solc::new", skip_all)]
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        Self::new_with_args(path, Vec::<String>::new())
    }

    /// A new instance which points to `solc` with additional cli arguments. Invokes `solc
    /// --version` to determine the version.
    ///
    /// Returns error if `solc` is not found in the system or if the version cannot be retrieved.
    pub fn new_with_args(
        path: impl Into<PathBuf>,
        extra_args: impl IntoIterator<Item: Into<String>>,
    ) -> Result<Self> {
        let path = resolve(&path.into())?;
        let extra_args = extra_args.into_iter().map(Into::into).collect::<Vec<_>>();
        let version = Self::version_with_args(path.clone(), &extra_args)?;
        Ok(Self::_new(path, version, extra_args))
    }

    /// A new instance which points to `solc` with the given version
    pub fn new_with_version(path: impl Into<PathBuf>, version: Version) -> Self {
        let path = path.into();
        Self::_new(resolve(&path).unwrap_or(path), version, Default::default())
    }

    fn _new(path: PathBuf, version: Version, extra_args: Vec<String>) -> Self {
        let this = Self {
            solc: path,
            version,
            base_path: None,
            allow_paths: Default::default(),
            include_paths: Default::default(),
            extra_args,
            compiler_output_cache: None,
        };
        this.debug_assert();
        this
    }

    fn debug_assert(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        if let Ok(v) = Self::version_with_args(&self.solc, &self.extra_args) {
            assert_eq!(v.major, self.version.major);
            assert_eq!(v.minor, self.version.minor);
            assert_eq!(v.patch, self.version.patch);
        }
    }

    /// Configures a cache for successful raw standard-json compiler output.
    ///
    /// Callers must only opt in compilers whose behavior is fully identified by the executable and
    /// its invocation path, reported version, standard-json input, and command arguments.
    /// Foundry does not opt local wrapper scripts into shared caching automatically.
    pub fn set_compiler_output_cache(&mut self, cache: std::sync::Arc<dyn CompilerOutputCache>) {
        self.compiler_output_cache = Some(cache);
    }

    /// Parses the given source looking for the `pragma` definition and
    /// returns the corresponding SemVer version requirement.
    pub fn source_version_req(source: &Source) -> Result<VersionReq> {
        Ok(SolData::parse_version_pragma(&source.content).ok_or(SolcError::PragmaNotFound)??)
    }

    /// Given a Solidity source, it detects the latest compiler version which can be used
    /// to build it, and returns it.
    ///
    /// If the required compiler version is not installed, it also proceeds to install it.
    #[cfg(feature = "svm-solc")]
    pub fn detect_version(source: &Source) -> Result<Version> {
        // detects the required solc version
        let sol_version = Self::source_version_req(source)?;
        Self::ensure_installed(&sol_version)
    }

    /// Given a Solidity version requirement, it detects the latest compiler version which can be
    /// used to build it, and returns it.
    ///
    /// If the required compiler version is not installed, it also proceeds to install it.
    #[cfg(feature = "svm-solc")]
    pub fn ensure_installed(sol_version: &VersionReq) -> Result<Version> {
        #[cfg(test)]
        take_solc_installer_lock!(_lock);

        // load the local / remote versions
        let versions = Self::installed_versions();

        let local_versions = Self::find_matching_installation(&versions, sol_version);
        let remote_versions = Self::find_matching_installation(&RELEASES.1, sol_version);

        // if there's a better upstream version than the one we have, install it
        Ok(match (local_versions, remote_versions) {
            (Some(local), None) => local,
            (Some(local), Some(remote)) => {
                if remote > local {
                    Self::blocking_install(&remote)?;
                    remote
                } else {
                    local
                }
            }
            (None, Some(version)) => {
                Self::blocking_install(&version)?;
                version
            }
            // do nothing otherwise
            _ => return Err(SolcError::VersionNotFound),
        })
    }

    /// Assuming the `versions` array is sorted, it returns the first element which satisfies
    /// the provided [`VersionReq`]
    pub fn find_matching_installation(
        versions: &[Version],
        required_version: &VersionReq,
    ) -> Option<Version> {
        // iterate in reverse to find the last match
        versions.iter().rev().find(|version| required_version.matches(version)).cloned()
    }

    /// Returns the path for a [svm](https://github.com/roynalnaruto/svm-rs) installed version.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use foundry_compilers::solc::Solc;
    /// use semver::Version;
    ///
    /// let solc = Solc::find_svm_installed_version(&Version::new(0, 8, 9))?;
    /// assert_eq!(solc, Some(Solc::new("~/.svm/0.8.9/solc-0.8.9")?));
    ///
    /// Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    #[instrument(skip_all)]
    #[cfg(feature = "svm-solc")]
    pub fn find_svm_installed_version(version: &Version) -> Result<Option<Self>> {
        let version = if version.pre.is_empty() {
            Version::new(version.major, version.minor, version.patch)
        } else {
            // Preserve version if it is a prerelease.
            version.clone()
        };
        let solc = svm::version_binary(&version.to_string());
        if !solc.is_file() {
            return Ok(None);
        }
        Ok(Some(Self::new_with_version(&solc, version)))
    }

    /// Returns the directory in which [svm](https://github.com/roynalnaruto/svm-rs) stores all versions
    ///
    /// This will be:
    /// - `~/.svm` on unix, if it exists
    /// - $XDG_DATA_HOME (~/.local/share/svm) if the svm folder does not exist.
    #[cfg(feature = "svm-solc")]
    pub fn svm_home() -> Option<PathBuf> {
        Some(svm::data_dir().to_path_buf())
    }

    /// Returns the `semver::Version` [svm](https://github.com/roynalnaruto/svm-rs)'s `.global_version` is currently set to.
    ///  `global_version` is configured with (`svm use <version>`)
    ///
    /// This will read the version string (eg: "0.8.9") that the  `~/.svm/.global_version` file
    /// contains
    #[cfg(feature = "svm-solc")]
    pub fn svm_global_version() -> Option<Version> {
        svm::get_global_version().ok().flatten()
    }

    /// Returns the list of all solc instances installed at `SVM_HOME`
    #[cfg(feature = "svm-solc")]
    pub fn installed_versions() -> Vec<Version> {
        svm::installed_versions().unwrap_or_default()
    }

    /// Returns the list of all versions that are available to download
    #[cfg(feature = "svm-solc")]
    pub fn released_versions() -> Vec<Version> {
        RELEASES.1.clone()
    }

    /// Installs the provided version of Solc in the machine under the svm dir and returns the
    /// [Solc] instance pointing to the installation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use foundry_compilers::{solc::Solc, utils::ISTANBUL_SOLC};
    ///
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let solc = Solc::install(&ISTANBUL_SOLC).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "svm-solc")]
    #[instrument(name = "Solc::install", skip_all)]
    pub async fn install(version: &Version) -> std::result::Result<Self, svm::SvmError> {
        trace!("installing solc version \"{}\"", version);
        crate::report::solc_installation_start(version);
        match svm::install(version).await {
            Ok(path) => {
                crate::report::solc_installation_success(version);
                Ok(Self::new_with_version(path, version.clone()))
            }
            Err(err) => {
                crate::report::solc_installation_error(version, &err.to_string());
                Err(err)
            }
        }
    }

    /// Blocking version of `Self::install`
    #[cfg(feature = "svm-solc")]
    #[instrument(name = "Solc::blocking_install", skip_all)]
    pub fn blocking_install(version: &Version) -> std::result::Result<Self, svm::SvmError> {
        use foundry_compilers_core::utils::RuntimeOrHandle;

        #[cfg(test)]
        crate::take_solc_installer_lock!(_lock);

        let version = if version.pre.is_empty() {
            Version::new(version.major, version.minor, version.patch)
        } else {
            // Preserve version if it is a prerelease.
            version.clone()
        };

        trace!("blocking installing solc version \"{}\"", version);
        crate::report::solc_installation_start(&version);
        // The async version `svm::install` is used instead of `svm::blocking_install`
        // because the underlying `reqwest::blocking::Client` does not behave well
        // inside of a Tokio runtime. See: https://github.com/seanmonstar/reqwest/issues/1017
        match RuntimeOrHandle::new().block_on(svm::install(&version)) {
            Ok(path) => {
                crate::report::solc_installation_success(&version);
                Ok(Self::new_with_version(path, version.clone()))
            }
            Err(err) => {
                crate::report::solc_installation_error(&version, &err.to_string());
                Err(err)
            }
        }
    }

    /// Verify that the checksum for this version of solc is correct. We check against the SHA256
    /// checksum from the build information published by [binaries.soliditylang.org](https://binaries.soliditylang.org/)
    #[cfg(feature = "svm-solc")]
    #[instrument(name = "Solc::verify_checksum", skip_all)]
    pub fn verify_checksum(&self) -> Result<()> {
        let version = self.version_short();
        let mut version_path = svm::version_path(version.to_string().as_str());
        version_path.push(format!("solc-{}", version.to_string().as_str()));
        trace!(target:"solc", "reading solc binary for checksum {:?}", version_path);
        let content =
            std::fs::read(&version_path).map_err(|err| SolcError::io(err, version_path.clone()))?;

        if !RELEASES.2 {
            // we skip checksum verification because the underlying request to fetch release info
            // failed so we have nothing to compare against
            return Ok(());
        }

        #[cfg(windows)]
        {
            // Prior to 0.7.2, binaries are released as exe files which are hard to verify: <https://github.com/foundry-rs/foundry/issues/5601>
            // <https://binaries.soliditylang.org/windows-amd64/list.json>
            const V0_7_2: Version = Version::new(0, 7, 2);
            if version < V0_7_2 {
                return Ok(());
            }
        }

        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(content);
        let checksum_calc = &hasher.finalize()[..];

        let checksum_found = &RELEASES
            .0
            .get_checksum(&version)
            .ok_or_else(|| SolcError::ChecksumNotFound { version: version.clone() })?;

        if checksum_calc == checksum_found {
            Ok(())
        } else {
            use alloy_primitives::hex;
            let expected = hex::encode(checksum_found);
            let detected = hex::encode(checksum_calc);
            warn!(target: "solc", "checksum mismatch for {:?}, expected {}, but found {} for file {:?}", version, expected, detected, version_path);
            Err(SolcError::ChecksumMismatch { version, expected, detected, file: version_path })
        }
    }

    /// Convenience function for compiling all sources under the given path
    pub fn compile_source(&self, path: &Path) -> Result<CompilerOutput> {
        let mut res: CompilerOutput = Default::default();
        for input in
            SolcInput::resolve_and_build(Source::read_sol_yul_from(path)?, Default::default())
        {
            let input = input.sanitized(&self.version);
            let output = self.compile(&input)?;
            res.merge(output)
        }

        Ok(res)
    }

    /// Same as [`Self::compile()`], but only returns those files which are included in the
    /// `CompilerInput`.
    ///
    /// In other words, this removes those files from the `CompilerOutput` that are __not__ included
    /// in the provided `CompilerInput`.
    ///
    /// # Examples
    pub fn compile_exact(&self, input: &SolcInput) -> Result<CompilerOutput> {
        let mut out = self.compile(input)?;
        out.retain_files(input.sources.keys().map(|p| p.as_path()));
        Ok(out)
    }

    /// Compiles with `--standard-json` and deserializes the output as [`CompilerOutput`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use foundry_compilers::{
    ///     artifacts::{SolcInput, Source},
    ///     compilers::{Compiler, CompilerInput},
    ///     solc::Solc,
    /// };
    ///
    /// let solc = Solc::new("solc")?;
    /// let input = SolcInput::resolve_and_build(
    ///     Source::read_sol_yul_from("./contracts".as_ref()).unwrap(),
    ///     Default::default(),
    /// );
    /// let output = solc.compile(&input)?;
    /// # Ok::<_, Box<dyn std::error::Error>>(())
    /// ```
    pub fn compile<T: Serialize>(&self, input: &T) -> Result<CompilerOutput> {
        self.compile_as(input)
    }

    /// Compiles with `--standard-json` and deserializes the output as the given `D`.
    #[instrument(name = "Solc::compile", skip_all)]
    pub fn compile_as<T: Serialize, D: DeserializeOwned>(&self, input: &T) -> Result<D> {
        let input = serde_json::to_vec(input)?;
        if self.compiler_output_cache.is_none() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "disabled",
                reason = "not_configured",
            );
        }
        let cache_key = self
            .compiler_output_cache
            .as_ref()
            .and_then(|_| self.compiler_output_cache_key(&input));
        if let Some((cache, (key, _))) = self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && let Some(output) = cached_compiler_output(cache.as_ref(), key)
        {
            return Ok(output);
        }

        let _guard = self.compiler_output_cache.as_ref().zip(cache_key.as_ref()).and_then(
            |(cache, (key, _))| loop {
                match cache.try_acquire(key) {
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(CACHE_LOCK_RETRY_DELAY);
                    }
                    result => break result.ok().flatten(),
                }
            },
        );
        if let Some((cache, (key, _))) = self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && let Some(output) = cached_compiler_output(cache.as_ref(), key)
        {
            return Ok(output);
        }

        let raw_output = self.compile_output_bytes(&input)?;
        // Validate UTF-8 once before deserializing compiler output.
        let text = simdutf8::basic::from_utf8(&raw_output).map_err(|_| SolcError::InvalidUtf8)?;
        let output = serde_json::from_str(text)?;
        // The executable may have changed between lookup and process creation.
        if let Some((cache, (key, identity))) =
            self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && cached_compiler_identity(&self.solc).as_ref() == Some(identity)
        {
            cache.write(key, &raw_output);
        }
        Ok(output)
    }

    fn compiler_output_cache_key(
        &self,
        input: &[u8],
    ) -> Option<(CompilerOutputCacheKey, CachedCompilerIdentity)> {
        if has_path_overrides(&self.extra_args) {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "bypass",
                reason = "path_override",
            );
            return None;
        }
        let identity = match cached_compiler_identity(&self.solc) {
            Some(identity) => identity,
            None => {
                trace!(
                    target: "foundry_compilers::compiler_cache",
                    outcome = "bypass",
                    reason = "compiler_identity_unavailable",
                );
                return None;
            }
        };
        // Dispatch wrappers can select a compiler from their invocation path.
        let mut invocation = self.solc.as_os_str().as_encoded_bytes().to_vec();
        invocation.extend_from_slice(identity.digest.as_slice());
        let key = CompilerOutputCacheKey::new_with_executable_hash(
            &self.version,
            keccak256(invocation),
            input,
            self.base_path.as_deref(),
            self.allow_paths.iter().map(PathBuf::as_path),
            self.include_paths.iter().map(PathBuf::as_path),
            &self.extra_args,
        );
        if !is_cacheable_compiler_input(input, key) {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "bypass",
                reason = "input_not_closed",
            );
            return None;
        }
        Some((key, identity))
    }

    /// Compiles with `--standard-json` and returns the raw `stdout` output.
    #[instrument(name = "Solc::compile_raw", skip_all)]
    pub fn compile_output<T: Serialize>(&self, input: &T) -> Result<Vec<u8>> {
        self.compile_output_bytes(&serde_json::to_vec(input)?)
    }

    fn compile_output_bytes(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut cmd = self.configure_cmd();

        trace!(input=%String::from_utf8_lossy(input));
        debug!(?cmd, "compiling");

        let mut child = retry_spawn(|| cmd.spawn()).map_err(self.map_io_err())?;
        debug!("spawned");

        {
            let mut stdin = io::BufWriter::new(child.stdin.take().unwrap());
            stdin.write_all(input).map_err(self.map_io_err())?;
            stdin.flush().map_err(self.map_io_err())?;
        }
        debug!("wrote JSON input to stdin");

        let output = child.wait_with_output().map_err(self.map_io_err())?;
        debug!(%output.status, output.stderr = ?String::from_utf8_lossy(&output.stderr), "finished");

        compile_output(output)
    }

    /// Returns the SemVer [`Version`], stripping the pre-release and build metadata.
    pub const fn version_short(&self) -> Version {
        Version::new(self.version.major, self.version.minor, self.version.patch)
    }

    /// Returns Solar's own release version if this compiler is Solar.
    ///
    /// Solar reports it in the build metadata of its solc-compatible version, for example
    /// `0.8.38+commit.e7a91fc7.solar.0.3.0`.
    pub fn solar_version(&self) -> Option<Version> {
        let (_, version) = self.version.build.as_str().split_once("solar.")?;
        Version::parse(version).ok()
    }

    /// Invokes `solc --version` and parses the output as a SemVer [`Version`].
    pub fn version(solc: impl Into<PathBuf>) -> Result<Version> {
        Self::version_with_args(solc, &[])
    }

    /// Invokes `solc --version` and parses the output as a SemVer [`Version`].
    pub fn version_with_args(solc: impl Into<PathBuf>, args: &[String]) -> Result<Version> {
        let solc = solc.into();
        // An executable or parent-directory alias can change without changing the cache key.
        if canonicalize(&solc).is_ok_and(|resolved| resolved != solc) {
            return Self::version_impl(&solc, args);
        }
        crate::cache_version(solc, args, |solc| Self::version_impl(solc, args))
    }

    fn version_impl(solc: &Path, args: &[String]) -> Result<Version> {
        let mut cmd = Command::new(solc);
        // Makes Solar report its solc-compatible version under any file name; solc ignores it.
        cmd.env("SOLC_WRAPPER", "1")
            .args(args)
            .arg("--version")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped());
        debug!(?cmd, "getting Solc version");
        let child = retry_spawn(|| cmd.spawn()).map_err(|e| SolcError::io(e, solc))?;
        let output = child.wait_with_output().map_err(|e| SolcError::io(e, solc))?;
        trace!(?output);
        let version = version_from_output(output)?;
        debug!(%version);
        Ok(version)
    }

    fn map_io_err(&self) -> impl FnOnce(std::io::Error) -> SolcError + '_ {
        move |err| SolcError::io(err, &self.solc)
    }

    /// Configures [Command] object depending on settings and solc version used.
    /// Some features are only supported by newer versions of solc, so we have to disable them for
    /// older ones.
    pub fn configure_cmd(&self) -> Command {
        let mut cmd = Command::new(&self.solc);
        cmd.stdin(Stdio::piped()).stderr(Stdio::piped()).stdout(Stdio::piped());
        cmd.args(&self.extra_args);

        if !self.allow_paths.is_empty() {
            cmd.arg("--allow-paths");
            cmd.arg(self.allow_paths.iter().map(|p| p.display()).join(","));
        }
        if let Some(base_path) = &self.base_path {
            if SUPPORTS_BASE_PATH.matches(&self.version) {
                if SUPPORTS_INCLUDE_PATH.matches(&self.version) {
                    // `--base-path` and `--include-path` conflict if set to the same path, so
                    // as a precaution, we ensure here that the `--base-path` is not also used
                    // for `--include-path`
                    for path in
                        self.include_paths.iter().filter(|p| p.as_path() != base_path.as_path())
                    {
                        cmd.arg("--include-path").arg(path);
                    }
                }

                cmd.arg("--base-path").arg(base_path);
            }

            cmd.current_dir(base_path);
        }

        cmd.arg("--standard-json");

        cmd
    }

    /// Reads a source unit using the same base and include paths passed to Solc's filesystem
    /// loader.
    pub(super) fn read_source_unit(&self, source_unit: &str) -> Result<String> {
        if self.extra_args.iter().any(|arg| {
            arg == "--base-path"
                || arg.starts_with("--base-path=")
                || arg == "--include-path"
                || arg.starts_with("--include-path=")
        }) {
            return Err(SolcError::msg(
                "cannot recover source-unit content when extra Solc arguments override base or include paths",
            ));
        }

        let source_unit = source_unit.strip_prefix("file://").unwrap_or(source_unit);
        let cwd = std::env::current_dir().map_err(|err| SolcError::io(err, "."))?;
        let child_cwd =
            self.base_path.as_deref().map(|path| absolute_path(&cwd, path)).unwrap_or(cwd);
        let mut roots = Vec::new();
        if let Some(base_path) = &self.base_path
            && SUPPORTS_BASE_PATH.matches(&self.version)
        {
            roots.push(absolute_path(&child_cwd, base_path));
        }
        if let Some(base_path) = &self.base_path
            && SUPPORTS_BASE_PATH.matches(&self.version)
            && SUPPORTS_INCLUDE_PATH.matches(&self.version)
        {
            roots.extend(
                self.include_paths
                    .iter()
                    .filter(|path| path.as_path() != base_path.as_path())
                    .map(|path| absolute_path(&child_cwd, path)),
            );
        }
        let mut candidates = if roots.is_empty() {
            vec![absolute_path(&child_cwd, Path::new(source_unit))]
        } else {
            roots.into_iter().map(|root| append_source_unit(&root, source_unit)).collect()
        };
        candidates.retain(|path| path.is_file());

        let path = match candidates.as_slice() {
            [] => {
                return Err(SolcError::msg(format!(
                    "source unit `{source_unit}` was emitted by Solc but could not be found in the base or include paths"
                )));
            }
            [path] => path,
            _ => {
                return Err(SolcError::msg(format!(
                    "source unit `{source_unit}` is ambiguous; found in {}",
                    candidates.iter().map(|path| path.display()).format(", ")
                )));
            }
        };
        std::fs::read_to_string(path).map_err(|err| SolcError::io(err, path))
    }

    /// Either finds an installed Solc version or installs it if it's not found.
    #[cfg(feature = "svm-solc")]
    pub fn find_or_install(version: &Version) -> Result<Self> {
        let solc = if let Some(solc) = Self::find_svm_installed_version(version)? {
            solc
        } else {
            Self::blocking_install(version)?
        };

        Ok(solc)
    }
}

// These process-local caches retain only digests, decisions, and import paths. Invocation keys
// cover the exact serialized input, so eligibility can reuse their existing digest. Filesystem
// resolution is never memoized here.
const MAX_INPUT_ANALYSES: usize = 1024;
const MAX_SOURCE_ANALYSES: usize = 1024;
const MAX_SOURCE_ANALYSIS_BYTES: usize = 8 * 1024 * 1024;
static INPUT_ANALYSES: LazyLock<Mutex<HashMap<CompilerOutputCacheKey, bool>>> =
    LazyLock::new(Mutex::default);
static SOURCE_ANALYSES: LazyLock<Mutex<SourceAnalyses>> = LazyLock::new(Mutex::default);

type SourceImports = Option<Arc<[PathBuf]>>;

#[derive(Default)]
struct SourceAnalyses {
    imports: HashMap<B256, SourceImports>,
    bytes: usize,
}

impl SourceAnalyses {
    fn insert(&mut self, key: B256, imports: SourceImports) {
        if self.imports.contains_key(&key) {
            return;
        }
        let bytes = std::mem::size_of::<(B256, SourceImports)>()
            + imports.as_ref().map_or(0, |imports| {
                std::mem::size_of_val(imports.as_ref())
                    + imports.iter().map(|path| path.capacity()).sum::<usize>()
                    + 2 * std::mem::size_of::<usize>()
            });
        if bytes > MAX_SOURCE_ANALYSIS_BYTES {
            return;
        }
        if self.imports.len() >= MAX_SOURCE_ANALYSES
            || self.bytes + bytes > MAX_SOURCE_ANALYSIS_BYTES
        {
            self.imports.clear();
            self.bytes = 0;
        }
        self.imports.insert(key, imports);
        self.bytes += bytes;
    }
}

fn cached_source_imports(source_name: &str, content: &str) -> SourceImports {
    let mut digest = Keccak256::new();
    digest.update((source_name.len() as u64).to_be_bytes());
    digest.update(source_name.as_bytes());
    digest.update(content.as_bytes());
    let key = digest.finalize();
    if let Ok(analyses) = SOURCE_ANALYSES.lock()
        && let Some(imports) = analyses.imports.get(&key)
    {
        return imports.clone();
    }
    // Parse outside the lock. Concurrent misses may repeat work but cannot block unrelated hits.
    let parsed = SolData::parse(content, Path::new(source_name));
    let imports = parsed.parse_result().is_ok().then(|| {
        parsed.imports.iter().map(|import| import.data().path().to_path_buf()).collect::<Arc<_>>()
    });
    if let Ok(mut analyses) = SOURCE_ANALYSES.lock() {
        analyses.insert(key, imports.clone());
    }
    imports
}

fn is_cacheable_compiler_input(input: &[u8], key: CompilerOutputCacheKey) -> bool {
    if let Ok(analyses) = INPUT_ANALYSES.lock()
        && let Some(eligible) = analyses.get(&key)
    {
        return *eligible;
    }
    let eligible = analyze_compiler_input(input);
    if let Ok(mut analyses) = INPUT_ANALYSES.lock() {
        if analyses.len() >= MAX_INPUT_ANALYSES {
            analyses.clear();
        }
        analyses.insert(key, eligible);
    }
    eligible
}

fn analyze_compiler_input(input: &[u8]) -> bool {
    let Ok(input) = serde_json::from_slice::<serde_json::Value>(input) else {
        return false;
    };
    if input.get("language").and_then(serde_json::Value::as_str) != Some("Solidity") {
        return false;
    }
    let Some(sources) = input.get("sources").and_then(serde_json::Value::as_object) else {
        return false;
    };
    let source_names = sources.keys().map(String::as_str).collect::<HashSet<_>>();
    if !source_names.iter().all(|source_name| is_canonical_source_unit(source_name)) {
        return false;
    }
    let Some(remappings) = canonical_remappings(&input) else {
        return false;
    };

    sources.iter().all(|(source_name, source)| {
        let Some(content) = source.get("content").and_then(serde_json::Value::as_str) else {
            return false;
        };
        let Some(imports) = cached_source_imports(source_name, content) else {
            return false;
        };
        imports.iter().all(|import| {
            resolved_import_source_unit(source_name, import, &remappings)
                .is_some_and(|source_unit| source_names.contains(source_unit.as_str()))
        })
    })
}

fn resolved_import_source_unit(
    source_name: &str,
    import_path: &Path,
    remappings: &[CanonicalRemapping<'_>],
) -> Option<String> {
    let source_unit = resolve_import_source_unit(source_name, import_path.to_str()?)?;
    let source_unit =
        if let Some(remapping) = select_remapping(source_name, &source_unit, remappings) {
            format!("{}{}", remapping.target, &source_unit[remapping.prefix.len()..])
        } else {
            source_unit
        };
    is_canonical_source_unit(&source_unit).then_some(source_unit)
}

/// Resolves the canonical subset of Solidity's string-based relative-import algorithm.
fn resolve_import_source_unit(source_name: &str, import_path: &str) -> Option<String> {
    if !import_path.starts_with("./") && !import_path.starts_with("../") {
        return is_canonical_source_unit(import_path).then(|| import_path.to_owned());
    }

    let mut source_segments = source_name.split('/').collect::<Vec<_>>();
    source_segments.pop();
    let mut import_path = import_path;
    if let Some(rest) = import_path.strip_prefix("./") {
        import_path = rest;
    } else {
        while let Some(rest) = import_path.strip_prefix("../") {
            source_segments.pop()?;
            import_path = rest;
        }
    }
    is_canonical_source_unit(import_path).then(|| {
        source_segments.push(import_path);
        source_segments.join("/")
    })
}

fn is_canonical_source_unit(source_unit: &str) -> bool {
    !source_unit.is_empty()
        && !source_unit.starts_with('/')
        && !source_unit.contains(':')
        && !source_unit.contains('\\')
        && source_unit.split('/').all(|segment| !matches!(segment, "" | "." | ".."))
}

/// A remapping in solc's `context:prefix=target` form; an empty context applies to every source.
#[derive(Clone, Copy)]
struct CanonicalRemapping<'a> {
    context: &'a str,
    prefix: &'a str,
    target: &'a str,
}

fn canonical_remappings(input: &serde_json::Value) -> Option<Vec<CanonicalRemapping<'_>>> {
    let Some(settings) = input.get("settings") else {
        return Some(Vec::new());
    };
    let settings = settings.as_object()?;
    let Some(remappings) = settings.get("remappings") else {
        return Some(Vec::new());
    };
    remappings
        .as_array()?
        .iter()
        .map(serde_json::Value::as_str)
        .map(|remapping| parse_canonical_remapping(remapping?))
        .collect()
}

/// Parses a remapping like solc's `ImportRemapper::parseRemapping`, which splits at the first `=`
/// and then at the first `:` before it, and admits only canonical paths.
fn parse_canonical_remapping(remapping: &str) -> Option<CanonicalRemapping<'_>> {
    let (context_prefix, target) = remapping.split_once('=')?;
    let (context, prefix) = context_prefix.split_once(':').unwrap_or(("", context_prefix));
    (is_canonical_remapping_context(context)
        && is_canonical_remapping_path(prefix)
        && is_canonical_remapping_path(target))
    .then_some(CanonicalRemapping { context, prefix, target })
}

fn is_canonical_remapping_context(context: &str) -> bool {
    context.is_empty() || is_canonical_remapping_path(context)
}

fn is_canonical_remapping_path(path: &str) -> bool {
    is_canonical_source_unit(path) || path.strip_suffix('/').is_some_and(is_canonical_source_unit)
}

/// Selects the remapping solc's `ImportRemapper::apply` uses for `source_unit` imported from
/// `importer`.
///
/// A remapping applies when its context is a prefix of the importing source unit name and its
/// prefix is a prefix of the imported source unit name. Among those, solc picks the longest
/// context, then the longest prefix, and the last listed remapping on ties.
fn select_remapping<'a>(
    importer: &str,
    source_unit: &str,
    remappings: &'a [CanonicalRemapping<'a>],
) -> Option<CanonicalRemapping<'a>> {
    remappings
        .iter()
        .enumerate()
        .filter(|(_, remapping)| {
            importer.starts_with(remapping.context) && source_unit.starts_with(remapping.prefix)
        })
        .max_by_key(|(index, remapping)| (remapping.context.len(), remapping.prefix.len(), *index))
        .map(|(_, remapping)| *remapping)
}

fn has_path_overrides(extra_args: &[String]) -> bool {
    extra_args.iter().any(|arg| {
        matches!(arg.as_str(), "--base-path" | "--include-path" | "--allow-paths")
            || arg.starts_with("--base-path=")
            || arg.starts_with("--include-path=")
            || arg.starts_with("--allow-paths=")
    })
}

fn absolute_path(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { cwd.join(path) }
}

fn append_source_unit(root: &Path, source_unit: &str) -> PathBuf {
    #[cfg(windows)]
    let source_unit = source_unit.trim_start_matches(['/', '\\']);
    #[cfg(not(windows))]
    let source_unit = source_unit.trim_start_matches('/');

    let mut path = root.as_os_str().to_os_string();
    path.push(std::path::MAIN_SEPARATOR_STR);
    path.push(source_unit);
    path.into()
}

#[cfg(feature = "async")]
impl Solc {
    /// Convenience function for compiling all sources under the given path
    pub async fn async_compile_source(&self, path: &Path) -> Result<CompilerOutput> {
        self.async_compile(&SolcInput::resolve_and_build(
            Source::async_read_all_from(path, SOLC_EXTENSIONS).await?,
            Default::default(),
        ))
        .await
    }

    /// Run `solc --stand-json` and return the `solc`'s output as
    /// `CompilerOutput`
    pub async fn async_compile<T: Serialize>(&self, input: &T) -> Result<CompilerOutput> {
        self.async_compile_as(input).await
    }

    /// Run `solc --stand-json` and return the `solc`'s output as the given json
    /// output
    pub async fn async_compile_as<T: Serialize, D: DeserializeOwned>(
        &self,
        input: &T,
    ) -> Result<D> {
        let input = serde_json::to_vec(input)?;
        if self.compiler_output_cache.is_none() {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "disabled",
                reason = "not_configured",
            );
        }
        let cache_key = self
            .compiler_output_cache
            .as_ref()
            .and_then(|_| self.compiler_output_cache_key(&input));
        if let Some((cache, (key, _))) = self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && let Some(output) = async_cached_compiler_output(cache, key).await
        {
            return Ok(output);
        }

        // Poll without a blocking task so cancellation also releases runtime shutdown.
        let _guard = if let Some((cache, (key, _))) =
            self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
        {
            loop {
                match cache.try_acquire(key) {
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        futures_timer::Delay::new(CACHE_LOCK_RETRY_DELAY).await;
                    }
                    result => break result.ok().flatten(),
                }
            }
        } else {
            None
        };
        if let Some((cache, (key, _))) = self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && let Some(output) = async_cached_compiler_output(cache, key).await
        {
            return Ok(output);
        }

        let output = self.async_compile_output_bytes(&input).await?;
        let parsed = serde_json::from_slice(&output)?;
        // The executable may have changed between lookup and process creation.
        if let Some((cache, (key, identity))) =
            self.compiler_output_cache.as_ref().zip(cache_key.as_ref())
            && cached_compiler_identity(&self.solc).as_ref() == Some(identity)
        {
            if cache.requires_blocking_io() {
                let cache = Arc::clone(cache);
                let key = *key;
                // Retain coordination through publication even if this future is cancelled.
                let _ = tokio::task::spawn_blocking(move || {
                    let _guard = _guard;
                    cache.write(&key, &output);
                })
                .await;
            } else {
                cache.write(key, &output);
            }
        }
        Ok(parsed)
    }

    pub async fn async_compile_output<T: Serialize>(&self, input: &T) -> Result<Vec<u8>> {
        self.async_compile_output_bytes(&serde_json::to_vec(input)?).await
    }

    async fn async_compile_output_bytes(&self, input: &[u8]) -> Result<Vec<u8>> {
        use tokio::{io::AsyncWriteExt, process::Command};

        let mut cmd: Command = self.configure_cmd().into();
        cmd.kill_on_drop(true);
        let mut child = async_retry_spawn(|| cmd.spawn()).await.map_err(self.map_io_err())?;
        let stdin = child.stdin.as_mut().unwrap();

        stdin.write_all(input).await.map_err(self.map_io_err())?;
        stdin.flush().await.map_err(self.map_io_err())?;

        compile_output(child.wait_with_output().await.map_err(self.map_io_err())?)
    }

    pub async fn async_version(solc: &Path) -> Result<Version> {
        let mut cmd = tokio::process::Command::new(solc);
        cmd.env("SOLC_WRAPPER", "1")
            .arg("--version")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .stdout(Stdio::piped());
        debug!(?cmd, "getting version");
        let child = async_retry_spawn(|| cmd.spawn()).await.map_err(|e| SolcError::io(e, solc))?;
        let output = child.wait_with_output().await.map_err(|e| SolcError::io(e, solc))?;
        let version = version_from_output(output)?;
        debug!(%version);
        Ok(version)
    }

    /// Compiles all `CompilerInput`s with their associated `Solc`.
    ///
    /// This will buffer up to `n` `solc` processes and then return the `CompilerOutput`s in the
    /// order in which they complete. No more than `n` futures will be buffered at any point in
    /// time, and less than `n` may also be buffered depending on the state of each future.
    pub async fn compile_many<I>(jobs: I, n: usize) -> crate::many::CompiledMany
    where
        I: IntoIterator<Item = (Self, SolcInput)>,
    {
        use futures_util::stream::StreamExt;

        let outputs = futures_util::stream::iter(
            jobs.into_iter()
                .map(|(solc, input)| async { (solc.async_compile(&input).await, solc, input) }),
        )
        .buffer_unordered(n)
        .collect::<Vec<_>>()
        .await;

        crate::many::CompiledMany::new(outputs)
    }
}

fn cached_compiler_identity(path: &Path) -> Option<CachedCompilerIdentity> {
    let before = std::fs::metadata(path).ok()?;
    let before = compiler_identity_fingerprint(&before);
    #[cfg(unix)]
    if let Some(identity) = COMPILER_IDENTITIES.lock().ok()?.get(path)
        && identity.fingerprint == before
    {
        return Some(identity.clone());
    }

    #[cfg(test)]
    record_compiler_identity_read(path);
    let executable = std::fs::read(path).ok()?;
    let after = std::fs::metadata(path).ok()?;
    let after = compiler_identity_fingerprint(&after);
    if before != after {
        return None;
    }
    let identity = CachedCompilerIdentity { fingerprint: after, digest: keccak256(executable) };
    COMPILER_IDENTITIES.lock().ok()?.insert(path.to_path_buf(), identity.clone());
    Some(identity)
}

fn compiler_identity_fingerprint(metadata: &std::fs::Metadata) -> CompilerIdentityFingerprint {
    CompilerIdentityFingerprint {
        len: metadata.len(),
        modified: metadata.modified().ok(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

#[cfg(test)]
fn record_compiler_identity_read(path: &Path) {
    *COMPILER_IDENTITY_READS.lock().unwrap().entry(path.to_path_buf()).or_default() += 1;
}

fn cached_compiler_output<D: DeserializeOwned>(
    cache: &dyn CompilerOutputCache,
    key: &CompilerOutputCacheKey,
) -> Option<D> {
    let output = cache.read(key)?;
    deserialize_cached_compiler_output(cache, key, &output)
}

#[cfg(feature = "async")]
async fn async_cached_compiler_output<D: DeserializeOwned>(
    cache: &Arc<dyn CompilerOutputCache>,
    key: &CompilerOutputCacheKey,
) -> Option<D> {
    let output = if cache.requires_blocking_io() {
        if let Some(output) = cache.read_local(key) {
            output
        } else {
            let cache = Arc::clone(cache);
            let key = *key;
            tokio::task::spawn_blocking(move || cache.read(&key)).await.ok().flatten()?
        }
    } else {
        cache.read(key)?
    };
    deserialize_cached_compiler_output(cache.as_ref(), key, &output)
}

fn deserialize_cached_compiler_output<D: DeserializeOwned>(
    cache: &dyn CompilerOutputCache,
    key: &CompilerOutputCacheKey,
    output: &[u8],
) -> Option<D> {
    match serde_json::from_slice(output) {
        Ok(output) => Some(output),
        Err(_) => {
            trace!(
                target: "foundry_compilers::compiler_cache",
                outcome = "corrupt",
                key = %key,
                reason = "invalid_json",
            );
            cache.invalidate(key);
            None
        }
    }
}

fn compile_output(output: Output) -> Result<Vec<u8>> {
    if output.status.success() { Ok(output.stdout) } else { Err(SolcError::solc_output(&output)) }
}

fn version_from_output(output: Output) -> Result<Version> {
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let version = stdout
            .lines()
            .rfind(|l| !l.trim().is_empty())
            .ok_or_else(|| SolcError::msg("Version not found in Solc output"))?;
        // NOTE: semver doesn't like `+` in g++ in build metadata which is invalid semver
        Ok(Version::from_str(&version.trim_start_matches("Version: ").replace(".g++", ".gcc"))?)
    } else {
        Err(SolcError::solc_output(&output))
    }
}

impl AsRef<Path> for Solc {
    fn as_ref(&self) -> &Path {
        &self.solc
    }
}

#[cfg(all(test, unix))]
mod cache_tests {
    use super::*;
    use crate::cache::compiler_output::LocalCompilerOutputCache;
    use serde::Serializer;
    use std::{
        env,
        os::unix::fs::{PermissionsExt, symlink},
        sync::atomic::{AtomicUsize, Ordering},
        thread,
        time::Instant,
    };

    static PATH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct FlippingInput(AtomicUsize);

    impl Serialize for FlippingInput {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let marker = self.0.fetch_add(1, Ordering::Relaxed);
            serde_json::json!({
                "language": "Solidity",
                "sources": {
                    "src/A.sol": { "content": format!("contract A {{}} // {marker}") },
                },
                "settings": { "optimizer": { "enabled": false, "runs": marker } },
            })
            .serialize(serializer)
        }
    }

    fn input_cache_key(input: &[u8]) -> CompilerOutputCacheKey {
        CompilerOutputCacheKey::new(
            &Version::new(0, 8, 35),
            b"solc",
            input,
            None,
            [],
            [],
            std::iter::empty::<&str>(),
        )
    }

    fn is_cacheable_compiler_input(input: &[u8]) -> bool {
        super::is_cacheable_compiler_input(input, input_cache_key(input))
    }

    fn uncached_compiler_input(input: &[u8]) -> bool {
        let Ok(input) = serde_json::from_slice::<serde_json::Value>(input) else {
            return false;
        };
        if input.get("language").and_then(serde_json::Value::as_str) != Some("Solidity") {
            return false;
        }
        let Some(sources) = input.get("sources").and_then(serde_json::Value::as_object) else {
            return false;
        };
        let source_names = sources.keys().map(String::as_str).collect::<HashSet<_>>();
        if !source_names.iter().all(|source_name| is_canonical_source_unit(source_name)) {
            return false;
        }
        let Some(remappings) = canonical_remappings(&input) else {
            return false;
        };

        sources.iter().all(|(source_name, source)| {
            let Some(content) = source.get("content").and_then(serde_json::Value::as_str) else {
                return false;
            };
            let parsed = SolData::parse(content, Path::new(source_name));
            if parsed.parse_result().is_err() {
                return false;
            }
            parsed.imports.iter().all(|import| {
                resolved_import_source_unit(source_name, import.data().path(), &remappings)
                    .is_some_and(|source_unit| source_names.contains(source_unit.as_str()))
            })
        })
    }

    #[derive(Clone)]
    struct ParseCounter(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ParseCounter {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            _id: &tracing::span::Id,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if attrs.metadata().name() == "SolData::parse" {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn compiler_cache_reuses_source_analysis_and_revalidates_inputs() {
        let count = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing_subscriber::layer::SubscriberExt::with(
            tracing_subscriber::Registry::default(),
            ParseCounter(count.clone()),
        );
        tracing::subscriber::with_default(subscriber, || {
            let mut input = serde_json::json!({
                "language": "Solidity",
                "sources": {
                    "reuse/Analysis.sol": {"content": "import './Dep.sol'; contract Analysis {}"},
                    "reuse/Dep.sol": {"content": "contract AnalysisDependency {}"},
                },
                "settings": {"optimizer": {"runs": 1234567}},
            });
            let eligible = |input: &serde_json::Value| {
                is_cacheable_compiler_input(&serde_json::to_vec(input).unwrap())
            };
            assert!(eligible(&input));
            assert_eq!(count.swap(0, Ordering::Relaxed), 2);
            assert!(eligible(&input));
            assert_eq!(count.swap(0, Ordering::Relaxed), 0, "warm input reparsed");
            input["settings"]["optimizer"]["runs"] = 7654321.into();
            assert!(eligible(&input));
            assert_eq!(count.swap(0, Ordering::Relaxed), 0, "changed settings reparsed sources");
            input["settings"]["remappings"] = serde_json::json!(["reuse/=missing/"]);
            assert!(!eligible(&input));
            input["settings"].as_object_mut().unwrap().remove("remappings");
            input["sources"].as_object_mut().unwrap().remove("reuse/Dep.sol");
            assert!(!eligible(&input));
            input["sources"]["reuse/Analysis.sol"]["content"] = "contract Changed {}".into();
            assert!(eligible(&input));
            assert_eq!(count.swap(0, Ordering::Relaxed), 1);
            input["sources"]["reuse/Analysis.sol"]["content"] = "contract Broken {".into();
            assert!(!eligible(&input));
            assert_eq!(count.swap(0, Ordering::Relaxed), 1);
            assert!(!eligible(&input));
            input["settings"]["optimizer"]["runs"] = 9999999.into();
            assert!(!eligible(&input));
            assert_eq!(count.load(Ordering::Relaxed), 0, "invalid source reparsed");
        });
    }

    #[test]
    fn compiler_cache_input_memo_preserves_json_validation() {
        let inputs = [
            r#"{"language":"Solidity","sources":{}}"#,
            r#"{"language":"Yul","language":"Solidity","sources":{}}"#,
            r#"{"language":"Solidity","sources":{"A.sol":{"content":"contract A {}","urls":["unused"]}},"settings":{"unknown":{"large":[1,2,3]}}}"#,
            r#"{"language":"Solidity","sources":{},"settings":null}"#,
            r#"{"language":"Solidity","sources":{},"settings":{"remappings":null}}"#,
            r#"{"language":"Solidity","sources":{"A.sol":{"content":null}}}"#,
            r#"{"language":"Solidity","sources":{"A.sol":{"content":"contract A {}","content":"contract Broken {"}}}"#,
            r#"{"language":"Solidity","sources":{"A.sol":{"urls":["A.sol"]}}}"#,
            r#"{"language":"Solidity","sources":{},"settings":{"remappings":[42]}}"#,
            r#"{"language":"Solidity","sources":{}} trailing"#,
        ];
        for input in inputs {
            let expected = uncached_compiler_input(input.as_bytes());
            assert_eq!(is_cacheable_compiler_input(input.as_bytes()), expected, "{input}");
            assert_eq!(is_cacheable_compiler_input(input.as_bytes()), expected, "warm {input}");
        }
    }

    #[test]
    fn compiler_cache_source_metadata_has_count_and_byte_limits() {
        let mut analyses = SourceAnalyses::default();
        for index in 0..=MAX_SOURCE_ANALYSES {
            analyses.insert(keccak256(index.to_be_bytes()), None);
            assert!(analyses.imports.len() <= MAX_SOURCE_ANALYSES);
            assert!(analyses.bytes <= MAX_SOURCE_ANALYSIS_BYTES);
        }
        assert_eq!(analyses.imports.len(), 1);
        let oversized = Some(Arc::from([PathBuf::from("x".repeat(MAX_SOURCE_ANALYSIS_BYTES))]));
        let previous_bytes = analyses.bytes;
        analyses.insert(keccak256(b"oversized"), oversized);
        assert_eq!(analyses.imports.len(), 1);
        assert_eq!(analyses.bytes, previous_bytes);
        for index in 0_usize..4 {
            analyses.insert(
                keccak256(index.to_be_bytes()),
                Some(Arc::from([PathBuf::from("x".repeat(MAX_SOURCE_ANALYSIS_BYTES / 2))])),
            );
            assert!(analyses.bytes <= MAX_SOURCE_ANALYSIS_BYTES);
        }
        assert!(analyses.imports.len() <= 2);
    }

    #[test]
    fn compiler_cache_concurrent_source_analysis_preserves_imports() {
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..16 {
                        let imports = cached_source_imports(
                            "concurrent/Cache.sol",
                            "import './Dep.sol'; contract ConcurrentCache {}",
                        )
                        .unwrap();
                        assert_eq!(imports.as_ref(), [PathBuf::from("./Dep.sol")]);
                        assert!(
                            cached_source_imports("concurrent/Cache.sol", "contract Invalid {")
                                .is_none()
                        );
                    }
                });
            }
        });
        let first = cached_source_imports(
            "concurrent/Cache.sol",
            "import './Dep.sol'; contract ConcurrentCache {}",
        )
        .unwrap();
        let second = cached_source_imports(
            "concurrent/Cache.sol",
            "import './Dep.sol'; contract ConcurrentCache {}",
        )
        .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    #[ignore = "manual compiler cache microbenchmark"]
    fn compiler_cache_eligibility_benchmark() {
        let content = format!(
            "contract A {{ {} }}",
            (0..1000)
                .map(|index| format!(
                    "function f{index}() external pure returns (uint) {{ return {index}; }}"
                ))
                .collect::<String>()
        );
        let input = serde_json::to_vec(&serde_json::json!({
            "language": "Solidity", "sources": {"src/Benchmark.sol": {"content": content}},
            "settings": {"optimizer": {"runs": 200}},
        }))
        .unwrap();
        let key = input_cache_key(&input);
        assert!(super::is_cacheable_compiler_input(&input, key));
        let start = std::time::Instant::now();
        for _ in 0..128 {
            assert!(is_cacheable_compiler_input(std::hint::black_box(&input)));
        }
        eprintln!("warm key plus eligibility: 128 x {} bytes {:?}", input.len(), start.elapsed());
        let start = std::time::Instant::now();
        for _ in 0..128 {
            std::hint::black_box(input_cache_key(std::hint::black_box(&input)));
            assert!(uncached_compiler_input(std::hint::black_box(&input)));
        }
        eprintln!(
            "uncached key plus eligibility: 128 x {} bytes {:?}",
            input.len(),
            start.elapsed()
        );
        let start = std::time::Instant::now();
        for _ in 0..128 {
            assert!(super::is_cacheable_compiler_input(std::hint::black_box(&input), key));
        }
        eprintln!(
            "cache eligibility (precomputed invocation key): 128 x {} bytes {:?}",
            input.len(),
            start.elapsed()
        );
        let start = std::time::Instant::now();
        for _ in 0..128 {
            assert!(uncached_compiler_input(std::hint::black_box(&input)));
        }
        eprintln!(
            "uncached eligibility (precomputed invocation key): 128 x {} bytes {:?}",
            input.len(),
            start.elapsed()
        );
        let variants = (0..128)
            .map(|runs| {
                let mut input: serde_json::Value = serde_json::from_slice(&input).unwrap();
                input["settings"]["optimizer"]["runs"] = runs.into();
                let input = serde_json::to_vec(&input).unwrap();
                let key = input_cache_key(&input);
                (input, key)
            })
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        for (variant, key) in &variants {
            assert!(super::is_cacheable_compiler_input(variant, *key));
        }
        eprintln!(
            "settings-only eligibility (precomputed invocation keys): 128 variants {:?}",
            start.elapsed()
        );
        let mut cold_elapsed = std::time::Duration::ZERO;
        for _ in 0..128 {
            INPUT_ANALYSES.lock().unwrap().clear();
            *SOURCE_ANALYSES.lock().unwrap() = SourceAnalyses::default();
            let start = std::time::Instant::now();
            assert!(super::is_cacheable_compiler_input(std::hint::black_box(&input), key));
            cold_elapsed += start.elapsed();
        }
        eprintln!(
            "cold eligibility (precomputed invocation key): 128 x {} bytes {:?}",
            input.len(),
            cold_elapsed
        );
    }

    #[test]
    fn compiler_identity_cache_avoids_warm_rehashes_and_detects_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("solc");
        let output_cache = Arc::new(LocalCompilerOutputCache::new(directory.path().join("cache")));
        let input = serde_json::to_vec(&serde_json::json!({
            "language": "Solidity",
            "sources": { "src/A.sol": { "content": "contract A {}" } },
            "settings": {},
        }))
        .unwrap();
        std::fs::write(&executable, b"first compiler identity").unwrap();
        let original_modified = std::fs::metadata(&executable).unwrap().modified().unwrap();

        let mut first = Solc::new_with_version(&executable, Version::new(0, 8, 35));
        first.set_compiler_output_cache(output_cache.clone());
        let identity_path = first.solc.clone();
        let (first_key, _) = first.compiler_output_cache_key(&input).unwrap();
        for _ in 1..128 {
            let mut warm = Solc::new_with_version(&executable, Version::new(0, 8, 35));
            warm.set_compiler_output_cache(output_cache.clone());
            assert_eq!(first_key, warm.compiler_output_cache_key(&input).unwrap().0);
        }
        assert_eq!(COMPILER_IDENTITY_READS.lock().unwrap()[&identity_path], 1);

        std::fs::write(&executable, b"other compiler identity").unwrap();
        std::fs::File::open(&executable)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(original_modified))
            .unwrap();

        let mut replaced = Solc::new_with_version(&executable, Version::new(0, 8, 35));
        replaced.set_compiler_output_cache(output_cache);
        let (replaced_key, _) = replaced.compiler_output_cache_key(&input).unwrap();
        assert_ne!(first_key, replaced_key);
        assert_eq!(COMPILER_IDENTITY_READS.lock().unwrap()[&identity_path], 2);
    }

    #[test]
    fn compiler_aliases_preserve_invocation_and_cache_identity() {
        let dir = tempfile::tempdir().unwrap();
        let executable = dir.path().join("dispatch");
        std::fs::write(
            &executable,
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo 'Version: 0.8.35'
else
    cat >/dev/null
    printf '{"invoked":"%s"}' "${0##*/}"
fi
"#,
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let cache = Arc::new(LocalCompilerOutputCache::new(dir.path().join("cache")));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": { "A.sol": { "content": "contract A {}" } },
        });
        for name in ["first", "second"] {
            let alias = dir.path().join(name);
            symlink(&executable, &alias).unwrap();
            let mut solc = Solc::new(&alias).unwrap();
            let expected = serde_json::json!({ "invoked": name });
            assert_eq!(solc.compile_as::<_, serde_json::Value>(&input).unwrap(), expected);
            solc.set_compiler_output_cache(cache.clone());
            assert_eq!(solc.compile_as::<_, serde_json::Value>(&input).unwrap(), expected);
            assert_eq!(solc.compile_as::<_, serde_json::Value>(&input).unwrap(), expected);
        }
        assert_eq!(cache.stats().entry_count, 2);
    }

    fn check_compiler_replacement(
        compile: impl Fn(&Solc, &serde_json::Value) -> serde_json::Value,
    ) {
        #[derive(Debug)]
        struct ReplacingCache {
            store: LocalCompilerOutputCache,
            executable: PathBuf,
            replacement: PathBuf,
            replaced: std::sync::atomic::AtomicBool,
        }
        impl CompilerOutputCache for ReplacingCache {
            fn read(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
                if !self.replaced.swap(true, Ordering::Relaxed) {
                    std::fs::copy(&self.replacement, &self.executable).unwrap();
                }
                self.store.read(key)
            }
            fn write(&self, key: &CompilerOutputCacheKey, output: &[u8]) {
                self.store.write(key, output);
            }
            fn invalidate(&self, key: &CompilerOutputCacheKey) {
                self.store.invalidate(key);
            }
        }
        for restore_during_run in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let executable = dir.path().join("solc");
            let replacement = dir.path().join("replacement");
            let script = |marker| {
                format!(
                    r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo 'Version: 0.8.35'
else
    cat >/dev/null
    echo '{{"marker":"{marker}"}}'
fi
"#
                )
            };
            std::fs::write(&executable, script("first")).unwrap();
            let replacement_script = if restore_during_run {
                script("second").replace(
                    "cat >/dev/null",
                    "cat >/dev/null\n    trap 'cp \"$0.original\" \"$0\"' EXIT",
                )
            } else {
                script("second")
            };
            std::fs::write(&replacement, replacement_script).unwrap();
            std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o755)).unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
            std::fs::copy(&executable, executable.with_extension("original")).unwrap();
            let cache = Arc::new(ReplacingCache {
                store: LocalCompilerOutputCache::new(dir.path().join("cache")),
                executable: executable.clone(),
                replacement,
                replaced: std::sync::atomic::AtomicBool::new(false),
            });
            let mut solc = Solc::new(&executable).unwrap();
            solc.set_compiler_output_cache(cache.clone());
            let input = serde_json::json!({
                "language": "Solidity",
                "sources": { "A.sol": { "content": "contract A {}" } },
            });
            assert_eq!(compile(&solc, &input)["marker"], "second");
            assert_eq!(cache.store.stats().entry_count, 0);
            std::fs::write(&executable, script("first")).unwrap();
            assert_eq!(compile(&solc, &input)["marker"], "first");
        }
    }

    #[test]
    fn compiler_replacement_during_lookup_does_not_poison_cache() {
        check_compiler_replacement(|solc, input| solc.compile_as(input).unwrap());
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_compiler_replacement_during_lookup_does_not_poison_cache() {
        let runtime = foundry_compilers_core::utils::RuntimeOrHandle::new();
        check_compiler_replacement(|solc, input| {
            runtime.block_on(solc.async_compile_as(input)).unwrap()
        });
    }

    #[test]
    fn retargeted_compiler_alias_invalidates_cached_output() {
        let dir = tempfile::tempdir().unwrap();
        let alias = dir.path().join("solc");
        let cache = Arc::new(LocalCompilerOutputCache::new(dir.path().join("cache")));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": { "A.sol": { "content": "contract A {}" } },
        });
        let mut solc = Solc::new_with_version(&alias, Version::new(0, 8, 35));
        solc.set_compiler_output_cache(cache.clone());
        for (marker, patch) in [("first", 35), ("second", 36)] {
            let target = dir.path().join(marker);
            std::fs::write(
                &target,
                format!(
                    r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo 'Version: 0.8.{patch}'
else
    cat >/dev/null
    echo '{{"marker":"{marker}"}}'
fi
"#
                ),
            )
            .unwrap();
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();
            if alias.exists() {
                std::fs::remove_file(&alias).unwrap();
            }
            symlink(&target, &alias).unwrap();
            assert_eq!(Solc::new(&alias).unwrap().version, Version::new(0, 8, patch));
            assert_eq!(solc.compile_as::<_, serde_json::Value>(&input).unwrap()["marker"], marker);
        }
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn retargeted_compiler_directory_refreshes_version() {
        let dir = tempfile::tempdir().unwrap();
        let alias = dir.path().join("current");
        for patch in [35, 36] {
            let target = dir.path().join(patch.to_string());
            std::fs::create_dir(&target).unwrap();
            let executable = target.join("solc");
            std::fs::write(&executable, format!("#!/bin/sh\necho 'Version: 0.8.{patch}'\n"))
                .unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
            if alias.exists() {
                std::fs::remove_file(&alias).unwrap();
            }
            symlink(&target, &alias).unwrap();
            assert_eq!(Solc::new(alias.join("solc")).unwrap().version, Version::new(0, 8, patch));
        }
    }

    #[test]
    fn compiler_output_cache_replays_raw_json_without_running_solc() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    touch "{}"
    echo '{{"contracts":{{}}}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });

        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        std::fs::remove_file(&invocations).unwrap();
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, second);
        assert!(!invocations.exists());
    }

    #[test]
    fn compiler_output_cache_rekeys_identical_wrappers_by_full_version() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
echo x >> "{}"
echo '{{"contracts":{{}}}}'
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new_with_version(
            &solc_path,
            Version::parse("0.8.35+first-wrapper-target").unwrap(),
        );
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let _ = std::fs::remove_file(&invocations);
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });
        let serialized_input = serde_json::to_vec(&input).unwrap();
        let (first_key, _) = solc.compiler_output_cache_key(&serialized_input).unwrap();

        let _: serde_json::Value = solc.compile_as(&input).unwrap();
        solc.version = Version::parse("0.8.35+second-wrapper-target").unwrap();
        let (second_key, _) = solc.compiler_output_cache_key(&serialized_input).unwrap();
        let _: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_ne!(first_key, second_key);
        assert_eq!(std::fs::read_to_string(invocations).unwrap().lines().count(), 2);
    }

    #[test]
    fn compiler_output_cache_keys_the_exact_json_sent_to_solc() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    echo x >> "{}"
    cat
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));

        let input = FlippingInput(AtomicUsize::new(1));
        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        let second: serde_json::Value = solc.compile_as(&input).unwrap();
        let cached: serde_json::Value = solc
            .compile_as(&serde_json::json!({
                "language": "Solidity",
                "sources": { "src/A.sol": { "content": "contract A {} // 2" } },
                "settings": { "optimizer": { "enabled": false, "runs": 2 } },
            }))
            .unwrap();

        assert_eq!(
            first,
            serde_json::json!({
                "language": "Solidity",
                "sources": { "src/A.sol": { "content": "contract A {} // 1" } },
                "settings": { "optimizer": { "enabled": false, "runs": 1 } },
            })
        );
        assert_eq!(
            second,
            serde_json::json!({
                "language": "Solidity",
                "sources": { "src/A.sol": { "content": "contract A {} // 2" } },
                "settings": { "optimizer": { "enabled": false, "runs": 2 } },
            })
        );
        assert_eq!(cached, second);
        assert_eq!(std::fs::read_to_string(invocations).unwrap().lines().count(), 2);
    }

    #[test]
    fn compiler_output_cache_repairs_json_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    echo x >> "{}"
    echo '{{"contracts":{{}}}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        let store = Arc::new(LocalCompilerOutputCache::new(dir.path().join("cache")));
        solc.set_compiler_output_cache(store.clone());
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": { "src/A.sol": { "content": "contract A {}" } },
        });
        let (key, _) =
            solc.compiler_output_cache_key(&serde_json::to_vec(&input).unwrap()).unwrap();
        store.write(&key, b"not json");

        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, serde_json::json!({ "contracts": {} }));
        assert_eq!(second, first);
        assert_eq!(std::fs::read_to_string(invocations).unwrap().lines().count(), 1);
    }

    #[test]
    fn compiler_output_cache_bypasses_sources_without_inline_content() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    echo x >> "{}"
    echo '{{}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": { "src/A.sol": { "urls": ["src/A.sol"] } },
        });

        let _: serde_json::Value = solc.compile_as(&input).unwrap();
        let _: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(std::fs::read_to_string(invocations).unwrap().lines().count(), 2);
    }

    #[test]
    fn compiler_output_cache_bypasses_omitted_imports_across_worktrees() {
        let dir = tempfile::tempdir().unwrap();
        let first_root = dir.path().join("first");
        let second_root = dir.path().join("second");
        for (root, output) in
            [(&first_root, r#"{"root":"first"}"#), (&second_root, r#"{"root":"second"}"#)]
        {
            let dependency = root.join("src/Dependency.sol");
            std::fs::create_dir_all(dependency.parent().unwrap()).unwrap();
            std::fs::write(dependency, output).unwrap();
        }

        let solc_path = dir.path().join("solc");
        std::fs::write(
            &solc_path,
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--base-path" ]; then
        cat "$2/src/Dependency.sol"
        exit 0
    fi
    shift
done
exit 1
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": { "src/A.sol": { "content": "import './Dependency.sol';" } },
        });

        solc.base_path = Some(first_root);
        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        solc.base_path = Some(second_root);
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, serde_json::json!({ "root": "first" }));
        assert_eq!(second, serde_json::json!({ "root": "second" }));
    }

    #[test]
    fn compiler_output_cache_bypasses_commented_imports() {
        let dir = tempfile::tempdir().unwrap();
        let first_root = dir.path().join("first");
        let second_root = dir.path().join("second");
        for (root, output) in
            [(&first_root, r#"{"root":"first"}"#), (&second_root, r#"{"root":"second"}"#)]
        {
            let dependency = root.join("src/Dependency.sol");
            std::fs::create_dir_all(dependency.parent().unwrap()).unwrap();
            std::fs::write(dependency, output).unwrap();
        }

        let solc_path = dir.path().join("solc");
        std::fs::write(
            &solc_path,
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--base-path" ]; then
        cat "$2/src/Dependency.sol"
        exit 0
    fi
    shift
done
exit 1
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": {
                "src/A.sol": { "content": "import /* generated */ './Dependency.sol';" },
            },
        });

        solc.base_path = Some(first_root);
        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        solc.base_path = Some(second_root);
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, serde_json::json!({ "root": "first" }));
        assert_eq!(second, serde_json::json!({ "root": "second" }));
    }

    #[test]
    fn compiler_output_cache_bypasses_bare_imports_with_relative_decoys() {
        let dir = tempfile::tempdir().unwrap();
        let first_root = dir.path().join("first");
        let second_root = dir.path().join("second");
        for (root, output) in
            [(&first_root, r#"{"root":"first"}"#), (&second_root, r#"{"root":"second"}"#)]
        {
            std::fs::create_dir_all(root).unwrap();
            std::fs::write(root.join("Dependency.sol"), output).unwrap();
        }

        let solc_path = dir.path().join("solc");
        std::fs::write(
            &solc_path,
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--base-path" ]; then
        cat "$2/Dependency.sol"
        exit 0
    fi
    shift
done
exit 1
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": {
                "src/A.sol": { "content": "import 'Dependency.sol';" },
                "src/Dependency.sol": { "content": "contract Decoy {}" },
            },
        });

        solc.base_path = Some(first_root);
        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        solc.base_path = Some(second_root);
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, serde_json::json!({ "root": "first" }));
        assert_eq!(second, serde_json::json!({ "root": "second" }));
    }

    #[test]
    fn compiler_output_cache_bypasses_remapped_imports_with_direct_decoys() {
        let dir = tempfile::tempdir().unwrap();
        let first_root = dir.path().join("first");
        let second_root = dir.path().join("second");
        for (root, output) in
            [(&first_root, r#"{"root":"first"}"#), (&second_root, r#"{"root":"second"}"#)]
        {
            let dependency = root.join("lib/foo/A.sol");
            std::fs::create_dir_all(dependency.parent().unwrap()).unwrap();
            std::fs::write(dependency, output).unwrap();
        }

        let solc_path = dir.path().join("solc");
        std::fs::write(
            &solc_path,
            r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
    exit 0
fi
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--base-path" ]; then
        cat "$2/lib/foo/A.sol"
        exit 0
    fi
    shift
done
exit 1
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": {
                "src/A.sol": { "content": "import 'foo/A.sol';" },
                "foo/A.sol": { "content": "contract Decoy {}" },
            },
            "settings": { "remappings": ["foo/=lib/foo/"] },
        });

        solc.base_path = Some(first_root);
        let first: serde_json::Value = solc.compile_as(&input).unwrap();
        solc.base_path = Some(second_root);
        let second: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(first, serde_json::json!({ "root": "first" }));
        assert_eq!(second, serde_json::json!({ "root": "second" }));
    }

    #[test]
    fn compiler_output_cache_reuses_a_path_resolved_executable() {
        let _lock = PATH_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let executable = dir.path().join("solc-on-path");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &executable,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    touch "{}"
    echo '{{}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();

        let original_path = env::var_os("PATH");
        let path = env::join_paths(
            std::iter::once(dir.path().to_path_buf())
                .chain(env::split_paths(original_path.as_deref().unwrap_or_default())),
        )
        .unwrap();
        // SAFETY: PATH_LOCK serializes this test's process-wide environment mutation.
        unsafe { env::set_var("PATH", path) };

        let mut solc = Solc::new("solc-on-path").unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });
        let _: serde_json::Value = solc.compile_as(&input).unwrap();
        std::fs::remove_file(&invocations).unwrap();
        let _: serde_json::Value = solc.compile_as(&input).unwrap();

        // SAFETY: Restore the process-wide PATH before releasing PATH_LOCK.
        unsafe {
            if let Some(path) = original_path {
                env::set_var("PATH", path);
            } else {
                env::remove_var("PATH");
            }
        }
        assert!(!invocations.exists());
    }

    #[test]
    fn compiler_output_cache_bypasses_path_overrides_in_extra_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
for arg in "$@"; do
    if [ "$arg" = "--version" ]; then
        echo "solc, the solidity compiler commandline interface"
        echo "Version: 0.8.35+commit.69074fbd"
        exit 0
    fi
done
echo x >> "{}"
echo '{{}}'
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new_with_args(&solc_path, ["--base-path", "/project"]).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });

        let _: serde_json::Value = solc.compile_as(&input).unwrap();
        let _: serde_json::Value = solc.compile_as(&input).unwrap();

        assert_eq!(std::fs::read_to_string(invocations).unwrap().lines().count(), 2);
    }

    #[test]
    fn compiler_output_cache_reuses_graph_expanded_remapping_imports() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    touch "{}"
    echo '{{}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": {
                "src/A.sol": { "content": "import 'forge-std/Test.sol';" },
                "lib/forge-std/src/Test.sol": { "content": "contract Test {}" },
            },
            "settings": { "remappings": ["forge-std/=lib/forge-std/src/"] },
        });

        let _: serde_json::Value = solc.compile_as(&input).unwrap();
        std::fs::remove_file(&invocations).unwrap();
        let _: serde_json::Value = solc.compile_as(&input).unwrap();

        assert!(!invocations.exists());
    }

    #[test]
    fn compiler_output_cache_admits_only_canonical_source_unit_closures() {
        let canonical = |source_name: &str, import: &str, dependency: &str, remappings: &[&str]| {
            serde_json::json!({
                "language": "Solidity",
                "sources": {
                    source_name: { "content": format!("import '{import}';") },
                    dependency: { "content": "contract Dependency {}" },
                },
                "settings": { "remappings": remappings },
            })
        };
        let cases = [
            (
                "direct source unit",
                canonical("src/A.sol", "lib/Dependency.sol", "lib/Dependency.sol", &[]),
                true,
            ),
            (
                "relative source unit is remapped after resolution",
                canonical("src/A.sol", "./Dependency.sol", "lib/Dependency.sol", &["src/=lib/"]),
                true,
            ),
            (
                "canonical forge std remapping",
                canonical(
                    "src/A.sol",
                    "forge-std/Test.sol",
                    "lib/forge-std/src/Test.sol",
                    &["forge-std/=lib/forge-std/src/"],
                ),
                true,
            ),
            (
                "longest prefix remapping wins",
                canonical(
                    "src/A.sol",
                    "forge-std/Test.sol",
                    "lib/forge-std/special/Test.sol",
                    &[
                        "forge-std/=lib/forge-std/src/",
                        "forge-std/Test.sol=lib/forge-std/special/Test.sol",
                    ],
                ),
                true,
            ),
            (
                "last equal prefix remapping wins",
                canonical(
                    "src/A.sol",
                    "forge-std/Test.sol",
                    "lib/forge-std/last/Test.sol",
                    &["forge-std/=lib/forge-std/first/", "forge-std/=lib/forge-std/last/"],
                ),
                true,
            ),
            (
                "source unit dot segment",
                canonical("src/./A.sol", "lib/Dependency.sol", "lib/Dependency.sol", &[]),
                false,
            ),
            (
                "direct import dot segment",
                canonical("src/A.sol", "lib/./Dependency.sol", "lib/Dependency.sol", &[]),
                false,
            ),
            (
                "absolute import dot segment",
                canonical("src/A.sol", "/lib/./Dependency.sol", "/lib/Dependency.sol", &[]),
                false,
            ),
            (
                "repeated separators",
                canonical("src/A.sol", "lib//Dependency.sol", "lib/Dependency.sol", &[]),
                false,
            ),
            (
                "url source unit",
                canonical(
                    "https://example.test/A.sol",
                    "lib/Dependency.sol",
                    "lib/Dependency.sol",
                    &[],
                ),
                false,
            ),
            (
                "excess parent traversal",
                canonical("src/A.sol", "../../Dependency.sol", "Dependency.sol", &[]),
                false,
            ),
            (
                "contextual remapping",
                canonical(
                    "src/A.sol",
                    "forge-std/Test.sol",
                    "lib/forge-std/src/Test.sol",
                    &["src/:forge-std/=lib/forge-std/src/"],
                ),
                true,
            ),
            (
                "noncanonical remapping target",
                canonical(
                    "src/A.sol",
                    "forge-std/Test.sol",
                    "lib/forge-std/src/Test.sol",
                    &["forge-std/=lib/forge-std/./src/"],
                ),
                false,
            ),
        ];

        for (name, input, expected) in cases {
            let input = serde_json::to_vec(&input).unwrap();
            assert_eq!(uncached_compiler_input(&input), expected, "legacy {name}");
            assert_eq!(is_cacheable_compiler_input(&input), expected, "{name}");
            assert_eq!(is_cacheable_compiler_input(&input), expected, "warm {name}");
        }
    }

    fn closure_input(sources: &[(&str, &str)], remappings: &[&str]) -> Vec<u8> {
        let sources = sources
            .iter()
            .map(|(name, content)| ((*name).to_owned(), serde_json::json!({ "content": content })))
            .collect::<serde_json::Map<_, _>>();
        serde_json::to_vec(&serde_json::json!({
            "language": "Solidity",
            "sources": sources,
            "settings": { "remappings": remappings },
        }))
        .unwrap()
    }

    fn assert_closure_cases(cases: &[(&str, Vec<u8>, bool)]) {
        for (name, input, expected) in cases {
            assert_eq!(uncached_compiler_input(input), *expected, "legacy {name}");
            assert_eq!(is_cacheable_compiler_input(input), *expected, "{name}");
            assert_eq!(is_cacheable_compiler_input(input), *expected, "warm {name}");
        }
    }

    #[test]
    fn compiler_output_cache_context_remapping_applies_only_inside_its_context() {
        let remappings = ["lib/up/:erc4626-tests/=lib/up/lib/erc4626-tests/"].as_slice();
        assert_closure_cases(&[
            (
                "inside context",
                closure_input(
                    &[
                        ("lib/up/test/A.sol", "import 'erc4626-tests/T.sol';"),
                        ("lib/up/lib/erc4626-tests/T.sol", "contract T {}"),
                    ],
                    remappings,
                ),
                true,
            ),
            (
                "outside context resolves without remapping",
                closure_input(
                    &[
                        ("src/A.sol", "import 'erc4626-tests/T.sol';"),
                        ("erc4626-tests/T.sol", "contract T {}"),
                        ("lib/up/lib/erc4626-tests/T.sol", "contract T {}"),
                    ],
                    remappings,
                ),
                true,
            ),
            (
                "outside context does not reach the remapped target",
                closure_input(
                    &[
                        ("src/A.sol", "import 'erc4626-tests/T.sol';"),
                        ("lib/up/lib/erc4626-tests/T.sol", "contract T {}"),
                    ],
                    remappings,
                ),
                false,
            ),
            (
                "context is a plain string prefix of the importer",
                closure_input(
                    &[
                        ("lib/upgrades/A.sol", "import 'erc4626-tests/T.sol';"),
                        ("lib/up/lib/erc4626-tests/T.sol", "contract T {}"),
                    ],
                    &["lib/up:erc4626-tests/=lib/up/lib/erc4626-tests/"],
                ),
                true,
            ),
            (
                "relative import is remapped in the importer context",
                closure_input(
                    &[
                        ("lib/up/src/A.sol", "import '../vendor/T.sol';"),
                        ("lib/up/lib/vendor/T.sol", "contract T {}"),
                    ],
                    &["lib/up/:lib/up/vendor/=lib/up/lib/vendor/"],
                ),
                true,
            ),
            (
                "empty context is global",
                closure_input(
                    &[("src/A.sol", "import 'oz/T.sol';"), ("lib/oz/T.sol", "contract T {}")],
                    &[":oz/=lib/oz/"],
                ),
                true,
            ),
        ]);
    }

    #[test]
    fn compiler_output_cache_context_remapping_selection_matches_solc() {
        let importer = ("lib/up/src/A.sol", "import 'oz/T.sol';");
        let case = |name, dependency: &str, remappings: &[&str], expected| {
            (name, closure_input(&[importer, (dependency, "contract T {}")], remappings), expected)
        };
        assert_closure_cases(&[
            case(
                "context beats a longer global prefix",
                "lib/up/lib/oz/T.sol",
                &["oz/T.sol=lib/oz/T.sol", "lib/up/:oz/=lib/up/lib/oz/"],
                true,
            ),
            case(
                "longer global prefix loses to context",
                "lib/oz/T.sol",
                &["oz/T.sol=lib/oz/T.sol", "lib/up/:oz/=lib/up/lib/oz/"],
                false,
            ),
            case(
                "longest context wins when listed first",
                "lib/long/T.sol",
                &["lib/up/src/:oz/=lib/long/", "lib/:oz/=lib/short/"],
                true,
            ),
            case(
                "longest context wins when listed last",
                "lib/long/T.sol",
                &["lib/:oz/=lib/short/", "lib/up/src/:oz/=lib/long/"],
                true,
            ),
            case(
                "shorter context loses",
                "lib/short/T.sol",
                &["lib/:oz/=lib/short/", "lib/up/src/:oz/=lib/long/"],
                false,
            ),
            case(
                "longest prefix wins within a context",
                "lib/special/T.sol",
                &["lib/up/:oz/T.sol=lib/special/T.sol", "lib/up/:oz/=lib/up/lib/oz/"],
                true,
            ),
            case(
                "last equal context and prefix wins",
                "lib/last/T.sol",
                &["lib/up/:oz/=lib/first/", "lib/up/:oz/=lib/last/"],
                true,
            ),
            case(
                "first equal context and prefix loses",
                "lib/first/T.sol",
                &["lib/up/:oz/=lib/first/", "lib/up/:oz/=lib/last/"],
                false,
            ),
            case(
                "context remapping to a missing source",
                "lib/oz/T.sol",
                &["oz/=lib/oz/", "lib/up/:oz/=lib/up/lib/oz/"],
                false,
            ),
        ]);
    }

    #[test]
    fn compiler_output_cache_global_and_context_remappings_coexist() {
        let remappings = ["oz/=lib/oz/", "up/=lib/up/src/", "lib/up/:oz/=lib/up/lib/oz/"];
        let sources = [
            ("src/A.sol", "import 'oz/Token.sol'; import 'up/Up.sol';"),
            ("lib/oz/Token.sol", "contract Token {}"),
            ("lib/up/src/Up.sol", "import 'oz/Token.sol';"),
            ("lib/up/lib/oz/Token.sol", "contract UpToken {}"),
        ];
        let without = |missing: &str| {
            sources.iter().copied().filter(|(name, _)| *name != missing).collect::<Vec<_>>()
        };
        assert_closure_cases(&[
            ("closed", closure_input(&sources, &remappings), true),
            (
                "missing global target",
                closure_input(&without("lib/oz/Token.sol"), &remappings),
                false,
            ),
            (
                "missing context target",
                closure_input(&without("lib/up/lib/oz/Token.sol"), &remappings),
                false,
            ),
        ]);
    }

    #[test]
    fn compiler_output_cache_rejects_malformed_context_remappings() {
        // Closed whether or not a remapping applies, so only rejecting the remapping bypasses.
        let sources = [
            ("lib/up/src/A.sol", "import 'oz/T.sol';"),
            ("lib/oz/T.sol", "contract T {}"),
            ("oz/T.sol", "contract T {}"),
        ];
        let mut cases =
            vec![("well formed", closure_input(&sources, &["lib/up/:oz/=lib/oz/"]), true)];
        for remapping in [
            "/lib/up/:oz/=lib/oz/",
            "../lib/up/:oz/=lib/oz/",
            "lib/../up/:oz/=lib/oz/",
            "lib/./up/:oz/=lib/oz/",
            "lib//up/:oz/=lib/oz/",
            "lib\\up\\:oz/=lib/oz/",
            "lib/up/:=lib/oz/",
            "lib/up/:oz:extra/=lib/oz/",
            "lib/up/:oz/=/lib/oz/",
            "lib/up/:oz/=lib/oz/../oz/",
            "lib/up/:oz/=",
        ] {
            cases.push((remapping, closure_input(&sources, &[remapping]), false));
        }
        assert_closure_cases(&cases);
    }

    /// Inputs that compile only if the compiler selects the context remapping the cache expects.
    fn context_remapping_resolution_cases() -> Vec<(&'static str, serde_json::Value)> {
        let input = |sources: serde_json::Value, remappings: &[&str]| {
            serde_json::json!({
                "language": "Solidity",
                "sources": sources,
                "settings": { "remappings": remappings },
            })
        };
        vec![
            (
                "context_scoped",
                input(
                    serde_json::json!({
                        "src/A.sol": {
                            "content": "import 'oz/Token.sol'; import 'up/Up.sol'; contract A is Token, Up {}"
                        },
                        "lib/oz/Token.sol": { "content": "contract Token {}" },
                        "lib/up/src/Up.sol": {
                            "content": "import 'oz/Token.sol'; contract Up is UpToken {}"
                        },
                        "lib/up/lib/oz/Token.sol": { "content": "contract UpToken {}" },
                    }),
                    &["oz/=lib/oz/", "up/=lib/up/src/", "lib/up/:oz/=lib/up/lib/oz/"],
                ),
            ),
            (
                "context_beats_longer_prefix",
                input(
                    serde_json::json!({
                        "lib/up/src/Up.sol": {
                            "content": "import 'oz/Token.sol'; contract Up is Dependency {}"
                        },
                        "lib/oz/Token.sol": { "content": "contract Fallback {}" },
                        "lib/up/lib/oz/Token.sol": { "content": "contract Dependency {}" },
                    }),
                    &["lib/up/:oz/=lib/up/lib/oz/", "oz/Token.sol=lib/oz/Token.sol"],
                ),
            ),
            (
                "longest_context",
                input(
                    serde_json::json!({
                        "lib/up/src/Up.sol": {
                            "content": "import 'oz/Token.sol'; contract Up is Dependency {}"
                        },
                        "lib/long/Token.sol": { "content": "contract Dependency {}" },
                        "lib/short/Token.sol": { "content": "contract Fallback {}" },
                    }),
                    &["lib/up/src/:oz/=lib/long/", "lib/:oz/=lib/short/"],
                ),
            ),
            (
                "string_prefix_context",
                input(
                    serde_json::json!({
                        "lib/upgrades/A.sol": {
                            "content": "import 'oz/Token.sol'; contract A is Dependency {}"
                        },
                        "oz/Token.sol": { "content": "contract Fallback {}" },
                        "lib/up/lib/oz/Token.sol": { "content": "contract Dependency {}" },
                    }),
                    &["lib/up:oz/=lib/up/lib/oz/"],
                ),
            ),
        ]
    }

    #[cfg(feature = "svm-solc")]
    #[test]
    fn compiler_output_cache_real_solc_resolution_matrix() {
        let Some(solc) = Solc::find_svm_installed_version(&Version::new(0, 8, 35)).unwrap() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let cases = [
            (
                "direct",
                serde_json::json!({
                    "language": "Solidity",
                    "sources": {
                        "src/A.sol": {
                            "content": "import 'lib/Dependency.sol'; contract A is Dependency {}"
                        },
                        "lib/Dependency.sol": { "content": "contract Dependency {}" },
                    },
                    "settings": {},
                }),
            ),
            (
                "relative_then_remap",
                serde_json::json!({
                    "language": "Solidity",
                    "sources": {
                        "src/A.sol": {
                            "content": "import './Dependency.sol'; contract A is Dependency {}"
                        },
                        "lib/Dependency.sol": { "content": "contract Dependency {}" },
                    },
                    "settings": { "remappings": ["src/=lib/"] },
                }),
            ),
            (
                "longest_prefix",
                serde_json::json!({
                    "language": "Solidity",
                    "sources": {
                        "src/A.sol": {
                            "content": "import 'forge-std/Test.sol'; contract A is Dependency {}"
                        },
                        "lib/forge-std/src/Test.sol": { "content": "contract Fallback {}" },
                        "lib/forge-std/special/Test.sol": { "content": "contract Dependency {}" },
                    },
                    "settings": {
                        "remappings": [
                            "forge-std/=lib/forge-std/src/",
                            "forge-std/Test.sol=lib/forge-std/special/Test.sol",
                        ],
                    },
                }),
            ),
            (
                "last_equal_prefix",
                serde_json::json!({
                    "language": "Solidity",
                    "sources": {
                        "src/A.sol": {
                            "content": "import 'forge-std/Test.sol'; contract A is Dependency {}"
                        },
                        "lib/forge-std/first/Test.sol": { "content": "contract Fallback {}" },
                        "lib/forge-std/last/Test.sol": { "content": "contract Dependency {}" },
                    },
                    "settings": {
                        "remappings": [
                            "forge-std/=lib/forge-std/first/",
                            "forge-std/=lib/forge-std/last/",
                        ],
                    },
                }),
            ),
        ];

        for (name, mut input) in cases.into_iter().chain(context_remapping_resolution_cases()) {
            input["settings"]["outputSelection"] = serde_json::json!({ "*": { "*": ["abi"] } });
            let cache_dir = dir.path().join(name);
            let mut solc = solc.clone();
            solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(&cache_dir)));
            let output: serde_json::Value = solc.compile_as(&input).unwrap();
            assert!(
                output["errors"]
                    .as_array()
                    .is_none_or(|errors| errors.iter().all(|error| error["severity"] != "error")),
                "{name}"
            );
            assert!(std::fs::read_dir(cache_dir).unwrap().next().is_some(), "{name}");
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_compiler_output_cache_replays_without_running_solc() {
        let dir = tempfile::tempdir().unwrap();
        let solc_path = dir.path().join("solc");
        let invocations = dir.path().join("invocations");
        std::fs::write(
            &solc_path,
            format!(
                r#"#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "solc, the solidity compiler commandline interface"
    echo "Version: 0.8.35+commit.69074fbd"
else
    touch "{}"
    echo '{{"contracts":{{}}}}'
fi
"#,
                invocations.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&solc_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&solc_path, permissions).unwrap();

        let mut solc = Solc::new(&solc_path).unwrap();
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(
            dir.path().join("cache"),
        )));
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });

        let first: serde_json::Value = solc.async_compile_as(&input).await.unwrap();
        std::fs::remove_file(&invocations).unwrap();
        let second: serde_json::Value = solc.async_compile_as(&input).await.unwrap();

        assert_eq!(first, second);
        assert!(!invocations.exists());
    }

    fn coalescing_compiler(root: &Path) -> Solc {
        let mut solc = Solc::new_with_version(root.join("solc"), Version::new(0, 8, 35));
        solc.set_compiler_output_cache(Arc::new(LocalCompilerOutputCache::new(root.join("cache"))));
        solc
    }

    fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(15);
        while !path.exists() {
            assert!(Instant::now() < deadline, "timed out waiting for {}", path.display());
            thread::sleep(Duration::from_millis(5));
        }
    }

    // The parent test starts copies of this test executable in separate working directories.
    #[test]
    fn compiler_output_cache_process_worker() {
        let Some(root) = env::var_os("FOUNDRY_CACHE_PROCESS_ROOT").map(PathBuf::from) else {
            return;
        };
        let solc = coalescing_compiler(&root);
        #[cfg(feature = "async")]
        if env::var_os("FOUNDRY_CACHE_CANCEL_WAITER").is_some() {
            let input = serde_json::json!({"language": "Solidity", "sources": {}});
            let (key, _) =
                solc.compiler_output_cache_key(&serde_json::to_vec(&input).unwrap()).unwrap();
            let guard =
                solc.compiler_output_cache.as_ref().unwrap().try_acquire(&key).unwrap().unwrap();
            let runtime =
                tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            runtime.block_on(async move {
                let waiter = tokio::spawn(async move {
                    solc.async_compile_as::<_, serde_json::Value>(&input).await
                });
                tokio::time::sleep(Duration::from_millis(100)).await;
                waiter.abort();
                assert!(waiter.await.unwrap_err().is_cancelled());
            });
            drop(runtime);
            drop(guard);
            return;
        }
        if env::var_os("FOUNDRY_CACHE_HOLD_LOCK").is_some() {
            let input =
                serde_json::to_vec(&serde_json::json!({"language": "Solidity", "sources": {}}))
                    .unwrap();
            let (key, _) = solc.compiler_output_cache_key(&input).unwrap();
            let _guard =
                solc.compiler_output_cache.as_ref().unwrap().try_acquire(&key).unwrap().unwrap();
            std::fs::write(root.join(format!("ready-{}", std::process::id())), b"").unwrap();
            loop {
                thread::park();
            }
        }
        std::fs::write(root.join(format!("ready-{}", std::process::id())), b"").unwrap();
        wait_for_file(&root.join("go"));
        let input = serde_json::json!({"language": "Solidity", "sources": {}});
        let output: serde_json::Value = if env::var_os("FOUNDRY_CACHE_PROCESS_ASYNC").is_some() {
            #[cfg(feature = "async")]
            {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(solc.async_compile_as(&input))
                    .unwrap()
            }
            #[cfg(not(feature = "async"))]
            panic!("async test needs the async feature");
        } else {
            solc.compile_as(&input).unwrap()
        };
        assert_eq!(output, serde_json::json!({"contracts": {}}));
    }

    fn coalescing_fixture() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let executable = root.join("solc");
        std::fs::write(
            &executable,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Version: 0.8.35'; exit 0; fi\ncat >/dev/null\necho x >> '{}'\nsleep 1\nprintf '%s\\n' '{{\"contracts\":{{}}}}'\n",
                root.join("invocations").display(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        directory
    }

    fn check_process_coalescing(asynchronous: bool) {
        let directory = coalescing_fixture();
        let root = directory.path();
        let mut children = Vec::new();
        for index in 0..4 {
            let worktree = root.join(format!("worktree-{index}"));
            std::fs::create_dir(&worktree).unwrap();
            let mut command = Command::new(env::current_exe().unwrap());
            command
                .args(["compiler_output_cache_process_worker", "--nocapture"])
                .current_dir(worktree)
                .env("FOUNDRY_CACHE_PROCESS_ROOT", root)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if asynchronous {
                command.env("FOUNDRY_CACHE_PROCESS_ASYNC", "1");
            }
            children.push(command.spawn().unwrap());
        }
        for child in &children {
            wait_for_file(&root.join(format!("ready-{}", child.id())));
        }
        std::fs::write(root.join("go"), b"").unwrap();
        for child in children {
            let output = child.wait_with_output().unwrap();
            assert!(output.status.success(), "{output:?}");
        }
        assert_eq!(std::fs::read_to_string(root.join("invocations")).unwrap(), "x\n");
    }

    #[test]
    fn compiler_output_cache_coalesces_processes() {
        check_process_coalescing(false);
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_compiler_output_cache_coalesces_processes() {
        check_process_coalescing(true);
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_compiler_output_cache_cancelled_waiter_releases_runtime() {
        let directory = coalescing_fixture();
        let mut child = Command::new(env::current_exe().unwrap())
            .arg("compiler_output_cache_process_worker")
            .env("FOUNDRY_CACHE_PROCESS_ROOT", directory.path())
            .env("FOUNDRY_CACHE_CANCEL_WAITER", "1")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("cancelled cache waiter blocked runtime shutdown");
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn compiler_output_cache_recovers_after_owner_process_exits() {
        let directory = coalescing_fixture();
        let root = directory.path();
        let mut child = Command::new(env::current_exe().unwrap())
            .arg("compiler_output_cache_process_worker")
            .env("FOUNDRY_CACHE_PROCESS_ROOT", root)
            .env("FOUNDRY_CACHE_HOLD_LOCK", "1")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        wait_for_file(&root.join(format!("ready-{}", child.id())));
        child.kill().unwrap();
        child.wait().unwrap();

        let solc = coalescing_compiler(root);
        let input = serde_json::json!({"language": "Solidity", "sources": {}});
        let (key, _) =
            solc.compiler_output_cache_key(&serde_json::to_vec(&input).unwrap()).unwrap();
        let key = key.to_string();
        let path = root.join("cache").join(&key[..2]).join(format!("{key}.lock"));
        LocalCompilerOutputCache::new(root.join("cache")).clean();
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
        file.try_lock().unwrap();
        file.unlock().unwrap();
        let output: serde_json::Value = solc.compile_as(&input).unwrap();
        assert_eq!(output, serde_json::json!({"contracts": {}}));
    }

    #[test]
    #[ignore = "requires Solar in SOLAR_PATH or PATH"]
    fn compiler_output_cache_solar_output_equality() {
        let solar = env::var_os("SOLAR_PATH").map(PathBuf::from).unwrap_or_else(|| "solar".into());
        let uncached = Solc::new(solar).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let cache = Arc::new(LocalCompilerOutputCache::new(directory.path()));
        let mut cached = uncached.clone();
        cached.set_compiler_output_cache(cache.clone());
        let input = serde_json::json!({
            "language": "Solidity",
            "sources": {
                "src/A.sol": {"content": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0; import './Math.sol'; contract A { function twice(uint x) external pure returns (uint) { return Math.twice(x); } }"},
                "src/Math.sol": {"content": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0; library Math { function twice(uint x) internal pure returns (uint) { return x * 2; } }"},
            },
            "settings": {
                "optimizer": {"enabled": true, "runs": 200},
                "evmVersion": "cancun",
                "outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode.object", "evm.bytecode.sourceMap", "evm.deployedBytecode.object", "evm.deployedBytecode.sourceMap"]}},
            },
        });
        let expected: serde_json::Value = uncached.compile_as(&input).unwrap();
        assert!(
            expected["errors"]
                .as_array()
                .is_none_or(|errors| errors.iter().all(|error| error["severity"] != "error")),
            "{expected}"
        );
        let contract = &expected["contracts"]["src/A.sol"]["A"];
        for field in [
            "/metadata",
            "/evm/bytecode/object",
            "/evm/bytecode/sourceMap",
            "/evm/deployedBytecode/object",
            "/evm/deployedBytecode/sourceMap",
        ] {
            assert!(
                !contract.pointer(field).and_then(serde_json::Value::as_str).unwrap().is_empty(),
                "{field}"
            );
        }
        let cold: serde_json::Value = cached.compile_as(&input).unwrap();
        let hit: serde_json::Value = cached.compile_as(&input).unwrap();
        assert_eq!(cold, expected);
        assert_eq!(hit, expected);
        assert_eq!(cache.stats().entry_count, 1);
    }

    #[test]
    #[ignore = "requires Solar in SOLAR_PATH or PATH"]
    fn compiler_output_cache_solar_context_remappings() {
        let solar = env::var_os("SOLAR_PATH").map(PathBuf::from).unwrap_or_else(|| "solar".into());
        let solar = Solc::new(solar).unwrap();
        let dir = tempfile::tempdir().unwrap();
        for (name, mut input) in context_remapping_resolution_cases() {
            input["settings"]["outputSelection"] = serde_json::json!({ "*": { "*": ["abi"] } });
            let cache = Arc::new(LocalCompilerOutputCache::new(dir.path().join(name)));
            let mut solar = solar.clone();
            solar.set_compiler_output_cache(cache.clone());
            let output: serde_json::Value = solar.compile_as(&input).unwrap();
            assert!(
                output["errors"]
                    .as_array()
                    .is_none_or(|errors| errors.iter().all(|error| error["severity"] != "error")),
                "{name}: {output}"
            );
            assert_eq!(cache.stats().entry_count, 1, "{name}");
            let hit: serde_json::Value = solar.compile_as(&input).unwrap();
            assert_eq!(hit, output, "{name}");
        }
    }

    #[cfg(feature = "async")]
    #[derive(Debug)]
    struct ControlledExternalCache {
        local_output: Option<Vec<u8>>,
        remote_reads: AtomicUsize,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    #[cfg(feature = "async")]
    impl CompilerOutputCache for ControlledExternalCache {
        fn read(&self, _key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
            self.remote_reads.fetch_add(1, Ordering::Release);
            // Bound a regression that accidentally blocks the runtime thread.
            self.release.lock().unwrap().recv_timeout(Duration::from_secs(5)).ok()?;
            Some(br#"{"source":"remote"}"#.to_vec())
        }

        fn write(&self, _key: &CompilerOutputCacheKey, _output: &[u8]) {
            panic!("a cache hit must not write compiler output");
        }

        fn requires_blocking_io(&self) -> bool {
            true
        }

        fn read_local(&self, _key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
            self.local_output.clone()
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "current_thread")]
    async fn async_compiler_output_cache_external_read_keeps_timer_responsive() {
        let (release, receiver) = std::sync::mpsc::channel();
        let external = Arc::new(ControlledExternalCache {
            local_output: None,
            remote_reads: AtomicUsize::new(0),
            release: Mutex::new(receiver),
        });
        let cache: Arc<dyn CompilerOutputCache> = external.clone();
        let key = input_cache_key(b"external-read-timer");
        let read = tokio::spawn(async move {
            async_cached_compiler_output::<serde_json::Value>(&cache, &key).await
        });

        tokio::time::timeout(Duration::from_secs(5), async {
            while external.remote_reads.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("external cache read did not start");
        tokio::time::sleep(Duration::from_millis(10)).await;
        let completed_before_release = read.is_finished();
        // Release the blocking worker before assertions, including on a failed regression.
        let _ = release.send(());
        assert!(!completed_before_release, "external cache read blocked the runtime timer");
        assert_eq!(read.await.unwrap(), Some(serde_json::json!({ "source": "remote" })));
        assert_eq!(external.remote_reads.load(Ordering::Acquire), 1);
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "current_thread")]
    async fn async_compiler_output_cache_warm_local_hit_skips_external_read() {
        let (_, receiver) = std::sync::mpsc::channel();
        let external = Arc::new(ControlledExternalCache {
            local_output: Some(br#"{"source":"local"}"#.to_vec()),
            remote_reads: AtomicUsize::new(0),
            release: Mutex::new(receiver),
        });
        let cache: Arc<dyn CompilerOutputCache> = external.clone();
        let key = input_cache_key(b"warm-local-read");

        let output = async_cached_compiler_output::<serde_json::Value>(&cache, &key).await;

        assert_eq!(output, Some(serde_json::json!({ "source": "local" })));
        assert_eq!(external.remote_reads.load(Ordering::Acquire), 0);
    }

    #[cfg(feature = "async")]
    #[derive(Debug)]
    struct ControlledExternalWriteCache {
        local: LocalCompilerOutputCache,
        writes_started: AtomicUsize,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    #[cfg(feature = "async")]
    impl CompilerOutputCache for ControlledExternalWriteCache {
        fn read(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
            self.local.read(key)
        }

        fn write(&self, key: &CompilerOutputCacheKey, output: &[u8]) {
            self.writes_started.fetch_add(1, Ordering::Release);
            if self.release.lock().unwrap().recv_timeout(Duration::from_secs(5)).is_ok() {
                self.local.write(key, output);
            }
        }

        fn requires_blocking_io(&self) -> bool {
            true
        }

        fn read_local(&self, key: &CompilerOutputCacheKey) -> Option<Vec<u8>> {
            self.local.read(key)
        }

        fn try_acquire(&self, key: &CompilerOutputCacheKey) -> io::Result<Option<Box<dyn Send>>> {
            self.local.try_acquire(key)
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "current_thread")]
    async fn async_compiler_output_cache_cancelled_write_retains_coordination() {
        let directory = coalescing_fixture();
        let (release, receiver) = std::sync::mpsc::channel();
        let cache = Arc::new(ControlledExternalWriteCache {
            local: LocalCompilerOutputCache::new(directory.path().join("cache")),
            writes_started: AtomicUsize::new(0),
            release: Mutex::new(receiver),
        });
        let mut solc = coalescing_compiler(directory.path());
        solc.set_compiler_output_cache(cache.clone());
        let input = serde_json::json!({ "language": "Solidity", "sources": {} });
        let (key, _) =
            solc.compiler_output_cache_key(&serde_json::to_vec(&input).unwrap()).unwrap();
        let compile =
            tokio::spawn(
                async move { solc.async_compile_as::<_, serde_json::Value>(&input).await },
            );

        tokio::time::timeout(Duration::from_secs(10), async {
            while cache.writes_started.load(Ordering::Acquire) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("external cache write did not start");
        compile.abort();
        assert!(compile.await.unwrap_err().is_cancelled());
        let still_locked = matches!(
            cache.local.try_acquire(&key),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock
        );
        let unpublished = cache.local.read(&key).is_none();
        let _ = release.send(());
        assert!(still_locked, "cancelled compile released coordination before publication");
        assert!(unpublished, "controlled write completed before it was released");

        let _guard = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match cache.local.try_acquire(&key) {
                    Ok(Some(guard)) => break guard,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    _ => panic!("local cache coordination unexpectedly unavailable"),
                }
            }
        })
        .await
        .expect("external cache write did not release coordination");
        let output: serde_json::Value =
            serde_json::from_slice(&cache.local.read(&key).unwrap()).unwrap();
        assert_eq!(output, serde_json::json!({ "contracts": {} }));
        assert_eq!(cache.writes_started.load(Ordering::Acquire), 1);
    }
}

#[cfg(test)]
#[cfg(feature = "svm-solc")]
mod tests {
    use super::*;
    use crate::Artifact;

    #[test]
    fn test_version_parse() {
        let req = SolData::parse_version_req(">=0.6.2 <0.8.21").unwrap();
        let semver_req: VersionReq = ">=0.6.2,<0.8.21".parse().unwrap();
        assert_eq!(req, semver_req);
    }

    fn solc() -> Solc {
        if let Some(solc) = Solc::find_svm_installed_version(&Version::new(0, 8, 18)).unwrap() {
            solc
        } else {
            Solc::blocking_install(&Version::new(0, 8, 18)).unwrap()
        }
    }

    #[test]
    fn solc_version_works() {
        Solc::version(solc().solc).unwrap();
    }

    #[test]
    fn can_parse_version_metadata() {
        let _version = Version::from_str("0.6.6+commit.6c089d02.Linux.gcc").unwrap();
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "multi_thread")]
    async fn async_solc_version_works() {
        Solc::async_version(&solc().solc).await.unwrap();
    }

    #[test]
    fn solc_compile_works() {
        let input = include_str!("../../../../../test-data/in/compiler-in-1.json");
        let input: SolcInput = serde_json::from_str(input).unwrap();
        let out = solc().compile(&input).unwrap();
        let other = solc().compile(&serde_json::json!(input)).unwrap();
        assert_eq!(out, other);
    }

    #[test]
    fn solc_metadata_works() {
        let input = include_str!("../../../../../test-data/in/compiler-in-1.json");
        let mut input: SolcInput = serde_json::from_str(input).unwrap();
        input.settings.push_output_selection("metadata");
        let out = solc().compile(&input).unwrap();
        for (_, c) in out.split().1.contracts_iter() {
            assert!(c.metadata.is_some());
        }
    }

    #[test]
    fn reads_source_units_like_solc() {
        let cwd = std::env::current_dir().unwrap();
        let temp = tempfile::tempdir_in(&cwd).unwrap();
        let relative_base = PathBuf::from(temp.path().file_name().unwrap());
        let effective_base = temp.path().join(&relative_base);
        let effective_include = temp.path().join("include");
        std::fs::create_dir_all(effective_base.join("src")).unwrap();
        std::fs::create_dir_all(effective_include.join("lib")).unwrap();
        std::fs::write(effective_base.join("src/A.sol"), "base").unwrap();
        std::fs::write(effective_include.join("lib/Include.sol"), "include").unwrap();
        let absolute = temp.path().join("Absolute.sol");
        std::fs::write(&absolute, "absolute").unwrap();

        let mut configured_solc = solc();
        configured_solc.base_path = Some(relative_base);
        configured_solc.include_paths.insert(PathBuf::from("include"));

        assert_eq!(configured_solc.read_source_unit("src/A.sol").unwrap(), "base");
        assert_eq!(configured_solc.read_source_unit("lib/Include.sol").unwrap(), "include");
        assert_eq!(configured_solc.read_source_unit("/src//A.sol").unwrap(), "base");
        assert_eq!(configured_solc.read_source_unit("file://src//A.sol").unwrap(), "base");

        let source = absolute.to_str().unwrap();
        assert_eq!(solc().read_source_unit(source).unwrap(), "absolute");
        assert_eq!(solc().read_source_unit(&format!("file://{source}")).unwrap(), "absolute");
        configured_solc.version = Version::new(0, 6, 8);
        assert_eq!(configured_solc.read_source_unit(source).unwrap(), "absolute");

        #[cfg(not(windows))]
        {
            let backslash_dir = append_source_unit(&effective_base, "\\src");
            std::fs::create_dir_all(&backslash_dir).unwrap();
            std::fs::write(backslash_dir.join("A.sol"), "backslash").unwrap();
            configured_solc.version = Version::new(0, 8, 26);
            assert_eq!(configured_solc.read_source_unit("\\src/A.sol").unwrap(), "backslash");
        }
    }

    #[test]
    fn rejects_unreliable_source_unit_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let include = temp.path().join("include");
        std::fs::create_dir_all(base.join("src")).unwrap();
        std::fs::create_dir_all(include.join("src")).unwrap();
        std::fs::write(base.join("src/A.sol"), "base").unwrap();
        std::fs::write(include.join("src/A.sol"), "include").unwrap();

        let mut solc = solc();
        solc.base_path = Some(base);
        solc.include_paths.insert(include);

        let err = solc.read_source_unit("src/A.sol").unwrap_err();
        assert!(err.to_string().contains("source unit `src/A.sol` is ambiguous"));

        solc.extra_args.push("--base-path=other".to_string());
        let err = solc.read_source_unit("src/A.sol").unwrap_err();
        assert!(err.to_string().contains("extra Solc arguments override base or include paths"));
    }

    #[test]
    fn can_compile_with_remapped_links() {
        let input: SolcInput = serde_json::from_str(include_str!(
            "../../../../../test-data/library-remapping-in.json"
        ))
        .unwrap();
        let out = solc().compile(&input).unwrap();
        let (_, mut contracts) = out.split();
        let contract = contracts.remove("LinkTest").unwrap();
        let bytecode = &contract.get_bytecode().unwrap().object;
        assert!(!bytecode.is_unlinked());
    }

    #[test]
    fn can_compile_with_remapped_links_temp_dir() {
        let input: SolcInput = serde_json::from_str(include_str!(
            "../../../../../test-data/library-remapping-in-2.json"
        ))
        .unwrap();
        let out = solc().compile(&input).unwrap();
        let (_, mut contracts) = out.split();
        let contract = contracts.remove("LinkTest").unwrap();
        let bytecode = &contract.get_bytecode().unwrap().object;
        assert!(!bytecode.is_unlinked());
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "multi_thread")]
    async fn async_solc_compile_works() {
        let input = include_str!("../../../../../test-data/in/compiler-in-1.json");
        let input: SolcInput = serde_json::from_str(input).unwrap();
        let out = solc().async_compile(&input).await.unwrap();
        let other = solc().async_compile(&serde_json::json!(input)).await.unwrap();
        assert_eq!(out, other);
    }

    #[cfg(feature = "async")]
    #[tokio::test(flavor = "multi_thread")]
    async fn async_solc_compile_works2() {
        let input = include_str!("../../../../../test-data/in/compiler-in-2.json");
        let input: SolcInput = serde_json::from_str(input).unwrap();
        let out = solc().async_compile(&input).await.unwrap();
        let other = solc().async_compile(&serde_json::json!(input)).await.unwrap();
        assert_eq!(out, other);
        let sync_out = solc().compile(&input).unwrap();
        assert_eq!(out, sync_out);
    }

    #[test]
    fn test_version_req() {
        let versions = ["=0.1.2", "^0.5.6", ">=0.7.1", ">0.8.0"];

        for version in &versions {
            let version_req = SolData::parse_version_req(version).unwrap();
            assert_eq!(version_req, VersionReq::from_str(version).unwrap());
        }

        // Solidity defines version ranges with a space, whereas the semver package
        // requires them to be separated with a comma
        let version_range = ">=0.8.0 <0.9.0";
        let version_req = SolData::parse_version_req(version_range).unwrap();
        assert_eq!(version_req, VersionReq::from_str(">=0.8.0,<0.9.0").unwrap());
    }

    #[test]
    #[cfg(feature = "full")]
    fn test_find_installed_version_path() {
        // This test does not take the lock by default, so we need to manually add it here.
        take_solc_installer_lock!(_lock);
        let version = Version::new(0, 8, 6);
        if svm::installed_versions()
            .map(|versions| !versions.contains(&version))
            .unwrap_or_default()
        {
            Solc::blocking_install(&version).unwrap();
        }
        drop(_lock);
        let res = Solc::find_svm_installed_version(&version).unwrap().unwrap();
        let expected = svm::data_dir().join(version.to_string()).join(format!("solc-{version}"));
        assert_eq!(res.solc, expected);
    }

    #[test]
    #[cfg(feature = "svm-solc")]
    fn can_install_solc_in_tokio_rt() {
        let version = Version::from_str("0.8.6").unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async { Solc::blocking_install(&version) });
        assert!(result.is_ok());
    }

    #[test]
    fn does_not_find_not_installed_version() {
        let ver = Version::new(1, 1, 1);
        let res = Solc::find_svm_installed_version(&ver).unwrap();
        assert!(res.is_none());
    }
}
