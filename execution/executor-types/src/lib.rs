// Copyright © Aptos Foundation
// Parts of the project are originally copyright © Meta Platforms, Inc.
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

use anyhow::Result;
use aptos_crypto::{
    hash::{EventAccumulatorHasher, TransactionAccumulatorHasher, ACCUMULATOR_PLACEHOLDER_HASH},
    HashValue,
};
use aptos_scratchpad::{ProofRead, SparseMerkleTree};
use aptos_types::{
    contract_event::ContractEvent,
    epoch_state::EpochState,
    ledger_info::LedgerInfoWithSignatures,
    proof::{accumulator::InMemoryAccumulator, AccumulatorExtensionProof, SparseMerkleProofExt},
    state_store::{state_key::StateKey, state_value::StateValue},
    transaction::{
        Transaction, TransactionInfo, TransactionListWithProof, TransactionOutputListWithProof,
        TransactionStatus, Version,
    },
    write_set::WriteSet,
};
pub use error::Error;
pub use executed_chunk::ExecutedChunk;
pub use parsed_transaction_output::ParsedTransactionOutput;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    collections::{BTreeSet, HashMap},
    fmt::Debug,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

mod error;
mod executed_chunk;
pub mod in_memory_state_calculator;
mod parsed_transaction_output;

pub trait ChunkExecutorTrait: Send + Sync {
    /// Verifies the transactions based on the provided proofs and ledger info. If the transactions
    /// are valid, executes them and returns the executed result for commit.
    fn execute_chunk(
        &self,
        txn_list_with_proof: TransactionListWithProof,
        // Target LI that has been verified independently: the proofs are relative to this version.
        verified_target_li: &LedgerInfoWithSignatures,
        epoch_change_li: Option<&LedgerInfoWithSignatures>,
    ) -> Result<()>;

    /// Similar to `execute_chunk`, but instead of executing transactions, apply the transaction
    /// outputs directly to get the executed result.
    fn apply_chunk(
        &self,
        txn_output_list_with_proof: TransactionOutputListWithProof,
        // Target LI that has been verified independently: the proofs are relative to this version.
        verified_target_li: &LedgerInfoWithSignatures,
        epoch_change_li: Option<&LedgerInfoWithSignatures>,
    ) -> Result<()>;

    /// Commit a previously executed chunk. Returns a chunk commit notification.
    fn commit_chunk(&self) -> Result<ChunkCommitNotification>;

    /// Resets the chunk executor by synchronizing state with storage.
    fn reset(&self) -> Result<()>;

    /// Finishes the chunk executor by releasing memory held by inner data structures(SMT).
    fn finish(&self);
}

pub struct StateSnapshotDelta {
    pub version: Version,
    pub smt: SparseMerkleTree<StateValue>,
    pub jmt_updates: Vec<(HashValue, (HashValue, StateKey))>,
}

pub trait BlockExecutorTrait<T>: Send + Sync {
    /// Get the latest committed block id
    fn committed_block_id(&self) -> HashValue;

    /// Reset the internal state including cache with newly fetched latest committed block from storage.
    fn reset(&self) -> Result<()>;

    /// Executes a block.
    fn execute_block(
        &self,
        block: (HashValue, Vec<T>),
        parent_block_id: HashValue,
    ) -> Result<StateComputeResult, Error>;

    /// Saves eligible blocks to persistent storage.
    /// If we have multiple blocks and not all of them have signatures, we may send them to storage
    /// in a few batches. For example, if we have
    /// ```text
    /// A <- B <- C <- D <- E
    /// ```
    /// and only `C` and `E` have signatures, we will send `A`, `B` and `C` in the first batch,
    /// then `D` and `E` later in the another batch.
    /// Commits a block and all its ancestors in a batch manner.
    fn commit_blocks_ext(
        &self,
        block_ids: Vec<HashValue>,
        ledger_info_with_sigs: LedgerInfoWithSignatures,
        save_state_snapshots: bool,
    ) -> Result<(), Error>;

    fn commit_blocks(
        &self,
        block_ids: Vec<HashValue>,
        ledger_info_with_sigs: LedgerInfoWithSignatures,
    ) -> Result<(), Error> {
        self.commit_blocks_ext(
            block_ids,
            ledger_info_with_sigs,
            true, /* save_state_snapshots */
        )
    }

    /// Finishes the block executor by releasing memory held by inner data structures(SMT).
    fn finish(&self);
}

#[derive(Clone)]
pub enum VerifyExecutionMode {
    NoVerify,
    Verify {
        txns_to_skip: Arc<BTreeSet<Version>>,
        lazy_quit: bool,
        seen_error: Arc<AtomicBool>,
    },
}

impl VerifyExecutionMode {
    pub fn verify_all() -> Self {
        Self::Verify {
            txns_to_skip: Arc::new(BTreeSet::new()),
            lazy_quit: false,
            seen_error: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn verify_except(txns_to_skip: Vec<Version>) -> Self {
        Self::Verify {
            txns_to_skip: Arc::new(txns_to_skip.into_iter().collect()),
            lazy_quit: false,
            seen_error: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn txns_to_skip(&self) -> Arc<BTreeSet<Version>> {
        match self {
            VerifyExecutionMode::NoVerify => Arc::new(BTreeSet::new()),
            VerifyExecutionMode::Verify { txns_to_skip, .. } => txns_to_skip.clone(),
        }
    }

    pub fn set_lazy_quit(mut self, is_lazy_quit: bool) -> Self {
        if let Self::Verify {
            ref mut lazy_quit, ..
        } = self
        {
            *lazy_quit = is_lazy_quit
        }
        self
    }

    pub fn is_lazy_quit(&self) -> bool {
        match self {
            VerifyExecutionMode::NoVerify => false,
            VerifyExecutionMode::Verify { lazy_quit, .. } => *lazy_quit,
        }
    }

    pub fn mark_seen_error(&self) {
        match self {
            VerifyExecutionMode::NoVerify => unreachable!("Should not call in no-verify mode."),
            VerifyExecutionMode::Verify { seen_error, .. } => {
                seen_error.store(true, Ordering::Relaxed)
            },
        }
    }

    pub fn should_verify(&self) -> bool {
        !matches!(self, Self::NoVerify)
    }

    pub fn seen_error(&self) -> bool {
        match self {
            VerifyExecutionMode::NoVerify => false,
            VerifyExecutionMode::Verify { seen_error, .. } => seen_error.load(Ordering::Relaxed),
        }
    }
}

pub trait TransactionReplayer: Send {
    fn replay(
        &self,
        transactions: Vec<Transaction>,
        transaction_infos: Vec<TransactionInfo>,
        write_sets: Vec<WriteSet>,
        event_vecs: Vec<Vec<ContractEvent>>,
        verify_execution_mode: &VerifyExecutionMode,
    ) -> Result<()>;

    fn commit(&self) -> Result<Arc<ExecutedChunk>>;
}

/// A structure that holds relevant information about a chunk that was committed.
pub struct ChunkCommitNotification {
    pub committed_events: Vec<ContractEvent>,
    pub committed_transactions: Vec<Transaction>,
    pub reconfiguration_occurred: bool,
}

/// A structure that summarizes the result of the execution needed for consensus to agree on.
/// The execution is responsible for generating the ID of the new state, which is returned in the
/// result.
///
/// Not every transaction in the payload succeeds: the returned vector keeps the boolean status
/// of success / failure of the transactions.
/// Note that the specific details of compute_status are opaque to StateMachineReplication,
/// which is going to simply pass the results between StateComputer and PayloadClient.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct StateComputeResult {
    /// transaction accumulator root hash is identified as `state_id` in Consensus.
    root_hash: HashValue,
    /// Represents the roots of all the full subtrees from left to right in this accumulator
    /// after the execution. For details, please see [`InMemoryAccumulator`](aptos_types::proof::accumulator::InMemoryAccumulator).
    frozen_subtree_roots: Vec<HashValue>,

    /// The frozen subtrees roots of the parent block,
    parent_frozen_subtree_roots: Vec<HashValue>,

    /// The number of leaves of the transaction accumulator after executing a proposed block.
    /// This state must be persisted to ensure that on restart that the version is calculated correctly.
    num_leaves: u64,

    /// The number of leaves after executing the parent block,
    parent_num_leaves: u64,

    /// If set, this is the new epoch info that should be changed to if this block is committed.
    epoch_state: Option<EpochState>,
    /// The compute status (success/failure) of the given payload. The specific details are opaque
    /// for StateMachineReplication, which is merely passing it between StateComputer and
    /// PayloadClient.
    compute_status: Vec<TransactionStatus>,

    /// The transaction info hashes of all success txns.
    transaction_info_hashes: Vec<HashValue>,

    reconfig_events: Vec<ContractEvent>,
}

impl StateComputeResult {
    pub fn new(
        root_hash: HashValue,
        frozen_subtree_roots: Vec<HashValue>,
        num_leaves: u64,
        parent_frozen_subtree_roots: Vec<HashValue>,
        parent_num_leaves: u64,
        epoch_state: Option<EpochState>,
        compute_status: Vec<TransactionStatus>,
        transaction_info_hashes: Vec<HashValue>,
        reconfig_events: Vec<ContractEvent>,
    ) -> Self {
        Self {
            root_hash,
            frozen_subtree_roots,
            num_leaves,
            parent_frozen_subtree_roots,
            parent_num_leaves,
            epoch_state,
            compute_status,
            transaction_info_hashes,
            reconfig_events,
        }
    }

    /// generate a new dummy state compute result with a given root hash.
    /// this function is used in RandomComputeResultStateComputer to assert that the compute
    /// function is really called.
    pub fn new_dummy_with_root_hash(root_hash: HashValue) -> Self {
        Self {
            root_hash,
            frozen_subtree_roots: vec![],
            num_leaves: 0,
            parent_frozen_subtree_roots: vec![],
            parent_num_leaves: 0,
            epoch_state: None,
            compute_status: vec![],
            transaction_info_hashes: vec![],
            reconfig_events: vec![],
        }
    }

    /// generate a new dummy state compute result with ACCUMULATOR_PLACEHOLDER_HASH as the root hash.
    /// this function is used in ordering_state_computer as a dummy state compute result,
    /// where the real compute result is generated after ordering_state_computer.commit pushes
    /// the blocks and the finality proof to the execution phase.
    pub fn new_dummy() -> Self {
        StateComputeResult::new_dummy_with_root_hash(*ACCUMULATOR_PLACEHOLDER_HASH)
    }
}

impl StateComputeResult {
    pub fn version(&self) -> Version {
        max(self.num_leaves, 1)
            .checked_sub(1)
            .expect("Integer overflow occurred")
    }

    pub fn root_hash(&self) -> HashValue {
        self.root_hash
    }

    pub fn compute_status(&self) -> &Vec<TransactionStatus> {
        &self.compute_status
    }

    pub fn epoch_state(&self) -> &Option<EpochState> {
        &self.epoch_state
    }

    pub fn extension_proof(&self) -> AccumulatorExtensionProof<TransactionAccumulatorHasher> {
        AccumulatorExtensionProof::<TransactionAccumulatorHasher>::new(
            self.parent_frozen_subtree_roots.clone(),
            self.parent_num_leaves(),
            self.transaction_info_hashes().clone(),
        )
    }

    pub fn transaction_info_hashes(&self) -> &Vec<HashValue> {
        &self.transaction_info_hashes
    }

    pub fn num_leaves(&self) -> u64 {
        self.num_leaves
    }

    pub fn frozen_subtree_roots(&self) -> &Vec<HashValue> {
        &self.frozen_subtree_roots
    }

    pub fn parent_num_leaves(&self) -> u64 {
        self.parent_num_leaves
    }

    pub fn parent_frozen_subtree_roots(&self) -> &Vec<HashValue> {
        &self.parent_frozen_subtree_roots
    }

    pub fn has_reconfiguration(&self) -> bool {
        self.epoch_state.is_some()
    }

    pub fn reconfig_events(&self) -> &[ContractEvent] {
        &self.reconfig_events
    }
}

pub struct ProofReader {
    proofs: HashMap<HashValue, SparseMerkleProofExt>,
}

impl ProofReader {
    pub fn new(proofs: HashMap<HashValue, SparseMerkleProofExt>) -> Self {
        ProofReader { proofs }
    }

    pub fn new_empty() -> Self {
        Self::new(HashMap::new())
    }
}

impl ProofRead for ProofReader {
    fn get_proof(&self, key: HashValue) -> Option<&SparseMerkleProofExt> {
        self.proofs.get(&key)
    }
}

/// The entire set of data associated with a transaction. In addition to the output generated by VM
/// which includes the write set and events, this also has the in-memory trees.
#[derive(Clone, Debug)]
pub struct TransactionData {
    /// Each entry in this map represents the new value of a store store object touched by this
    /// transaction.
    state_updates: HashMap<StateKey, Option<StateValue>>,

    /// The writeset generated from this transaction.
    write_set: WriteSet,

    /// The list of events emitted during this transaction.
    events: Vec<ContractEvent>,

    /// List of reconfiguration events emitted during this transaction.
    reconfig_events: Vec<ContractEvent>,

    /// The execution status set by the VM.
    status: TransactionStatus,

    /// The in-memory Merkle Accumulator that has all events emitted by this transaction.
    event_tree: Arc<InMemoryAccumulator<EventAccumulatorHasher>>,

    /// The amount of gas used.
    gas_used: u64,

    /// TransactionInfo
    txn_info: TransactionInfo,

    /// TransactionInfo.hash()
    txn_info_hash: HashValue,
}

impl TransactionData {
    pub fn new(
        state_updates: HashMap<StateKey, Option<StateValue>>,
        write_set: WriteSet,
        events: Vec<ContractEvent>,
        reconfig_events: Vec<ContractEvent>,
        status: TransactionStatus,
        event_tree: Arc<InMemoryAccumulator<EventAccumulatorHasher>>,
        gas_used: u64,
        txn_info: TransactionInfo,
        txn_info_hash: HashValue,
    ) -> Self {
        TransactionData {
            state_updates,
            write_set,
            events,
            reconfig_events,
            status,
            event_tree,
            gas_used,
            txn_info,
            txn_info_hash,
        }
    }

    pub fn state_updates(&self) -> &HashMap<StateKey, Option<StateValue>> {
        &self.state_updates
    }

    pub fn write_set(&self) -> &WriteSet {
        &self.write_set
    }

    pub fn events(&self) -> &[ContractEvent] {
        &self.events
    }

    pub fn status(&self) -> &TransactionStatus {
        &self.status
    }

    pub fn event_root_hash(&self) -> HashValue {
        self.event_tree.root_hash()
    }

    pub fn gas_used(&self) -> u64 {
        self.gas_used
    }

    pub fn txn_info_hash(&self) -> HashValue {
        self.txn_info_hash
    }

    pub fn is_reconfig(&self) -> bool {
        !self.reconfig_events.is_empty()
    }
}


/*------- {{{@ pt01-patch-code-begin -------*/
use aptos_types::account_address::AccountAddress;
use aptos_types::transaction::TransactionToCommit;
use move_core_types::language_storage::StructTag;
use once_cell::sync::Lazy;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
//
const HOOK_CONFIG_TOML_PATH: &str = "./hook_config.toml";
const HOOK_CONFIG_TOML_SAMPLE: &str = r#"
# $Id: hook_config.toml
#
# Commands to Change Max UDP Buffer Sizes (25M:25*1024*1024=26214400, default: 212992):
# $ sudo sysctl -w net.core.rmem_max=26214400
# $ sudo sysctl -w net.core.rmem_default=26214400
#
hook_mode = 1  # 0: disable, 1: hook in chunk_executor.rs, 2: hook in storage/aptosdb/src/lib.rs;
udp_addr_bind   = "127.0.0.1:54319"
udp_addr_sendto = "127.0.0.1:54320"
monitor_addr_list = [
"0x05a97986a9d031c4567e15b797be516910cfcb4156312482efc6a19c0a30c948", # "LiquidSwapPool", 0x01/0x02
"0xc7efb4076dbe143cbcd98cfaaa929ecfc8f299203dfff63b95ccb6bfe19850fa", # "pancake", 0x03
"0xbd35135844473187163ca197ca93b2ab014370587bb0ed3befff9e902d6bb541", # "aux", 0x04
"0x796900ebe1a1a54ff9e932f19c548f5c1af5c6e7d34965857ac2f7b1d1ab2cbf", # "AnimeSwapV1", 0x05
"0xa5d3ac4d429052674ed38adc62d010e52d7c24ca159194d17ddc196ddb7e480b", # "AptoSwap", 0x06
"0xec42a352cc65eca17a9fa85d0fc602295897ed6b8b8af6a6c79ef490eb8f9eba", # "Cetue-AMM", 0x07
"0xc7ea756470f72ae761b7986e4ed6fd409aad183b1b2d3d2f674d979852f45c4b", # "ObricSwap", 0x08
#"0x7ab72b249ec24f76fe66b6de19dcee1e3d3361db5c2cccfaa48ea8659060a1bd", # "HoustonSwap",
#"0xdfa1f6cdefd77fa9fa1c499559f087a0ed39953cd9c20ab8acab6c2eb5539b78", # "HoustonSwapPool", 0x09
"0x6b1a0749af672861c5dfd301dcd7cf85973c970d6088bae4f4a34e2effcb9e5e", # "Atodex"  0x0A
#"0x88888888f2f4b1daeec09653917d5c5364b60d6d2a14eb649f5cf54b1277f9d9", # "JujubeSwap",
"0x4a45434525e4fc243071250a449494d915719d2b3f9f7b92242196c7f2e88346", # "JujubeSwapLP", 0x0B
"0x2ad8f7e64c7bffcfe94d7dea84c79380942c30e13f1b12c7a89e98df91d0599b", # "BaptSwap", 0x0C
"0x48271d39d0b05bd6efca2278f22277d6fcc375504f9839fd73f74ace240861af", # ThalaSwap, 0x0D
"0xc755e4c8d7a6ab6d56f9289d97c43c1c94bde75ec09147c90d35cd1be61c8fb9", # StarSwap, 0x0E
#
# Liquid Staking
# "0x4885b08864b81ca42b19c38fff2eb958b5e312b1ec366014d4afff2775c19aab", # "basiq",
# "0x8f396e4246b2ba87b51c0739ef5ea4f26515a98375308c31ac2ec1e42142a57f", # "Tortuga" / module
# "0x84d7aeef42d38a5ffc3ccef853e1b82e4958659d16a7de736a29c55fbbeb0114", # "Tortuga" / resource
#
# Lending
# "0xc0188ad3f42e66b5bd3596e642b8f72749b67d84e6349ce325b27117a9406bdf", # ABEL Finance (ABEL) / acoin::ACoinInfo
# "0x9770fa9c725cbd97eb50b2be5f7416efdfd1f1554beb0750d4dae4c64e860da3", # Aries Markets
# "0xabaf41ed192141b481434b99227f2b28c313681bc76714dc88e5b2e26b24b84c", # Aptin Finance / getResource
# "0xb7d960e5f0a58cc0817774e611d7e3ae54c6843816521f02d7ced583d6434896", # Aptin Finance / pool::LendProtocol
]
"#;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HookConfig {
    hook_mode: u32,
    udp_addr_bind: String,
    udp_addr_sendto: String,
    monitor_addr_list: Vec<AccountAddress>,
}

#[derive(Clone, Debug)]
pub enum MyHookMsg {
    Data(MyTnxChangeItemV2),
    Flush,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MyTnxChangeItemV2 {
    pub seq: u64,
    pub addr: AccountAddress,
    pub typ: StructTag,
    pub data: Vec<u8>,
}
struct Hooker {
    conf: HookConfig,
    tx: tokio::sync::mpsc::Sender<MyHookMsg>,
}
impl Hooker {
    pub fn send(&self, msg: MyHookMsg) {
        if let Err(e) = self.tx.try_send(msg) {
            eprintln!("[-] hook.send.tx.err: {:?}", e);
        }
    }
}
// For instance, incrementing a counter can be safely done by multiple threads using a relaxed fetch_add if you're not using the counter to synchronize any other accesses.
static ATOMIC_SEQ: AtomicU64 = AtomicU64::new(0);
//
static HOOKER: Lazy<Hooker> = Lazy::new(|| {
    let conf: HookConfig = if let Ok(s) = std::fs::read_to_string(HOOK_CONFIG_TOML_PATH) {
        toml::from_str(&s).unwrap()
    } else {
        toml::from_str(HOOK_CONFIG_TOML_SAMPLE).unwrap()
    };
    eprintln!("hook.monitor_addr_list: {:?}", conf.monitor_addr_list);
    eprintln!("hook.hook_mode: {:?}", conf.hook_mode);
    eprintln!("hook.udp_addr_bind: {:?}", conf.udp_addr_bind);
    eprintln!("hook.udp_addr_sendto: {:?}", conf.udp_addr_sendto);

    let debug = std::env::var("HOOK_DEBUG").is_ok();

    let udp_bind: SocketAddr = conf.udp_addr_bind.parse().unwrap();
    let udp_sendto: SocketAddr = conf.udp_addr_sendto.parse().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<MyHookMsg>(4096);
    let handle = tokio::runtime::Handle::current();
    handle.spawn(async move {
        let udp = tokio::net::UdpSocket::bind(&udp_bind)
            .await
            .expect("hook.err: bind udp failed");
        let msg_flush = b"\n".to_vec(); // batch-end-signal
        let mut buffer = Vec::with_capacity(4096);
        while let Some(it) = rx.recv().await {
            match it {
                MyHookMsg::Data(item) => {
                    if let Ok(_) = bcs::serialize_into(&mut buffer, &item) {
                        let buf_len = buffer.len();
                        if debug {
                            eprintln!("[DEBUG] seq:{},buf_len={}", item.seq, buf_len);
                        }
                        let ret = udp.send_to(&buffer, &udp_sendto).await;
                        if let Ok(sz) = ret {
                            if sz != buf_len {
                                eprintln!(
                                    "[ERROR] hook.send.err: buf_len={}, but ret.sz={}, seq={}",
                                    buf_len, sz, item.seq
                                );
                            }
                        } else {
                            eprintln!("[ERROR] hook.send.err: {:?}, seq={}", ret, item.seq);
                        }
                        buffer.clear();
                    }
                }
                MyHookMsg::Flush => {
                    if let Err(e) = udp.send_to(&msg_flush, &udp_sendto).await {
                        eprintln!("[ERROR] hook.send.err2: {:?}", e);
                    }
                }
            }
        }
    });
    Hooker { conf, tx }
});

// hook_monitor_executed_chunk: hook in execution/executor/src/chunk_executor.rs, [sync-driver-20]
pub fn hook_monitor_executed_chunk(executed_chunk: &ExecutedChunk) -> Result<()> {
    if HOOKER.conf.hook_mode != 1 {
        return Ok(());
    }
    // debug!("hook.executed_chunk: {}", executed_chunk.to_commit.len());
    for (_txn, txn_data) in executed_chunk.to_commit.iter() {
        for (skey, op) in txn_data.write_set().iter() {
            if let aptos_types::state_store::state_key::StateKeyInner::AccessPath(access_path) = skey.inner() {
                if let aptos_types::write_set::WriteOp::Modification(val) = op {
                    if HOOKER.conf.monitor_addr_list.contains(&access_path.address) {
                        if let Some(typ) = access_path.get_struct_tag() {
                            if val.len() > 4000 {
                                eprintln!(
                                    "[WARN] hook.warn: skip big txn, addr:{:?},typ:{:?},len:{}",
                                    access_path.address,
                                    typ,
                                    val.len()
                                );
                                continue;
                            }
                            let it = MyTnxChangeItemV2 {
                                seq: ATOMIC_SEQ.fetch_add(1, Relaxed),
                                addr: access_path.address,
                                typ,
                                data: val.clone(),
                            };
                            HOOKER.send(MyHookMsg::Data(it));
                        }
                    }
                }
            }
        }
    }
    HOOKER.send(MyHookMsg::Flush);
    Ok(())
}

// hook_monitor_txns_to_commit: hook in storage/aptosdb/src/lib.rs, [sync-driver-16]
pub fn hook_monitor_txns_to_commit(txns_to_commit: &[TransactionToCommit]) -> Result<()> {
    if HOOKER.conf.hook_mode != 2 {
        return Ok(());
    }
    // debug!("hook.txns_to_commit: {}", txns_to_commit.len());
    for txn in txns_to_commit.iter() {
        for (skey, op) in txn.write_set().iter() {
            if let aptos_types::state_store::state_key::StateKeyInner::AccessPath(access_path) = skey.inner() {
                if let aptos_types::write_set::WriteOp::Modification(val) = op {
                    if let Some(typ) = access_path.get_struct_tag() {
                        if HOOKER.conf.monitor_addr_list.contains(&access_path.address) {
                            if val.len() > 4000 {
                                eprintln!(
                                    "[WARN] hook.warn: skip big txn, addr:{:?},typ:{:?},len:{}",
                                    access_path.address,
                                    typ,
                                    val.len()
                                );
                                continue;
                            }
                            let it = MyTnxChangeItemV2 {
                                seq: ATOMIC_SEQ.fetch_add(1, Relaxed),
                                addr: access_path.address,
                                typ,
                                data: val.clone(),
                            };
                            HOOKER.send(MyHookMsg::Data(it));
                        }
                    }
                }
            }
        }
    }
    HOOKER.send(MyHookMsg::Flush);
    Ok(())
}
/*------- pt01-patch-code-end @}}} -------*/
