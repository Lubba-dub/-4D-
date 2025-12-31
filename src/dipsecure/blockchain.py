import json
from dataclasses import dataclass
from typing import Optional
from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider
from eth_account import Account
from solcx import install_solc, set_solc_version, compile_source


SOLIDITY_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ImageRegistry {
    struct Record {
        bytes32 sha256Hash;
        uint64 timestamp;
        string patientId;
        string note;
        bytes32 pHash;
    }
    mapping(bytes32 => Record) public records;
    event Registered(bytes32 indexed contentHash, bytes32 indexed pHash, string patientId);

    function register(bytes32 contentHash, bytes32 pHash, string calldata patientId, string calldata note) public {
        require(records[contentHash].timestamp == 0, "exists");
        records[contentHash] = Record(contentHash, uint64(block.timestamp), patientId, note, pHash);
        emit Registered(contentHash, pHash, patientId);
    }

    function exists(bytes32 contentHash) public view returns (bool) {
        return records[contentHash].timestamp != 0;
    }
}
"""


@dataclass
class ChainConfig:
    rpc_url: Optional[str] = None  # If None, use EthereumTester
    private_key: Optional[str] = None


class ImageChain:
    def __init__(self, cfg: ChainConfig):
        if cfg.rpc_url:
            self.web3 = Web3(Web3.HTTPProvider(cfg.rpc_url))
            if not self.web3.is_connected():
                raise RuntimeError("Cannot connect to RPC")
        else:
            self.web3 = Web3(EthereumTesterProvider())

        # Compile contract
        install_solc("0.8.20")
        set_solc_version("0.8.20")
        compiled = compile_source(SOLIDITY_SOURCE, output_values=["abi", "bin"])
        _, contract_interface = compiled.popitem()
        self.abi = contract_interface["abi"]
        self.bytecode = contract_interface["bin"]

        # Account
        if cfg.private_key:
            self.account = Account.from_key(cfg.private_key)
        else:
            # Use pre-funded account from EthereumTester
            accounts = self.web3.eth.accounts
            if not accounts:
                raise RuntimeError("No tester accounts available")
            self.account = self.web3.eth.account.from_key(
                # Transform address to local account for signing via default account
                # Note: With EthereumTesterProvider, setting default_account is sufficient
                "0x0000000000000000000000000000000000000000000000000000000000000001"
            )
            self.web3.eth.default_account = accounts[0]

        # In tester mode, transactions use default_account; in RPC mode use provided account
        if cfg.rpc_url:
            self.web3.eth.default_account = self.account.address
        self.contract = None

    def deploy(self):
        Contract = self.web3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx_hash = Contract.constructor().transact()
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        self.contract = self.web3.eth.contract(address=receipt.contractAddress, abi=self.abi)
        return receipt.contractAddress

    def register(self, sha256_hash_hex: str, phash_int: int, patient_id: str, note: str):
        if self.contract is None:
            raise RuntimeError("Contract not deployed")
        content_hash_bytes32 = self.web3.to_bytes(hexstr=sha256_hash_hex)
        p_hash_bytes32 = (phash_int).to_bytes(32, byteorder="big", signed=False)
        tx = self.contract.functions.register(content_hash_bytes32, p_hash_bytes32, patient_id, note).transact()
        receipt = self.web3.eth.wait_for_transaction_receipt(tx)
        return receipt.transactionHash.hex()

    def exists(self, sha256_hash_hex: str) -> bool:
        content_hash_bytes32 = self.web3.to_bytes(hexstr=sha256_hash_hex)
        return self.contract.functions.exists(content_hash_bytes32).call()

