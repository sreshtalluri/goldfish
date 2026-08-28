"""Deterministic ledger environment.

Design constraints (see PRD section 4):
  1. Long range dependency is forced: identifiers minted early are required late.
  2. Verification is programmatic: full state is inspectable, no LLM judge.
  3. Cheap: pure Python, no I/O, so a run costs only model tokens.

The environment is a small accounts payable workload. The agent intakes vendors
(which mints an account code it will need much later), posts transactions, and
files a closing report. Tool failures are deterministic and are the source of
the negative-knowledge probe class.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _code_for(vendor: str, seed: int) -> str:
    h = hashlib.sha256(f"{vendor}:{seed}".encode()).hexdigest()
    return f"AC-{h[:4].upper()}-{int(h[4:8], 16) % 9000 + 1000}"


@dataclass
class ToolResult:
    ok: bool
    payload: Any
    error: str | None = None

    def render(self) -> str:
        if self.ok:
            return json.dumps(self.payload, sort_keys=True)
        return f"ERROR: {self.error}"


@dataclass
class Account:
    code: str
    vendor: str
    balance: float = 0.0
    frozen: bool = False


@dataclass
class LedgerEnv:
    """A deterministic accounts-payable workload.

    Everything is derived from `seed`, so two runs with the same seed present
    the agent with an identical world. Only the context strategy varies.
    """

    seed: int = 0
    n_vendors: int = 8

    accounts: dict[str, Account] = field(default_factory=dict)
    vendor_to_code: dict[str, str] = field(default_factory=dict)
    pending: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    report_filed: dict[str, Any] | None = None

    # Deterministic behavioural quirks the agent must learn and then remember.
    FAILING_METHOD = "wire"
    WORKING_METHOD = "ach"

    def __post_init__(self) -> None:
        vendors = [f"vendor-{i:02d}" for i in range(self.n_vendors)]
        # Two vendors are frozen. Posting to them is the hard constraint.
        self.frozen_vendors = {vendors[2], vendors[5]}
        self.pending = [
            {
                "id": f"TX-{i:03d}",
                "vendor": vendors[i % self.n_vendors],
                "amount": round(100.0 + (i * 37 + self.seed * 13) % 900 + 0.25, 2),
            }
            for i in range(self.n_vendors * 2)
        ]

    # ------------------------------------------------------------------ tools

    def tool_schema(self) -> list[dict[str, Any]]:
        return [
            {"name": "list_pending", "args": [], "doc": "List unprocessed transactions."},
            {"name": "intake_vendor", "args": ["vendor"], "doc": "Register a vendor. Returns its permanent account code."},
            {"name": "get_account", "args": ["code"], "doc": "Look up an account by code."},
            {"name": "post_transaction", "args": ["code", "tx_id", "amount", "method"], "doc": "Post a transaction to an account."},
            {"name": "policy_lookup", "args": [], "doc": "Return current posting policy, including frozen vendors."},
            {"name": "write_note", "args": ["key", "value"], "doc": "Write to the durable scratchpad."},
            {"name": "read_note", "args": ["key"], "doc": "Read from the durable scratchpad."},
            {"name": "file_report", "args": ["total_posted", "accounts_touched"], "doc": "File the closing report and end the episode."},
        ]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            result = ToolResult(False, None, f"unknown tool {name!r}")
        else:
            try:
                result = fn(**args)
            except TypeError as exc:
                result = ToolResult(False, None, f"bad arguments for {name}: {exc}")
        self.call_log.append({"tool": name, "args": args, "ok": result.ok, "payload": result.payload, "error": result.error})
        return result

    def _t_list_pending(self) -> ToolResult:
        return ToolResult(True, [tx for tx in self.pending if not tx.get("posted")])

    def _t_policy_lookup(self) -> ToolResult:
        return ToolResult(True, {"frozen_vendors": sorted(self.frozen_vendors), "settlement_currency": "USD"})

    def _t_intake_vendor(self, vendor: str) -> ToolResult:
        if vendor in self.vendor_to_code:
            return ToolResult(True, {"vendor": vendor, "code": self.vendor_to_code[vendor], "new": False})
        code = _code_for(vendor, self.seed)
        self.accounts[code] = Account(code=code, vendor=vendor, frozen=vendor in self.frozen_vendors)
        self.vendor_to_code[vendor] = code
        return ToolResult(True, {"vendor": vendor, "code": code, "new": True})

    def _t_get_account(self, code: str) -> ToolResult:
        acct = self.accounts.get(code)
        if acct is None:
            return ToolResult(False, None, f"no account {code!r}. Account codes are minted by intake_vendor.")
        return ToolResult(True, {"code": acct.code, "vendor": acct.vendor, "balance": acct.balance, "frozen": acct.frozen})

    def _t_post_transaction(self, code: str, tx_id: str, amount: float, method: str) -> ToolResult:
        acct = self.accounts.get(code)
        if acct is None:
            return ToolResult(False, None, f"no account {code!r}")
        if method == self.FAILING_METHOD:
            # Deterministic, permanent, and explained once. This is the
            # negative-knowledge probe: does the agent retry it after compaction?
            return ToolResult(False, None, "settlement rail 'wire' is decommissioned for this ledger. Use method 'ach'.")
        if method != self.WORKING_METHOD:
            return ToolResult(False, None, f"unknown method {method!r}")
        tx = next((t for t in self.pending if t["id"] == tx_id), None)
        if tx is None:
            return ToolResult(False, None, f"no pending transaction {tx_id!r}")
        # Note: a frozen account still accepts the post. The environment does not
        # enforce the constraint, because a constraint the environment enforces
        # measures nothing about the agent's memory.
        acct.balance = round(acct.balance + amount, 2)
        tx["posted"] = True
        return ToolResult(True, {"tx_id": tx_id, "code": code, "new_balance": acct.balance})

    def _t_write_note(self, key: str, value: str) -> ToolResult:
        self.notes[key] = value
        return ToolResult(True, {"key": key, "bytes": len(value)})

    def _t_read_note(self, key: str) -> ToolResult:
        if key not in self.notes:
            return ToolResult(False, None, f"no note {key!r}")
        return ToolResult(True, {"key": key, "value": self.notes[key]})

    def _t_file_report(self, total_posted: float, accounts_touched: int) -> ToolResult:
        self.report_filed = {"total_posted": total_posted, "accounts_touched": accounts_touched}
        return ToolResult(True, {"filed": True})

    # ------------------------------------------------------------- inspection

    def truth(self) -> dict[str, Any]:
        """Ground truth for behavioural checkers. Never shown to the agent."""
        posted = [t for t in self.pending if t.get("posted")]
        frozen_codes = {c for c, a in self.accounts.items() if a.frozen}
        violations = [
            c for c in self.call_log
            if c["tool"] == "post_transaction" and c["ok"] and c["args"].get("code") in frozen_codes
        ]
        wire_retries = [
            c for c in self.call_log
            if c["tool"] == "post_transaction" and c["args"].get("method") == self.FAILING_METHOD
        ]
        return {
            "total_posted": round(sum(t["amount"] for t in posted), 2),
            "n_posted": len(posted),
            "n_pending_total": len(self.pending),
            "accounts_touched": len({c.code for c in self.accounts.values() if c.balance}),
            "constraint_violations": len(violations),
            "wire_attempts": len(wire_retries),
            "wire_retries_after_first": max(0, len(wire_retries) - 1),
            "report_filed": self.report_filed,
            "vendor_to_code": dict(self.vendor_to_code),
        }
