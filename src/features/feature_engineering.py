"""
Feature extractor for webshell detection.
Extracts PHP/ASP/JSP-specific risk signals from source files.
These signals are used by the API for explainability alongside
the deep-learning model prediction.
"""

import math
import os
import re
from dataclasses import asdict, dataclass

# ── Dangerous PHP/ASP functions that commonly appear in webshells ──────────
DANGEROUS_FUNCTIONS = [
    "eval",
    "base64_decode",
    "base64_encode",
    "gzinflate",
    "gzuncompress",
    "gzdecode",
    "str_rot13",
    "preg_replace",
    "assert",
    "create_function",
    "system",
    "exec",
    "passthru",
    "shell_exec",
    "popen",
    "proc_open",
    "pcntl_exec",
    "call_user_func",
    "call_user_func_array",
    "$_GET",
    "$_POST",
    "$_REQUEST",
    "$_COOKIE",
    "$_SERVER",
    "file_get_contents",
    "file_put_contents",
    "fwrite",
    "fopen",
    "curl_exec",
    "move_uploaded_file",
    "phpinfo",
    "chr(",
    "ord(",
    "hex2bin",
    "pack(",
    "unpack(",
    # ASP/ASPX patterns
    "Response.Write",
    "Server.Execute",
    "CreateObject",
    "WScript.Shell",
    "ADODB.Stream",
    # JSP patterns
    "Runtime.getRuntime",
    "ProcessBuilder",
]

# ── Obfuscation patterns ─────────────────────────────────────────────────
OBFUSCATION_PATTERNS = [
    r"\\x[0-9a-fA-F]{2}",  # hex escape sequences
    r"\\[0-7]{2,3}",  # octal escapes
    r"chr\(\d+\)",  # chr() chaining
    r"\$\{[^\}]+\}",  # variable variables
    r"\$[a-zA-Z_]{1,3}\s*=",  # suspiciously short var names
    r"base64_decode\s*\(",
    r"eval\s*\(",
    r"gzinflate\s*\(",
    r"\.\s*\$[a-zA-Z_]\w*\s*\.",  # string concatenation obfuscation
]


@dataclass
class FileFeatures:
    # File metadata
    filename: str = ""
    file_size_bytes: int = 0
    extension: str = ""
    line_count: int = 0
    # Entropy
    entropy: float = 0.0  # Shannon entropy (bits per byte)
    entropy_risk: str = "Low"
    # Dangerous function hits
    dangerous_func_count: int = 0
    dangerous_funcs_found: list = None  # type: ignore
    # Obfuscation
    obfuscation_score: int = 0  # count of matched obfuscation patterns
    # Overall heuristic risk score (0–100)
    heuristic_risk: float = 0.0
    risk_label: str = "Low"

    def __post_init__(self):
        if self.dangerous_funcs_found is None:
            self.dangerous_funcs_found = []

    def to_dict(self) -> dict:
        return asdict(self)


def _shannon_entropy(text: str) -> float:
    """Computes Shannon entropy (bits per character) of a string."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values() if c)
    return round(entropy, 4)


def extract_features(text: str, filename: str = "unknown") -> FileFeatures:
    """
    Extract all heuristic features from file content.

    Parameters
    ----------
    text:     Raw file content as a string.
    filename: Original file name (used for extension detection).

    Returns
    -------
    FileFeatures dataclass with all computed signals.
    """
    ext = os.path.splitext(filename)[-1].lower() if filename != "unknown" else ""
    lines = text.splitlines()

    # ── Entropy ──────────────────────────────────────────────────────────
    entropy = _shannon_entropy(text)
    if entropy > 5.5:
        entropy_risk = "High"
    elif entropy > 4.5:
        entropy_risk = "Medium"
    else:
        entropy_risk = "Low"

    # ── Dangerous functions ───────────────────────────────────────────────
    found_funcs = []
    text_lower = text.lower()
    for func in DANGEROUS_FUNCTIONS:
        pattern = re.escape(func.lower())
        if re.search(pattern, text_lower):
            found_funcs.append(func)

    # ── Obfuscation score ─────────────────────────────────────────────────
    obf_score = 0
    for pat in OBFUSCATION_PATTERNS:
        matches = re.findall(pat, text)
        obf_score += len(matches)

    # ── Heuristic risk (0–100) ────────────────────────────────────────────
    risk = 0.0
    # Entropy: 0–30 points
    risk += min(30.0, (entropy / 8.0) * 30)
    # Dangerous functions: 0–50 points (5 pts each, capped at 10 funcs)
    risk += min(50.0, len(found_funcs) * 5)
    # Obfuscation: 0–20 points
    risk += min(20.0, obf_score * 2)

    if risk >= 60:
        risk_label = "High"
    elif risk >= 30:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return FileFeatures(
        filename=filename,
        file_size_bytes=len(text.encode("utf-8", errors="replace")),
        extension=ext,
        line_count=len(lines),
        entropy=entropy,
        entropy_risk=entropy_risk,
        dangerous_func_count=len(found_funcs),
        dangerous_funcs_found=found_funcs[:20],  # cap to 20 for response size
        obfuscation_score=obf_score,
        heuristic_risk=round(risk, 2),
        risk_label=risk_label,
    )
