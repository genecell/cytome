"""Embedded variable-length summit- & midpoint-centered peak DNA sequences.

Provides:
- :func:`store_peak_sequences`: Pure-Python memory-mapped `.fa` + `.fa.fai` reader (no `pysam` /
  `htslib` thread-local `SIGFPE` hazard) that extracts for each peak the span
  ``[min(summit, mid) - max_flank, max(summit, mid) + max_flank)`` and stores 4-bit nibble-packed
  ``A=0, C=1, G=2, T=3, N=4`` sequences in ``peak_sequences`` compressed with ``zstd``.
- :func:`get_peak_sequences`: Fast vectorized sequence retriever supporting ``center='summit'`` or
  ``center='midpoint'``, any ``window <= 2 * max_flank``, subset ``peak_indices``, reverse-complement
  augmentation, and ``encoding='uint8' | 'onehot' | 'str'``.
"""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from cytome.io.compression import compress_blob, decompress_blob
from cytome.io.gtf_import import chrom_alternatives

# Lookup table from ASCII byte (0..255) to {A=0, C=1, G=2, T=3, N=4}
_ASCII_TO_CODE = np.full(256, 4, dtype=np.uint8)
for _ch, _val in (("A", 0), ("a", 0), ("C", 1), ("c", 1), ("G", 2), ("g", 2), ("T", 3), ("t", 3)):
    _ASCII_TO_CODE[ord(_ch)] = _val

# Reverse complement lookup for codes 0..4: A(0)<->T(3), C(1)<->G(2), N(4)->N(4)
_RC_CODE_LUT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

# One-hot lookup (5, 4) for A, C, G, T, N (N -> uniform 0.25)
_ONEHOT_LUT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25, 0.25],
    ],
    dtype=np.float32,
)

_CODE_TO_ASCII = np.frombuffer(b"ACGTN", dtype=np.uint8)


def resolve_default_fasta_path(ds, fasta_path=None) -> Path:
    """The reference sequence of ``ds``: ``fasta_path`` (a FASTA or a ``.2bit``), which must exist.

    Nothing is searched for on disk and no genome is guessed from the chromosome names: the same call gives
    the same sequence on any machine, or fails.
    """
    if fasta_path is None:
        raise ValueError("the reference sequence is passed as fasta_path= (a FASTA or .2bit of the store's genome)")
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA file not found: {p}")
    return p


# Lookup table for UCSC 2bit byte -> 4 base codes in {A=0, C=1, G=2, T=3}
# In UCSC .2bit format: T=0, C=1, A=2, G=3, packed 4 bases per byte (bits 7..6, 5..4, 3..2, 1..0)
_TWOBIT_VAL_TO_CODE = np.array([3, 1, 0, 2], dtype=np.uint8)
_TWOBIT_BYTE_LUT = np.empty((256, 4), dtype=np.uint8)
for _b in range(256):
    _TWOBIT_BYTE_LUT[_b, 0] = _TWOBIT_VAL_TO_CODE[(_b >> 6) & 3]
    _TWOBIT_BYTE_LUT[_b, 1] = _TWOBIT_VAL_TO_CODE[(_b >> 4) & 3]
    _TWOBIT_BYTE_LUT[_b, 2] = _TWOBIT_VAL_TO_CODE[(_b >> 2) & 3]
    _TWOBIT_BYTE_LUT[_b, 3] = _TWOBIT_VAL_TO_CODE[_b & 3]


class _Mmap2BitReader:
    """Zero-dependency memory-mapped UCSC `.2bit` reader."""

    def __init__(self, twobit_path: Path):
        import struct
        self.fasta_path = Path(twobit_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"2bit file not found: {self.fasta_path}")
        self._fh = open(self.fasta_path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        sig_le = struct.unpack("<I", self._mm[0:4])[0]
        if sig_le == 0x1A412743:
            self._endian = "<"
        elif struct.unpack(">I", self._mm[0:4])[0] == 0x1A412743:
            self._endian = ">"
        else:
            raise ValueError(f"Invalid .2bit signature in {self.fasta_path}")
        _, version, seq_count, _ = struct.unpack(f"{self._endian}4I", self._mm[0:16])
        if version != 0:
            raise ValueError(f"Unsupported .2bit version {version}")
        self.index: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
        pos = 16
        seq_offsets = {}
        for _ in range(seq_count):
            name_len = self._mm[pos]
            pos += 1
            name = self._mm[pos : pos + name_len].decode("ascii", errors="replace")
            pos += name_len
            offset = struct.unpack(f"{self._endian}I", self._mm[pos : pos + 4])[0]
            pos += 4
            seq_offsets[name] = offset
        # Parse per-chromosome headers lazily or upfront (usually <100 contigs)
        u32_dt = np.dtype(f"{self._endian}u4")
        for name, off in seq_offsets.items():
            dna_size, n_blk_cnt = struct.unpack(f"{self._endian}2I", self._mm[off : off + 8])
            p = off + 8
            n_starts = np.frombuffer(self._mm, dtype=u32_dt, count=n_blk_cnt, offset=p).astype(np.int64)
            p += 4 * n_blk_cnt
            n_sizes = np.frombuffer(self._mm, dtype=u32_dt, count=n_blk_cnt, offset=p).astype(np.int64)
            p += 4 * n_blk_cnt
            m_blk_cnt = struct.unpack(f"{self._endian}I", self._mm[p : p + 4])[0]
            p += 4 + 8 * m_blk_cnt + 4  # skip maskStarts, maskSizes, reserved
            self.index[name] = (int(dna_size), int(p), n_starts, n_starts + n_sizes)

    def resolve_chrom(self, chrom: str) -> Optional[str]:
        if chrom in self.index:
            return chrom
        for alt in chrom_alternatives(chrom):
            if alt in self.index:
                return alt
        return None

    def fetch_codes(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch `uint8` base codes (`A=0, C=1, G=2, T=3, N=4`) for `[start, end)` with N-padding."""
        total_len = int(end - start)
        if total_len <= 0:
            return np.empty(0, dtype=np.uint8)
        out = np.full(total_len, 4, dtype=np.uint8)
        real_chrom = self.resolve_chrom(chrom)
        if real_chrom is None:
            return out
        seq_len, dna_offset, n_starts, n_ends = self.index[real_chrom]
        q_start = max(0, int(start))
        q_end = min(seq_len, int(end))
        if q_end <= q_start:
            return out

        b_start = q_start // 4
        b_end = (q_end + 3) // 4
        raw_bytes = np.frombuffer(self._mm, dtype=np.uint8, count=(b_end - b_start), offset=dna_offset + b_start)
        unpacked = _TWOBIT_BYTE_LUT[raw_bytes].ravel()
        sub = unpacked[(q_start - b_start * 4) : (q_start - b_start * 4) + (q_end - q_start)].copy()

        # Apply N-blocks overlapping [q_start, q_end)
        if len(n_starts) > 0:
            idx0 = int(np.searchsorted(n_ends, q_start, side="right"))
            idx1 = int(np.searchsorted(n_starts, q_end, side="left"))
            for k in range(idx0, idx1):
                ns = max(int(n_starts[k]), q_start) - q_start
                ne = min(int(n_ends[k]), q_end) - q_start
                if ne > ns:
                    sub[ns:ne] = 4

        dest_offset = q_start - int(start)
        out[dest_offset : dest_offset + len(sub)] = sub
        return out

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _MmapFastaReader:
    """Zero-dependency memory-mapped FASTA reader using `.fai` index."""

    def __init__(self, fasta_path: Path):
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
        self.fai_path = Path(str(self.fasta_path) + ".fai")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        if self.fai_path.exists():
            with open(self.fai_path, "rt") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 5:
                        name = parts[0].split()[0]
                        self.index[name] = (
                            int(parts[1]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                        )
        else:
            self._build_index_in_memory()

        self._fh = open(self.fasta_path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def _build_index_in_memory(self) -> None:
        """Scan an uncompressed FASTA once to build `(length, offset, line_blen, line_wlen)`."""
        with open(self.fasta_path, "rb") as f:
            chrom = None
            length = 0
            offset = 0
            line_blen = 0
            line_wlen = 0
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    if chrom is not None:
                        self.index[chrom] = (length, offset, line_blen or 1, line_wlen or 1)
                    break
                if line.startswith(b">"):
                    if chrom is not None:
                        self.index[chrom] = (length, offset, line_blen or 1, line_wlen or 1)
                    header = line[1:].strip().decode("utf-8", errors="replace")
                    chrom = header.split()[0]
                    length = 0
                    offset = f.tell()
                    line_blen = 0
                    line_wlen = 0
                else:
                    stripped = line.rstrip(b"\r\n")
                    blen = len(stripped)
                    wlen = len(line)
                    if line_blen == 0 and blen > 0:
                        line_blen = blen
                        line_wlen = wlen
                    length += blen

    def resolve_chrom(self, chrom: str) -> Optional[str]:
        if chrom in self.index:
            return chrom
        for alt in chrom_alternatives(chrom):
            if alt in self.index:
                return alt
        return None

    def fetch_codes(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Fetch `uint8` base codes (`A=0, C=1, G=2, T=3, N=4`) for `[start, end)` with N-padding."""
        total_len = int(end - start)
        if total_len <= 0:
            return np.empty(0, dtype=np.uint8)
        out = np.full(total_len, 4, dtype=np.uint8)
        real_chrom = self.resolve_chrom(chrom)
        if real_chrom is None:
            return out
        seq_len, byte_offset, line_blen, line_wlen = self.index[real_chrom]
        q_start = max(0, int(start))
        q_end = min(seq_len, int(end))
        if q_end <= q_start:
            return out

        # Byte positions in the FASTA file
        start_line, start_col = divmod(q_start, line_blen)
        end_line, end_col = divmod(q_end - 1, line_blen)
        b_start = byte_offset + start_line * line_wlen + start_col
        b_end = byte_offset + end_line * line_wlen + end_col + 1

        raw = np.frombuffer(self._mm[b_start:b_end], dtype=np.uint8)
        # Filter out newline characters (\n=10, \r=13)
        if line_wlen != line_blen:
            raw = raw[(raw != 10) & (raw != 13)]
        n_copy = min(len(raw), q_end - q_start)
        dest_offset = q_start - int(start)
        out[dest_offset : dest_offset + n_copy] = _ASCII_TO_CODE[raw[:n_copy]]
        return out

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pack_nibbles(codes: np.ndarray) -> bytes:
    """Pack `uint8` values in `{0..4}` two per byte (`(even << 4) | odd`)."""
    n = len(codes)
    if n % 2 == 1:
        padded = np.empty(n + 1, dtype=np.uint8)
        padded[:n] = codes
        padded[n] = 4
        codes = padded
    packed = (codes[0::2] << 4) | (codes[1::2] & 0x0F)
    return packed.tobytes()


def _unpack_nibbles(buf: bytes, total_bases: int) -> np.ndarray:
    """Unpack 4-bit nibble buffer back into `uint8` array of length `total_bases`."""
    packed = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(len(packed) * 2, dtype=np.uint8)
    out[0::2] = packed >> 4
    out[1::2] = packed & 0x0F
    return out[:total_bases]


def store_peak_sequences(
    ds,
    fasta_path=None,
    *,
    max_flank: int = 500,
    chunk_size: int = 4096,
    compression: str = "zstd",
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Extract and embed summit- and midpoint-centered peak DNA sequences into ``peak_sequences``.

    For every peak ``(chr, start, end_, summit)`` in ``ds.peaks``, extracts the genomic span:
    ``[min(abs_summit, abs_mid) - max_flank, max(abs_summit, abs_mid) + max_flank)``
    so that any window ``window <= 2 * max_flank`` centered on EITHER the peak summit OR the peak
    midpoint is guaranteed to lie completely inside the stored slice without boundary clipping.
    Moreover, by construction ``mid_pos == length - summit_pos``, so both centers are known
    purely from ``(length, summit_pos)``!
    """
    ds.require_writable("cytome.store_peak_sequences")
    conn = ds._conn
    n_peaks = conn.execute("SELECT COUNT(*) FROM peaks").fetchone()[0]
    if n_peaks == 0:
        raise ValueError("Cannot store peak sequences: `peaks` table is empty")

    existing = conn.execute("SELECT COUNT(*) FROM peak_sequences").fetchone()[0]
    if existing > 0 and not force:
        meta_info = ds.metadata.get("peak_sequences") or {}
        return {"status": "existing", "n_peaks": int(n_peaks), "n_chunks": int(existing), **meta_info}

    resolved_fa = resolve_default_fasta_path(ds, fasta_path=fasta_path)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(peaks)")}
    summit_expr = "summit" if "summit" in cols else "NULL"
    if "summit_offset" in cols:
        summit_expr = f"COALESCE(summit, summit_offset)"

    rows = conn.execute(
        f"SELECT peak_idx, chr, start, end_, {summit_expr} FROM peaks ORDER BY peak_idx"
    ).fetchall()

    conn.execute("DELETE FROM peak_sequences")
    max_flank = int(max_flank)
    chunk_size = int(chunk_size)
    total_bases_stored = 0
    total_compressed_bytes = 0
    n_chunks = 0

    reader_cls = _Mmap2BitReader if str(resolved_fa).lower().endswith(".2bit") else _MmapFastaReader
    with reader_cls(resolved_fa) as reader:
        for c_idx, p0 in enumerate(range(0, len(rows), chunk_size)):
            batch = rows[p0 : p0 + chunk_size]
            p1 = p0 + len(batch)
            lengths = np.empty(len(batch), dtype=np.uint16)
            summit_pos = np.empty(len(batch), dtype=np.uint16)
            seq_parts = []

            for i, (_, chrom, start, end_, sum_val) in enumerate(batch):
                s = int(start)
                e = int(end_)
                span = max(e - s, 1)
                abs_mid = (s + e) // 2
                if sum_val is not None:
                    sv = int(sum_val)
                    if 0 <= sv <= span:
                        abs_summit = s + sv
                    elif s <= sv <= e:
                        abs_summit = sv
                    else:
                        abs_summit = abs_mid
                else:
                    abs_summit = abs_mid

                # Cap extreme span difference so uint16 never overflows (65535 bp)
                if abs(abs_summit - abs_mid) > 30000:
                    abs_summit = abs_mid

                ext_start = min(abs_summit, abs_mid) - max_flank
                ext_end = max(abs_summit, abs_mid) + max_flank
                codes = reader.fetch_codes(str(chrom), ext_start, ext_end)
                lengths[i] = len(codes)
                summit_pos[i] = abs_summit - ext_start
                seq_parts.append(codes)

            concat_codes = np.concatenate(seq_parts) if seq_parts else np.empty(0, dtype=np.uint8)
            packed_bytes = _pack_nibbles(concat_codes)
            len_blob = compress_blob(lengths.tobytes(), compression)
            sum_blob = compress_blob(summit_pos.tobytes(), compression)
            seq_blob = compress_blob(packed_bytes, compression)

            conn.execute(
                """
                INSERT INTO peak_sequences (
                    chunk_idx, peak_start, peak_end, max_len,
                    lengths_blob, summit_pos_blob, seq_blob, compression
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c_idx,
                    p0,
                    p1,
                    int(2 * max_flank),
                    len_blob,
                    sum_blob,
                    seq_blob,
                    compression,
                ),
            )
            total_bases_stored += int(len(concat_codes))
            total_compressed_bytes += len(len_blob) + len(sum_blob) + len(seq_blob)
            n_chunks += 1

    conn.commit()
    info = {
        "n_peaks": len(rows),
        "n_chunks": n_chunks,
        "max_flank": max_flank,
        "max_window": 2 * max_flank,
        "total_bases": total_bases_stored,
        "compressed_bytes": total_compressed_bytes,
        "fasta": resolved_fa.name,
    }
    ds.metadata["peak_sequences"] = info
    ds.flush()
    if verbose:
        print(
            f"cytome.store_peak_sequences: stored {len(rows):,} peaks ({total_bases_stored / 1e6:.1f} Mbp) "
            f"in {total_compressed_bytes / 1e6:.1f} MB ({n_chunks} chunks)"
        )
    return info


def get_peak_sequences(
    ds,
    window: int = 500,
    *,
    center: str = "summit",
    peak_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
    encoding: str = "uint8",
    reverse_complement: bool = False,
) -> Union[np.ndarray, list[str]]:
    """Retrieve summit- or midpoint-centered DNA sequences for all or selected peaks.

    Parameters
    ----------
    ds : CytomeDataset
        Open dataset with ``peak_sequences`` populated.
    window : int
        Sequence window length in bp (default ``500``; up to ``2 * max_flank``, typically ``1000``).
    center : ``'summit'`` or ``'midpoint'``
        Center point for extracting the ``window`` bp sequence.
    peak_indices : array-like of int, optional
        0-based peak indices (or boolean mask) to fetch. If ``None``, fetches all peaks in order.
    encoding : ``'uint8'``, ``'onehot'``, or ``'str'``
        - ``'uint8'``: ``(N, window)`` ``uint8`` array with ``A=0, C=1, G=2, T=3, N=4``.
        - ``'onehot'``: ``(N, 4, window)`` ``float32`` tensor (channels ``A, C, G, T``; ``N`` is ``0.25``).
        - ``'str'``: ``list[str]`` of uppercase DNA strings.
    reverse_complement : bool
        If True, reverse-complement the extracted windows before returning.
    """
    window = int(window)
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if center not in ("summit", "midpoint"):
        raise ValueError(f"center must be 'summit' or 'midpoint', got {center!r}")
    if encoding not in ("uint8", "onehot", "str"):
        raise ValueError(f"encoding must be 'uint8', 'onehot', or 'str', got {encoding!r}")

    conn = ds._conn
    chunk_meta = conn.execute(
        "SELECT chunk_idx, peak_start, peak_end, max_len FROM peak_sequences ORDER BY chunk_idx"
    ).fetchall()
    if not chunk_meta:
        raise RuntimeError(
            "Dataset has no embedded `peak_sequences`; run `ds.store_peak_sequences(fasta_path=...)` first."
        )

    max_supported = int(chunk_meta[0][3])
    if window > max_supported:
        raise ValueError(
            f"Requested window={window} exceeds stored max_window={max_supported} "
            f"(max_flank={max_supported // 2})."
        )

    total_peaks = int(chunk_meta[-1][2])
    if peak_indices is None:
        req_idx = np.arange(total_peaks, dtype=np.int64)
        all_sequential = True
    else:
        arr = np.asarray(peak_indices)
        if arr.dtype == bool:
            if len(arr) != total_peaks:
                raise ValueError(f"Boolean mask length {len(arr)} != n_peaks {total_peaks}")
            req_idx = np.flatnonzero(arr).astype(np.int64)
        else:
            req_idx = arr.astype(np.int64)
        if np.any((req_idx < 0) | (req_idx >= total_peaks)):
            raise IndexError(f"peak_indices out of bounds [0, {total_peaks})")
        all_sequential = False

    n_out = len(req_idx)
    out_u8 = np.full((n_out, window), 4, dtype=np.uint8)
    half_left = window // 2

    if n_out > 0:
        if all_sequential:
            rows = conn.execute(
                "SELECT chunk_idx, peak_start, peak_end, lengths_blob, summit_pos_blob, seq_blob, compression "
                "FROM peak_sequences ORDER BY chunk_idx"
            ).fetchall()
            for _, p0, p1, len_blob, sum_blob, seq_blob, comp in rows:
                lengths = np.frombuffer(decompress_blob(len_blob, comp), dtype=np.uint16).astype(np.int64)
                sum_pos = np.frombuffer(decompress_blob(sum_blob, comp), dtype=np.uint16).astype(np.int64)
                tot_bases = int(lengths.sum())
                codes = _unpack_nibbles(decompress_blob(seq_blob, comp), tot_bases)

                offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
                np.cumsum(lengths, out=offsets[1:])
                centers = sum_pos if center == "summit" else (lengths - sum_pos)
                starts = offsets[:-1] + (centers - half_left)
                # Vectorized 2D gather: starts[:, None] + arange(window)[None, :]
                gather_idx = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
                out_u8[p0:p1] = codes[gather_idx]
        else:
            # Determine which chunks contain requested peaks
            chunk_starts = np.array([r[1] for r in chunk_meta], dtype=np.int64)
            chunk_of_req = np.searchsorted(chunk_starts, req_idx, side="right") - 1
            unique_chunks = np.unique(chunk_of_req)
            placeholders = ",".join("?" for _ in unique_chunks)
            rows = conn.execute(
                f"SELECT chunk_idx, peak_start, peak_end, lengths_blob, summit_pos_blob, seq_blob, compression "
                f"FROM peak_sequences WHERE chunk_idx IN ({placeholders}) ORDER BY chunk_idx",
                [int(c) for c in unique_chunks],
            ).fetchall()
            for c_idx, p0, p1, len_blob, sum_blob, seq_blob, comp in rows:
                lengths = np.frombuffer(decompress_blob(len_blob, comp), dtype=np.uint16).astype(np.int64)
                sum_pos = np.frombuffer(decompress_blob(sum_blob, comp), dtype=np.uint16).astype(np.int64)
                tot_bases = int(lengths.sum())
                codes = _unpack_nibbles(decompress_blob(seq_blob, comp), tot_bases)

                offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
                np.cumsum(lengths, out=offsets[1:])
                centers = sum_pos if center == "summit" else (lengths - sum_pos)

                mask_c = chunk_of_req == int(c_idx)
                out_positions = np.flatnonzero(mask_c)
                local_peaks = req_idx[out_positions] - int(p0)
                starts = offsets[local_peaks] + (centers[local_peaks] - half_left)
                gather_idx = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
                out_u8[out_positions] = codes[gather_idx]

    if reverse_complement:
        out_u8 = _RC_CODE_LUT[out_u8[:, ::-1]]

    if encoding == "uint8":
        return out_u8
    if encoding == "onehot":
        # _ONEHOT_LUT[out_u8] has shape (N, window, 4) -> transpose to (N, 4, window)
        return np.ascontiguousarray(_ONEHOT_LUT[out_u8].transpose(0, 2, 1))
    # encoding == 'str'
    ascii_mat = _CODE_TO_ASCII[out_u8]
    return [row.tobytes().decode("ascii") for row in ascii_mat]
