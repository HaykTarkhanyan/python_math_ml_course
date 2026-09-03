"""Fetch 1000 Genomes chr22 genotypes and cache a small teaching matrix.

Instructor-side, run once (like fetch_fashion_mnist.py). Students and the live
"genes mirror geography" practical load the committed
``data/genomes_1000g_chr22.npz`` -- numpy only, no bioinformatics stack.

Source: 1000 Genomes phase 3 (open access), chromosome 22 + the sample panel:
  http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
2,504 individuals, 26 populations, 5 super-populations.

Filtering (streamed, nothing big held in memory):
  - biallelic single-nucleotide variants only (one REF base, one ALT base),
  - common: 0.05 <= AF <= 0.95 (a rare variant is individually too noisy to help
    a PCA at this scale, and Patterson scaling would upweight it into noise --
    they also triple the matrix),
  - every KEEP_EVERY-th passing SNP, as a cheap stand-in for LD thinning.

Output npz:
  genotypes  (2504, n_snps) int8 -- 0/1/2 copies of the alternate allele
  samples, pop, super_pop         -- one entry per row
  snp_id, pos, af                 -- one entry per column (NOTE: the v5b VCF ships
                                     "." for every ID -- no rsIDs; kept for layout)
  source                          -- the exact URL, for provenance

Run:  ./ma/Scripts/python.exe ml/10_dimensionality_reduction/py_src/fetch_1000g_genotypes.py
(~205 MB download on first run, cached in ~/.cache/1000genomes; the parse streams
the whole chromosome and takes a few minutes.)
"""
from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np

CHAPTER = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
OUT_NPZ = CHAPTER / "data" / "genomes_1000g_chr22.npz"

CACHE = Path.home() / ".cache" / "1000genomes"
BASE = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
VCF_NAME = "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
PANEL_NAME = "integrated_call_samples_v3.20130502.ALL.panel"

MAF_MIN = 0.05
KEEP_EVERY = 6          # every 6th passing SNP; measured: 97,216 passing -> 16,202 columns

# The four phased diploid genotypes a kept biallelic SNP can have. Anything else
# (missing data, multi-digit alleles) is unexpected for this release -> raise.
_GT = {"0|0": 0, "0|1": 1, "1|0": 1, "1|1": 2}


def _download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        logging.info("cached: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest
    url = f"{BASE}/{name}"
    logging.info("downloading %s ...", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    logging.info("done: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def _parse_af(info: str) -> float:
    for field in info.split(";"):
        if field.startswith("AF="):
            return float(field[3:])
    raise ValueError(f"no AF= field in INFO: {info[:120]}")


def main() -> None:
    vcf = _download(VCF_NAME)
    panel = _download(PANEL_NAME)

    rows, meta = [], []            # meta: (snp_id, pos, af)
    samples = None
    n_lines = n_pass = 0
    with gzip.open(vcf, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                logging.info("%d samples in VCF header", len(samples))
                continue
            n_lines += 1
            if n_lines % 200_000 == 0:
                logging.info("  scanned %dk variants, kept %d", n_lines // 1000, len(rows))
            chrom, pos, snp_id, ref, alt, _qual, _filt, info, _fmt, gts = line.split("\t", 9)
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                continue
            af = _parse_af(info)
            if not (MAF_MIN <= af <= 1 - MAF_MIN):
                continue
            n_pass += 1
            if n_pass % KEEP_EVERY:
                continue
            try:
                row = np.array([_GT[g] for g in gts.rstrip("\n").split("\t")], dtype=np.int8)
            except KeyError as e:
                raise ValueError(f"unexpected genotype {e} at {chrom}:{pos}") from None
            if len(row) != len(samples):
                raise ValueError(f"{len(row)} genotypes vs {len(samples)} samples at {chrom}:{pos}")
            rows.append(row)
            meta.append((snp_id, int(pos), af))

    X = np.vstack(rows).T          # (samples, snps)
    logging.info("matrix: %s from %d scanned / %d passing variants", X.shape, n_lines, n_pass)

    pop_by_sample, super_by_sample = {}, {}
    for ln in panel.read_text().splitlines()[1:]:
        if ln.strip():
            sid, pop, sup = ln.split("\t")[:3]
            pop_by_sample[sid], super_by_sample[sid] = pop, sup
    missing = [s for s in samples if s not in pop_by_sample]
    if missing:
        raise ValueError(f"{len(missing)} VCF samples missing from panel, e.g. {missing[:3]}")

    OUT_NPZ.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        genotypes=X,
        samples=np.array(samples),
        pop=np.array([pop_by_sample[s] for s in samples]),
        super_pop=np.array([super_by_sample[s] for s in samples]),
        snp_id=np.array([m[0] for m in meta]),
        pos=np.array([m[1] for m in meta], dtype=np.int64),
        af=np.array([m[2] for m in meta], dtype=np.float32),
        source=f"{BASE}/{VCF_NAME}",
        maf_min=MAF_MIN, keep_every=KEEP_EVERY,
    )
    logging.info("wrote %s (%.1f MB)", OUT_NPZ, OUT_NPZ.stat().st_size / 1e6)


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / "fetch_1000g_genotypes.log", encoding="utf-8")],
    )
    main()
