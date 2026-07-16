# Phason Evidence Audit standard

A Phason Evidence Audit (PEA) is a claim-level, independent technical evaluation. It separates four questions that are often collapsed into “reproducible”: whether the public artifact runs, whether the reported result can be reconstructed, whether the interpretation survives reasonable controls, and where the conclusion stops transferring.

## Required evidence chain

1. **Scope** — freeze one claim, acceptance criteria, source repository, commit, and compute boundary.
2. **Reconstruct** — build an environment independently and retain failed or rejected attempts in the run log.
3. **Challenge** — test ordinary variation, baselines, uncertainty, leakage risks, and explicit boundary conditions.
4. **Record** — publish observed results separately from interpretation, with machine-readable outputs.
5. **Conclude** — issue a scoped verdict and state what the audit does not establish.

## Verdict vocabulary

- **Supported** — the scoped evidence materially agrees with the claim.
- **Supported with qualifications** — the mechanism or ordering is recovered, but important scope limits remain.
- **Partially supported** — some material components hold while others cannot be reconstructed or fail.
- **Unsupported** — the scoped evidence materially contradicts the claim.
- **Unresolved** — access, compute, ambiguity, or artifact gaps prevent a defensible conclusion.

## Integrity and corrections

Recorded artifacts are pinned by SHA-256 in `evidence-manifest.json` and checked automatically by GitHub Actions. Authors and maintainers may submit context or corrections through the repository’s correction issue form. Accepted changes are versioned; prior evidence is not silently overwritten.

## Commission an audit

Founding audits begin at **$400 USD** for one tightly scoped ML claim. Compute and restricted data access are scoped separately. Enquiries: [aparajeet.shadangi@proton.me](mailto:aparajeet.shadangi@proton.me?subject=Phason%20Evidence%20Audit%20enquiry).
