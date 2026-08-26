# V7.1 Proof/Search Separation: 24-Group Diagnostic

Date: 2026-08-27

## Change under test

The peak witness is feasibility-only. Its recovered Bay Patterns are retained
as explicitly registered feasibility-proof columns in the LP and integer
masters, but its group-area support no longer expands `A_restricted[g]`.
Stage 1 constructs the pricing domain only from its best support, diversified
near-optimal pool, and metadata-safety ranking. Dynamically priced columns must
remain inside that domain.

The V7 business model, objective, peak cap, exact pricing implementation,
120-second production budget, and 120-second Complete-MIP baseline were not
changed.

## Correctness gates

- 138 unit/micro tests passed.
- An outside-domain pattern is accepted only when its exact signature is
  registered as a proof column.
- Proof columns do not change the allowed group set used by pricing.
- Independently validated integer incumbents remain feasible for the complete
  V7 business model.

## Single-seed results

Case: `scale_g024_s401`.

| Configuration | Group-area edges | Group-bay edges | Root LP | V7 UB | Complete-MIP UB | V7 UB delta |
|---|---:|---:|---:|---:|---:|---:|
| Feasibility witness support mandatory, cap 2 | 152 | 1,417 | 0.048385 | 0.057558 | 0.045754 | +25.80% |
| Proof/search separated, cap 2 | 85 | 881 | 0.052099 | 0.060389 | 0.045754 | +31.99% |
| Proof/search separated, cap 3 | 109 | 1,364 | 0.052099 | 0.060389 | 0.045754 | +31.99% |

For separated cap 2, the RIM proved its supplied pool optimal in about 41.2
seconds. For cap 3, the final restricted gap was approximately 0.0068% after
the full production budget. The unchanged UB and root LP show that the extra
candidate slot added no useful column combination.

## Finding

Proof/search separation is a cleaner and correctly enforced architecture, but
it does not improve solution quality under the current Stage-1 candidate
mechanism. The failure is not caused by insufficient RIM search time or by the
peak hard cap: the restricted LP itself is already above the full-model
Complete-MIP incumbent.

The next useful diagnostic is an oracle/coverage comparison between the
Complete-MIP incumbent group-area support and the Stage-1 ranking. Blindly
raising `additional_candidate_area_cap` should not be treated as a remedy.

That audit has now been completed. Only 4 of the incumbent's 28 group-area
pairs are covered under either cap 2 or cap 3; see
`oracle_coverage_24_20260827/REPORT.md`.

## Artifacts

- `preexperiment_outputs/v7_1_proof_search_separation_24_20260827`
- `preexperiment_outputs/v7_1_proof_search_separation_cap3_24_20260827`
- Historical mandatory-witness comparison:
  `preexperiment_outputs/v7_1_feasibility_witness_24_48_96_20260827`
