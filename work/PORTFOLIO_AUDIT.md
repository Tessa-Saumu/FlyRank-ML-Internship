# Portfolio evidence audit — 13 October 2026

Source main `bf263f2d8381d1718ade241f72d04756f679ee22`. No previous project audit
or implementation-plan file exists in this inspected remote tree; this concise
record reconciles the earlier central portfolio evidence rather than replacing it.

## Prior work versus the older evidence report

The README is already personal, not unchanged starter boilerplate (`a43e004`),
with correct grouped/forward limitations and explicit supplied reference-code
boundaries. The paper was redesigned and its introduction/prose refined through
merged PR8 (`087335e`, `651f730`, `f61f919`). Current paper links use `main`, not
the formerly open PR5 branch. Older claims that those gaps are still untouched
are stale. work/README and capstone-report packaging still needed completion.

## Findings and repairs in this run

- Receipts show LR four wins/one loss; RF three wins/two ties at grouped P@50.
  Older paper/notebook prose claimed every-fold wins and mislabeled LR's loss
  as a tie. Corrected prose in capstone, w05, w06 and paper; retained all numbers.
- Capstone embedded a duplicate old plotting implementation that overwrote the
  redesigned charts. It now invokes the maintained figure generator; receipt
  assertions explicitly load their own inputs and assert fold win/tie/loss counts.
- Added the missing concise capstone report, personal work index and correct
  personal badge targets. Shared starter pipeline remains untouched.
- Added pinned offline-evidence environment and CI job; verifier executes every
  capstone cell/31 receipt checks and regenerates figures in an isolated temporary
  directory, preserving source evidence. This is not training or a warehouse run.

## Fresh checks and limits

HTTP 200 from the published paper, with `flyrank.ai` credit present, on 13 October
2026. Reachability is not a visual/mobile rendering check or proof the local new
corrections are deployed. Hosted Pages builds are successful. Main's latest
smoke/data-path/personalize failures have no executed steps in the inspected job
metadata; logs unavailable. No code cause, billing cause, or green-CI claim can
be inferred. The fresh offline capstone check passed all cells and 31 headline
receipt assertions; full warehouse/model rebuilding was not attempted.

Safe flagship framing: outcome-based ranking validation, leakage diagnostics and
restrained decision support. Grouped LR 0.636 versus rule 0.512 did not establish
forward lift (LR 0.54 versus April base about 0.56). No causal/business impact,
editor adoption, production ranking system or 79M-row local training claim.
