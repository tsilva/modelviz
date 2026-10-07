# Linked architecture mapping — design QA

Date: 2026-10-07

Source visual truth: `/Users/tsilva/.codex/generated_images/01a1171e-613d-7a12-b82c-b54372ff1719/exec-bd8d3671-5ab1-4548-ab12-870669a478d7.png` (displayed option 2).

Implementation: `http://127.0.0.1:55085/` in the native Codex in-app Browser.

Implementation screenshot: `/Users/tsilva/.codex/visualizations/2026/10/07/01a1171e-613d-7a12-b82c-b54372ff1719/modelviz-desktop.png`.

Full-view comparison: `/Users/tsilva/.codex/visualizations/2026/10/07/01a1171e-613d-7a12-b82c-b54372ff1719/modelviz-comparison.png`.

Focused inspector comparison: `/Users/tsilva/.codex/visualizations/2026/10/07/01a1171e-613d-7a12-b82c-b54372ff1719/modelviz-inspector-comparison.png`.

Mobile screenshot: `/Users/tsilva/.codex/visualizations/2026/10/07/01a1171e-613d-7a12-b82c-b54372ff1719/modelviz-mobile.png`.

## Comparison conditions

Desktop viewport: 1487 × 1058 CSS pixels. Both source and implementation images
are 1487 × 1058 pixels at 1× density; no scaling was used in the comparison file.
The focused comparison crops both artifacts at original density. State: split
view, first block expanded, attention selected, navigator closed, Overview tab.

The source uses illustrative GPT-2 data. The implementation uses the authored
two-block attention fixture (30 actual operators), so model names, counts, tensor
shapes and recognized groups deliberately differ. This review checks the approved
linked-pane workflow, hierarchy and visual system, not reproduction of hypothetical
model data. The user approved real mappings, inferred labels and progressive
expansion as the implementation scope.

Mobile viewport: 390 × 844 CSS pixels. With the native browser's 15-pixel vertical
scrollbar, content width is 375 pixels; the full-page image is 375 × 1688 pixels.
The graphs stack vertically and retain independent scrolling and zoom controls.
The document has no horizontal overflow, and both header actions fit inside the
375-pixel content width.

## Findings and comparison history

Initial evidence: `modelviz-desktop-initial.png` and
`modelviz-comparison-initial.png` in the screenshot directory above.

- [P2, fixed] Residual connections in the clean graph shared the centerline with
  intermediate nodes, hiding their bypass role. Skip edges now route beside the
  column; ports also account for expanded-group button height. Final full-view
  evidence visibly shows the input-to-residual bypass.
- [P2, fixed] At the narrow breakpoint the brand competed with the loading
  buttons. The below-400-pixel typography, icon size and gaps were reduced. Final
  DOM measurements place Open ONNX at x=144.74–245.89 and Browse web at
  x=257.89–361.00, within the 375-pixel content width.
- [P3, resolved] Inspector explanatory text was overly small. Evidence paragraphs
  now use 12-pixel text with 18-pixel line height; secondary notes use 11 pixels.

Final full-view and focused images were opened together with their respective
source regions after these fixes. No actionable P0/P1/P2 issues remain.

## Required fidelity surfaces

- Typography: existing Inter/system sans family retained; compact graph labels,
  clear pane headings and selected-block title. Recorded tensor and source names
  use monospace. Long names truncate in graph/list rows and remain available in
  the inspector or tooltip.
- Spacing and layout: thin header, contextual view row, two primary graph panes,
  shared bottom inspector. Optional navigator borrows the approved third concept's
  hierarchy. Flat surfaces and separators replace nested panels. Large graphs
  scroll instead of shrinking the entire model into unreadable labels.
- Colors: existing blue/teal tokens retained, purple normalization/activation,
  green inputs, blue mapping state, subtle dot grid and light dividers.
- Assets: the actual supplied ModelViz icon is reused rather than the mock's
  invented cube mark. Lucide supplies controls. Graph nodes and edges are live,
  data-driven UI, with no screenshot used as an interactive substitute.
- Copy: fabricated head counts and percentage confidence were replaced by source
  tensor data, recognition evidence and exact/pattern/scope/fused labels. Unknown
  shapes remain explicitly unrecorded. The dashed cross-pane selection link is
  distinguished from tensor edges in the README.

## Functional validation

Verified in the in-app Browser:

- Loading the authored two-block ONNX fixture and the ONNX Model Zoo MNIST-12
  export; replacing an existing loaded file resets selection and search.
- Expanding exporter blocks and recognized attention into underlying primitives,
  including inferred query/key/value roles; navigator expansion and collapse.
- Selecting architecture highlights eight actual attention operators; selecting a
  raw query projection selects its containing group and exposes its role.
- Raw node and tensor tabs, full trace list, initializer shapes and group boundary
  tensors; recorded Conv attributes from the real Model Zoo export.
- Search with no results, clearing search, selected-subgraph isolation, and Raw,
  Clean and Split modes.
- Pointer panning (raw scrollTop changed from 160 to 370), zoom, and selection
  framing. Persistent controls remain available on desktop and mobile.
- Malformed-file error retains the previously loaded model.
- Browser console checked after the final interactions: no warnings or errors.

Automated validation: `pnpm test` passes 34 tests covering parser field numbers,
signed attributes, unknown shapes, source identity, named and unnamed attention,
fused operators, dense/conv/embedding patterns, transformed weights, real tensor
boundaries, residual quotient edges, expansion, graph layout, and existing
dependency regressions. Parser regressions additionally cover repeated packed
attribute segments, all inline initializer storage formats, and truncated known
and skipped fields, packed elements, varints, and protobuf groups. Parallel
architecture branches retain their real fork and join after display reordering.
`pnpm build` succeeds.

After the parser fixes, the real MNIST export was loaded again in the in-app
Browser. A file containing a valid graph followed by an unknown field claiming
127 missing bytes was rejected with a truncation alert; the previously loaded
12-operator model remained visible. No browser warnings or errors were recorded.

## Expected limits and follow-up polish

Recognition is an explicit initial set of structural patterns; unfamiliar operators
remain exact primitives. Exporter scopes are name-based hierarchy, not architectural
proof. Fused internals are unavailable, and nested control-flow graphs are opaque
parent operators in this iteration. The hand-authored attention fixture and real
MNIST export do not establish recognition coverage across every exporter or large
production transformer. Additional exporter fixtures can extend that coverage.

At overview zoom, long tensor names require the inspector; this is intentional
rather than adding oversized nodes to every graph. More sophisticated crossing
minimization is a future graph-layout refinement.

Implementation checklist: completed visual repairs, desktop/mobile interaction
checks, source comparison, automated regressions and production build.

final result: passed
