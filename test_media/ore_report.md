# Vision Assistant — Intelligence Report

**Sensors:** EO + IR
**Detector:** rf-detr-base+tiled | raw detections: 43 -> refined: 33 | inference: 966 ms | mean conf: 0.39

## Sensor Analysis
### EO
(VLM reasoning disabled — refined detector output only.)

### IR
(VLM reasoning disabled — refined detector output only.)

## Confirmed Objects
**EO:**
- **person** | conf 0.86 | centre | small
- **person** | conf 0.86 | middle-left | small
- **person** | conf 0.79 | lower-left | small
- **dining table** | conf 0.58 | lower-left | small
    - interaction: at by a person
- **person** | conf 0.51 | lower-left | very small
- **dining table** | conf 0.47 | lower-left | medium
- **person** | conf 0.45 | centre | very small

## Contextual Interpretation
**EO:** scene assessed as *indoor / workspace* (cued by detected couch, dining table, laptop); class ranking refined accordingly (context refines confidence, it does not add objects).
**IR:** scene assessed as *indoor / workspace* (cued by detected toilet); class ranking refined accordingly (context refines confidence, it does not add objects).

## Human-Object Interactions
- One person appears to be at a dining table.
- One person appears to be holding a laptop (low confidence).
- One person appears to be at a bench (low confidence).
- One person appears to be at a chair (low confidence).

## Possible Objects (Low Confidence)
**EO:**
- A possible **bench** (or possibly dining table (0.32), couch (0.33)) in the centre (conf 0.45) — confidence insufficient to confirm.
- A possible **laptop** in the lower-left (conf 0.42) — confidence insufficient to confirm.
- A possible **umbrella** (or possibly person (0.30)) in the centre (conf 0.41) — confidence insufficient to confirm.
- A possible **laptop** in the lower-left (conf 0.41) — confidence insufficient to confirm.
- A possible **person** in the lower-left (conf 0.40) — confidence insufficient to confirm.
- A possible **laptop** in the lower-left (conf 0.37) — confidence insufficient to confirm.
- A possible **fire hydrant** (or possibly car (0.27)) in the lower-right (conf 0.35) — confidence insufficient to confirm.
- A possible **person** in the lower-left (conf 0.35) — confidence insufficient to confirm.
- A possible **bottle** in the lower-left (conf 0.33) — confidence insufficient to confirm.
- A possible **dining table** in the lower-left (conf 0.29) — confidence insufficient to confirm.
- A possible **laptop** in the lower-left (conf 0.29) — confidence insufficient to confirm.
- A possible **bottle** in the lower-left (conf 0.28) — confidence insufficient to confirm.
- A possible **chair** in the centre (conf 0.28) — confidence insufficient to confirm.
- A possible **suitcase** in the lower-right (conf 0.28) — confidence insufficient to confirm.
- A possible **person** in the lower-right (conf 0.27) — confidence insufficient to confirm.
- A possible **bottle** in the lower-left (conf 0.27) — confidence insufficient to confirm.
- A possible **bottle** (or possibly knife (0.27), spoon (0.26)) in the lower-left (conf 0.27) — confidence insufficient to confirm.
- A possible **bottle** (or possibly cup (0.26)) in the lower-left (conf 0.27) — confidence insufficient to confirm.
- A possible **suitcase** in the lower-right (conf 0.26) — confidence insufficient to confirm.
- A possible **chair** in the lower-left (conf 0.25) — confidence insufficient to confirm.
**IR:**
- A possible **toilet** in the middle-left (conf 0.32) — confidence insufficient to confirm.
- A possible **person** in the centre (conf 0.31) — confidence insufficient to confirm.
- A possible **kite** in the lower-centre (conf 0.30) — confidence insufficient to confirm.
- A possible **kite** in the lower-centre (conf 0.30) — confidence insufficient to confirm.
- A possible **teddy bear** in the middle-left (conf 0.27) — confidence insufficient to confirm.
- A possible **kite** in the lower-right (conf 0.27) — confidence insufficient to confirm.

## Thermal Observations
(VLM reasoning disabled — refined detector output only.)

## Scene Assessment
(VLM reasoning disabled.)

## Cross-Modal Correlation
_none_

## Fusion Assessment
_none_

## Confidence Levels
7 confirmed + 26 possible (low-confidence); mean confidence 0.39; high (>=0.60): 3; medium (0.45-0.59): 4; low (<0.45): 26

## Operator Notes
_none_