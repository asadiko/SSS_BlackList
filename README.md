# Blacklist — Face-Recognition Watchlist Microservice

A standalone microservice for the Shoplifting Security System: admins enroll a
person with one or more reference photos and connect them to one or more
cameras. When **that person** (and only them) appears on **their assigned
cameras**, the service raises a `blacklist_match` warning — with the matched
face crop and similarity score as evidence.

Fully self-contained: local SQLite DB + photo storage, its own capture threads,
its own REST API and admin page. It shares only `cameras.yaml` (camera sources)
with the main stack, and optionally forwards alerts to the operator backend.

```
admin page ──► REST API ──► SQLite (persons/photos/embeddings/assignments/sightings)
                             │
RTSP cameras (assigned only) ─► capture @ FACE_FPS ─► detect ─► quality gate
   ─► embed ─► gallery match (per-camera list) ─► K-of-N confirm ─► ALERT
                                                     │
                                     crop saved + sighting row + POST to backend
```

---

## Quick start

This service lives at `/home/asado/SSS/Blacklist` and runs **independently** of
the ShopliftingSecuritySystem stack — its own compose, image, DB and volume.

```bash
cd /home/asado/SSS/Blacklist
docker compose up --build -d
# admin page →  http://<server>:8100
# API docs   →  http://<server>:8100/docs
```

To expose the GPU (needs the NVIDIA container runtime):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

**GPU/CPU is automatic** — the service uses the CUDA execution provider when
onnxruntime reports one, otherwise it runs on CPU. Same image for both; no flag.

Camera sources are read from the SSS camera registry
(`../ShopliftingSecuritySystem_SSS/cameras.yaml` by default). Point elsewhere with:

```bash
CAMERAS_FILE=/path/to/cameras.yaml docker compose up -d
```

### Run the tests

```bash
cd /home/asado/SSS/Blacklist
docker compose run --rm blacklist-test
```

---

## Using it

1. Open the admin page (`:8100`). Create a person (name + notes).
2. Upload 1–5 reference photos. Uploads are **quality-gated** — blurry photos,
   photos with several people, or faces that are too small are rejected with
   the reason, because bad reference photos are the #1 false-positive source.
   Passport-style frontal photos work best; multiple photos (with/without
   glasses, different angles) improve recall.
3. Click the camera tags to connect the person to one or more cameras.
   A capture worker starts automatically for each camera that has at least one
   assigned person (and stops when the last person is unassigned).
4. When the person is recognised, a **sighting** appears (with the face crop
   and similarity) and a `blacklist_match` event is POSTed to the backend if
   `BACKEND_URL` is configured. Review sightings with ✓ (confirmed) /
   ✗ (false match) — the review log is your threshold-tuning data.

## False-positive controls (defaults)

An alert fires only when ALL of these pass:

| Layer | Default | Env |
|---|---|---|
| Enrollment: single sharp face ≥112px, det ≥0.65 | strict | `ENROLL_*` |
| Runtime face gate: ≥80px, det ≥0.55, not blurred | permissive | `RUNTIME_*` |
| Cosine similarity threshold | 0.50 (insightface) / 0.40 (opencv) | `MATCH_THRESHOLD` |
| K-of-N: hits within window on the same camera | 3 hits / 5 s | `CONFIRM_HITS`, `CONFIRM_WINDOW_S` |
| Per-(camera, person) cooldown | 300 s | `COOLDOWN_S` |

A single-frame match can **never** alert. Raise `MATCH_THRESHOLD` and/or
`CONFIRM_HITS` for fewer FPs; lower them if real sightings are missed.

## Face engine

| Backend | Detector + embedder | Dim | When |
|---|---|---|---|
| `insightface` (preferred) | SCRFD + ArcFace (buffalo_l) | 512 | default when installed |
| `opencv` (fallback) | YuNet + SFace | 128 | if insightface unavailable |

`FACE_BACKEND=auto` (default) picks the best available. Model weights are baked
into the image at build time — **no downloads at runtime** (air-gap friendly).
If you switch backends later, re-upload photos (embeddings are per-backend;
stale ones are skipped with a warning, never mismatched).

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `BL_DATA_DIR` | `/data` | SQLite DB, photos, alert crops (mount a volume) |
| `CAMERAS_YAML` | `/app/cameras.yaml` | shared camera registry |
| `FACE_BACKEND` | `auto` | `auto` / `insightface` / `opencv` |
| `FACE_FPS` | `3` | recognition sampling rate per camera |
| `MATCH_THRESHOLD` | per-backend | cosine similarity cut-off |
| `CONFIRM_HITS` / `CONFIRM_WINDOW_S` | `3` / `5` | K-of-N confirmation |
| `COOLDOWN_S` | `300` | one alert per person/camera per this period |
| `BL_API_KEY` | *(empty)* | when set, API requires `X-API-Key` header |
| `BACKEND_URL` / `BACKEND_SERVICE_TOKEN` | *(empty)* | forward alerts to the Go backend `/internal/events` |
| `API_PORT` | `8100` | service port |

## API

`GET /docs` serves the interactive OpenAPI UI. Summary:

```
GET    /health                            liveness + engine info (open)
GET    /stats                             counts, index size, worker status
GET    /cameras                           cameras from cameras.yaml
POST   /persons                           {name, notes}
GET    /persons                           list w/ photos + assignments
GET    /persons/{id}
DELETE /persons/{id}                      deactivate (removed from matching)
POST   /persons/{id}/photos               multipart image (quality-gated)
DELETE /persons/{id}/photos/{photo_id}
PUT    /persons/{id}/cameras              {camera_ids: ["cam_door", ...]}
GET    /sightings?status=pending&limit=   alert history
POST   /sightings/{id}/verify             {status: confirmed|false_match}
```

Alert payload forwarded to the backend:

```json
{ "event_type": "blacklist_match", "sighting_id": 12, "camera_id": "cam_door",
  "person_id": 3, "person_name": "John Doe", "similarity": 0.71, "ts": 431.2 }
```

## Operational notes

- **Camera placement matters more than any threshold.** Assign blacklist
  watching to entrance/door-height cameras with near-frontal faces. Top-down
  ceiling cameras rarely produce usable faces — the quality gates make them
  produce *no* matches rather than false ones.
- The `/media/*` photo/crop routes are unauthenticated (needed by `<img>`
  tags); deploy the service on an internal network, and set `BL_API_KEY` to
  protect the API itself.
- Biometric watchlists are regulated in many jurisdictions (GDPR etc.) —
  ensure you have signage/legal basis before enabling in production.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `engine: UNAVAILABLE` in header | image built offline with no models; rebuild with network, or mount weights into `FACEWEIGHTS` |
| photo upload rejected | the reason is in the response — better photo needed |
| no alerts ever | is the person **assigned to that camera**? is the camera `enabled` with a valid `source` in cameras.yaml? check `GET /stats` → `workers` |
| too many false matches | raise `MATCH_THRESHOLD` (+0.05 steps), raise `CONFIRM_HITS`; review sightings log |
| device shows `cpu` on a GPU host | GPU not exposed to the container — use the gpu override compose file |
