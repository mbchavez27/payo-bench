# Gold layer — datasheet

Frozen, task-ready benchmark artifacts. Results in the paper are computed from these files. Any change bumps the version and updates `MANIFEST.json`.

| Folder                 | Contents                                                          |
| ---------------------- | ----------------------------------------------------------------- |
| `emotion-recognition/` | Test set, `label-schema.json` (emotion taxonomy), model predictions |
| `advice-generation/`   | Prompts, human reference advice, model responses                  |
| `human-validation/`    | Stratified sample and rater sheets for Filipino readers           |

## Datasheet (fill in before release)

- **Version:**
- **Composition:** number of instances per task and split
- **Label schema:** see `emotion-recognition/label-schema.json`
- **Provenance:** every row keeps its Reddit `id` / `comment_id` back to bronze
- **PII:** no usernames; `author_id` is a salted hash
- **Intended use:**
- **License:**
