Teach the scripts that edit drawings to read and write the Excalidraw plugin's `.excalidraw.md` format, then convert the drawings. ^desc

## Why
Obsidian's Excalidraw plugin opens our `.excalidraw` files in "compatibility mode" and offers to convert them to `.excalidraw.md`, a Markdown note with the drawing inside, usually compressed. **Don't accept the conversion yet.**
- `map.py` (research map), `roadmap.py` (todo roadmap) and `diagram.py` (architecture diagram) only understand plain `.excalidraw` JSON.
- After a conversion they would keep editing the old file, or find it gone, and hand edits and script edits would drift apart.

Compatibility mode loses nothing for viewing and editing. The new format adds Obsidian features: links inside the drawing that show in backlinks and the graph, and Markdown embeds.

## How
1. In `mapmodel.py` (research-map skill) and `diagram.py` (doc-diagram skill), read and write `.excalidraw.md`. The plugin accepts an uncompressed `json` block in the note's Drawing section; keep the front matter and the text-element section the plugin expects.
2. Convert `docs/research/research_map.excalidraw`, `docs/todo_roadmap.excalidraw` and `docs/architecture/aws_architecture.excalidraw` in one go, then check each one renders and opens in Obsidian.
3. Update the skills' docs (research-map, todo-board, doc-diagram).
