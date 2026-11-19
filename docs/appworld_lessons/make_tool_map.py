"""Write the CugaLite tool name -> AppWorld API map that lesson retrieval needs (CUGA_LESSON_TOOL_MAP).

CUGA names an AppWorld API `<app>_<api_name>_<path without the app prefix>_<method>`; the AppWorld API docs give the
inverse. Usage: python make_tool_map.py <appworld data dir>/api_docs/standard tool_to_api.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuga_to_react_layout import tool_to_api  # noqa: E402

if __name__ == "__main__":
    api_docs_dir, out_path = sys.argv[1], sys.argv[2]
    mapping = tool_to_api(api_docs_dir)
    Path(out_path).write_text(json.dumps(mapping, indent=0, sort_keys=True))
    print(f"tool map: {len(mapping)} names -> {out_path}")
