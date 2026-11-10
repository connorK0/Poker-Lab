"""``python -m poker_lab.play``: play against the model in your browser (local only)."""

from __future__ import annotations

import argparse
import webbrowser

from poker_lab.play.data import DataDir
from poker_lab.play.server import make_server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8790, help="port on 127.0.0.1 (default 8790)")
    ap.add_argument(
        "--data-dir",
        help="where your hands go (default: POKER_LAB_PLAY_DIR or %%LOCALAPPDATA%%/poker_lab/play)",
    )
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)
    data = DataDir(args.data_dir)
    server = make_server(args.port, data)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(
        f"Poker-Lab play: {url}  (Ctrl+C to stop)\nYour hands and reports: {data.root}", flush=True
    )
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        report = server.end_session()
        if report and report.get("report_file"):
            print(f"Session report: {report['report_file']}")
        server.server_close()


if __name__ == "__main__":
    main()
