"""Play against the model: push/fold (the model learns your ranges and exploits them) and
no-limit hold'em (archetype bots or advisors, the model profiles you), on a local web page.

Start it with ``python -m poker_lab.play``. Local only: the server binds 127.0.0.1, the page
loads nothing from outside, and your hands are saved outside the repository (``data``). This is
a practice game against the program; nothing here reads hands from another site or app.
"""
