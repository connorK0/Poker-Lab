"""Interactive results explorer: every phase's results in one offline HTML page.

    python -m experiments.explorer            # rebuild experiments/results/explorer.html

The builder reads the committed results JSON, turns each finding into a *section* (controls,
precomputed views, charts, takeaways) and inlines the data, ``static/app.js`` and
``static/app.css`` into one self-contained file. Nothing is fetched at view time.

Adding results for a new experiment means adding a section in that phase's module
(``phase1.py``, ``phase2.py``, ...). ``tests/test_explorer.py`` fails if a results file has
no section reading it, or if the committed page is stale.
"""
