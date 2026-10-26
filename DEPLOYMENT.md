# Deploying DataLens to Streamlit Community Cloud

DataLens needs no database, secrets or background workers. Streamlit Community Cloud runs it straight from
a GitHub repository.

## 1. What Community Cloud uses from this repo

| File | Role |
|---|---|
| `app.py` | Entrypoint. Select it as the "Main file path". |
| `requirements.txt` | **The** dependency file. Community Cloud uses only one dependency file, and `requirements.txt` takes priority over `pyproject.toml`. That ordering matters, because `pyproject.toml` here only configures pytest and declares no dependencies. |
| `.streamlit/config.toml` | Theme, 50 MB upload limit, usage statistics disabled. Read automatically. |
| `requirements-dev.txt`, `tests/`, `.github/` | Development only. Ignored by Community Cloud. |

### Runtime pinning

- **Python version.** Community Cloud doesn't read `runtime.txt` or `.python-version`. You choose the
  version in **Advanced settings** at deploy time, and the default is 3.12. Choose **3.14**, the version
  DataLens was developed and tested on (`.python-version` records it for local tools like uv and pyenv).
  The pinned stack also resolves to binary wheels on 3.12 and 3.13, and CI tests all three. 3.11 and
  older will **not** work (numpy 2.5 needs Python 3.12 or newer).
- **Packages.** Every direct dependency is pinned exactly (`==`). One transitive package is pinned too:
  `wrapt`. statsmodels → formulaic declares `wrapt>=1.17.0rc1` on Python 3.13+, and a pre-release in a
  specifier makes pip install the newest *release candidate*. The pin keeps it on a stable release.
  Other transitive packages float within their parents' constraints.
- **Verifying a pin change** without that Python installed (resolves Linux wheels like Community Cloud):

  ```bash
  python -m pip install --dry-run --ignore-installed --only-binary=:all: --python-version 3.14 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 --target _resolve -r requirements.txt
  ```

## 2. Before you deploy

1. `python -m pytest -q` passes locally. This includes `tests/test_app.py`, which runs `app.py` headlessly
   and walks through every workflow step.
2. Push to GitHub on `main`. The **tests** workflow (`.github/workflows/tests.yml`) runs the suite on
   Python 3.12, 3.13 and 3.14. Deploy only from a green commit.
3. The repository may be public or private. Community Cloud can deploy from private repos once you grant
   it access.

## 3. Deploy

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Click **Create app** → **Yup, I have an app**.
3. Choose the repository, branch `main`, and main file path `app.py`.
4. Optionally pick an **App URL** subdomain, for example `datalens-<yourname>`.
5. Open **Advanced settings**, set **Python version = 3.14**, leave Secrets empty, and **Save**.
6. Click **Deploy**. The first build takes a few minutes, mostly installing scipy, statsmodels, scikit-learn
   and pyarrow. Build logs appear in the panel on the right. Only users with write access to the repo
   can see them.

After that, every push to `main` redeploys automatically.

**Changing Python later** requires deleting the app and redeploying with the same repo, branch, file and
subdomain, then choosing the new version under Advanced settings. Note the subdomain first.

## 4. After deploying: smoke check (2 minutes)

1. The welcome screen shows the upload area and the sample-data panel; steps 01–04 are locked.
2. **Gunakan contoh data sintetis**: the Data step renders, with the histograms and correlation heatmap.
3. **Regresi**: select `jam_belajar`, `jam_tidur`, `kehadiran_pct` → **Estimasi model**. The slope for
   `jam_belajar` should be ≈ 2.38 (sample data, seed 42).
4. **Diagnostik**: four plots and the test table render.
5. **Regresi → Perbandingan model** → **Bandingkan model** finishes in a few seconds.
6. **Laporan** → **Buat laporan** → download. The file opens offline with working charts.
7. Upload a CSV of your own, including one with `;` separators and decimal commas (Opsi format CSV).

## 5. Operating notes and limits

- **Resources.** Community Cloud apps get a modest, shared CPU/RAM allowance. DataLens keeps whole
  datasets in memory, and diagnostics and CV scale with rows. Datasets up to roughly a few hundred
  thousand rows × tens of columns are comfortable. The 50 MB upload cap (`config.toml`) is the hard limit.
- **Sleeping.** Apps with no traffic for a while go to sleep. The first visitor wakes them with one click,
  which takes a few seconds.
- **Privacy.** Uploaded files are parsed in server memory for the current session only. DataLens writes
  nothing to disk or to a database, and nothing persists after the session ends. A public app still means
  anyone with the link can use it, so don't present it as a place for confidential data. Usage
  statistics are disabled (`browser.gatherUsageStats = false`).
- **Reports** with charts are about 5 MB each, because plotly.js is embedded for offline viewing. Users who
  need a smaller file can uncheck "Sertakan grafik interaktif".

## 6. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| Build fails resolving `numpy==2.5.4` | Python ≤ 3.11 selected → redeploy with 3.14 (see "Changing Python later"). |
| `ImportError: cannot import name …` right after a push | A stale module in a long-running session. Use **Reboot app** from the app menu. Locally, restart `streamlit run`. |
| App shows "Terjadi kesalahan tak terduga di bagian ini" | A bug in one section, which the error boundary contained. Open "Detail teknis" for the traceback and check the logs. |
| Slow first load | Cold start after sleep, or a fresh build. Wait for the build log to show the app is running. |
