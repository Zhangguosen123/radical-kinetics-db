import streamlit as st
import pandas as pd
import numpy as np

# 检测RDKit是否安装成功
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    st.success("RDKit 导入成功！")
except ImportError as e:
    st.error(f"RDKit 导入失败：{e}")
    st.error("请检查RDKit是否正确安装，或确认Python版本为3.12")
    st.stop()  # 停止应用运行，避免后续报错

# 你的原有代码...
import os
import re
import sqlite3
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ===================== RDKit imports =====================
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D

# ===================== Page config =====================
st.set_page_config(
    page_title="RadLogk-AI | Reaction Kinetics Database",
    page_icon="🧪",
    layout="wide",
)

# ===================== CSS =====================
st.markdown(
    r"""
<style>
/* base */
html, body, [class*="css"] {font-family: "Times New Roman", Times, serif;}
body {background-color: #F6F7FB;}
a {text-decoration:none;}
.small{color:#6B7280; font-size: 13.5px; line-height:1.35;}
.hr{height:1px;background:#EEF0F3;margin:10px 0;}

/* top bar */
.topbar{
  background: linear-gradient(90deg, #0B2A6F 0%, #133B9A 60%, #0B2A6F 100%);
  padding: 14px 18px;
  border-radius: 14px;
  color: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  margin: 10px 0 14px 0;
  display:flex;
  align-items:center;
  justify-content:center;
  position:relative;
}
.brand-badge{
  position:absolute;
  left:14px;
  width:38px;height:38px;
  border-radius:10px;
  background: rgba(255,255,255,0.14);
  border:1px solid rgba(255,255,255,0.22);
  display:flex; align-items:center; justify-content:center;
  font-weight: 800;
  font-size: 14px;
}
.topbar h1{font-size: 22px; margin: 0; font-weight: 800; text-align:center;}

/* cards */
.card{
  background: white;
  border-radius: 14px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  padding: 14px 14px;
  border: 1px solid #EEF0F3;
  margin-bottom: 12px;
}
.card h3{margin:0 0 8px 0; font-size: 18px;}
.section-title{
  font-size: 20px;
  font-weight: 900;
  margin: 0 0 10px 0;
  color:#0F172A;
}

/* pills */
.pill{
  display:inline-block; padding:6px 10px; border-radius: 999px;
  background:#F1F5F9; border:1px solid #E2E8F0;
  font-size: 13.5px; margin-right: 6px; margin-bottom: 8px;
}

/* result entry (vertical layout) */
.entry-head{
  font-weight: 900;
  font-size: 18px;
  color:#0F172A;
  margin-bottom: 10px;
}
.vbox{
  background:#F8FAFC;
  border:1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
  margin-bottom: 8px;
}
.vtitle{
  font-size: 13.5px;
  color:#64748B;
  margin-bottom: 4px;
  font-weight: 700;
}
.vvalue{
  font-size: 17px;
  color:#0F172A;
  font-weight: 800;
  word-break: break-word;
  line-height:1.25;
}

/* calculations: descriptor grid (two columns, compact, bigger text) */
.desc-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:10px;
}
.desc-box{
  background:#F8FAFC;
  border:1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
}
.desc-name{
  font-size: 14px;
  color:#64748B;
  font-weight: 800;
  margin-bottom: 4px;
}
.desc-val{
  font-size: 18px;   /* bigger */
  color:#0F172A;
  font-weight: 900;
}

/* monospaced wrapped display for fingerprints (no horizontal scroll) */
.mono-wrap{
  font-family: "Courier New", Courier, monospace;
  font-size: 14.5px;
  line-height: 1.35;
  background: #F8FAFC;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 10px 12px;
  white-space: pre-wrap;
  word-break: break-all;
}

/* buttons */
.stButton>button {border-radius: 10px; padding: 10px 12px; width: 100%; font-weight: 900;}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Files =====================
# 关键修改：改为相对路径，与.py文件同目录
RADICAL_FILES = {
    "Hydroxyl radical (·OH)": "Hydroxyl_radical_ultimate.csv",
    "Sulfate radical (SO4•−)": "Sulfate_radical_ultimate.csv",
    "Carbonate radical (CO3•−)": "Carbonate_radical_ultimate.csv",
    "Ozone (O3)": "Ozone_ultimate.csv",
    "Free chlorine (HOCl/OCl−)": "FreeChlorine_ultimate.csv",
    "Chlorine radical (Cl•)": "ChlorineRadical_ultimate.csv",
    "Dichlorine radical (Cl2•−)": "DichlorineRadical_ultimate.csv",
    "Singlet Oxygen (1O2)": "SingletOxygen_ultimate.csv"
}
REQUIRED_COLS = ["Chemical compound", "Cas", "Smiles", "Logk", "Chemical_class_27", "Ph", "T", "Ref"]

# ===================== Metrics DB (aggregate only) =====================
DB_PATH = "radlogk_metrics.db"

def _db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
    """)
    for k in ["visits", "queries", "downloads"]:
        cur.execute("INSERT OR IGNORE INTO metrics(key, value) VALUES(?, ?)", (k, 0))
    conn.commit()
    conn.close()

def _db_get_all():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM metrics")
    rows = cur.fetchall()
    conn.close()
    return {k: int(v) for k, v in rows}

def _db_inc(key: str, n: int = 1):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE metrics SET value = value + ? WHERE key = ?", (n, key))
    conn.commit()
    conn.close()

_db_init()

# visits: count once per session (no identifiers)
if "visit_counted" not in st.session_state:
    st.session_state["visit_counted"] = True
    _db_inc("visits", 1)

# calculation cache (per session)
if "calc_cache" not in st.session_state:
    st.session_state["calc_cache"] = {}  # key: f"{system}__{rid}" -> dict

# last search persistence (avoid losing results on rerun)
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None  # DataFrame
if "last_system" not in st.session_state:
    st.session_state["last_system"] = list(RADICAL_FILES.keys())[0]
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# developer unlock state
if "dev_unlocked" not in st.session_state:
    st.session_state["dev_unlocked"] = False

# ===================== Query params helper =====================
def get_query_params():
    try:
        qp = st.query_params
        return dict(qp)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def qp_has_dev_flag() -> bool:
    qp = get_query_params()
    v = None
    if "dev" in qp:
        v = qp["dev"]
        if isinstance(v, list) and len(v) > 0:
            v = v[0]
    return str(v).strip() in ["1", "true", "yes", "on"]

# ===================== Utils =====================
def norm_text(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").map(lambda x: x.strip()).replace({"nan":"", "NaN":""})

def is_cas_like(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^\d{2,7}-\d{2}-\d$", s))

@st.cache_data(show_spinner=False)
def load_data():
    out = {}
    for system, path in RADICAL_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig").copy()

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{system} dataset missing required columns: {missing}")

        for col in ["Chemical compound", "Cas", "Smiles", "Ref", "Chemical_class_27"]:
            df[col] = safe_str_series(df[col])

        df["_name_norm"] = df["Chemical compound"].map(norm_text)
        df["_cas_norm"] = df["Cas"].map(norm_text).map(lambda x: x.replace(" ", ""))
        df["_rid"] = np.arange(1, len(df) + 1)
        out[system] = df
    return out

def mol_from_smiles(smiles: str):
    s = str(smiles).strip()
    if s == "" or s.lower() in ["nan", "none", "unrecorded"]:
        return None
    return Chem.MolFromSmiles(s)

def rdkit_svg(smiles: str, w=260, h=190) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# descriptors
DESCRIPTOR_FUNCS = {
    "MolWt": lambda m: float(Descriptors.MolWt(m)),
    "MolLogP": lambda m: float(Descriptors.MolLogP(m)),
    "TPSA": lambda m: float(rdMolDescriptors.CalcTPSA(m)),
    "HBD": lambda m: float(Lipinski.NumHDonors(m)),
    "HBA": lambda m: float(Lipinski.NumHAcceptors(m)),
    "NumRotatableBonds": lambda m: float(Lipinski.NumRotatableBonds(m)),
    "RingCount": lambda m: float(Lipinski.RingCount(m)),
    "NumAromaticRings": lambda m: float(Lipinski.NumAromaticRings(m)),
    "HeavyAtomCount": lambda m: float(Lipinski.HeavyAtomCount(m)),
    "FractionCSP3": lambda m: float(Lipinski.FractionCSP3(m)),
    "MolMR": lambda m: float(Descriptors.MolMR(m)),
}
ALL_DESC_NAMES = list(DESCRIPTOR_FUNCS.keys())

def compute_descriptors(smiles: str, selected: list[str]) -> dict:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES cannot be parsed; descriptors unavailable.")
    return {k: float(DESCRIPTOR_FUNCS[k](mol)) for k in selected}

def compute_morgan_bits(smiles: str, n_bits: int = 1024, radius: int = 2) -> list[int]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES cannot be parsed; Morgan unavailable.")
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))
    arr = np.zeros((int(n_bits),), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.tolist()

def compute_maccs_bits(smiles: str) -> list[int]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("SMILES cannot be parsed; MACCS unavailable.")
    bv = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
    arr = np.zeros((bv.GetNumBits(),), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.tolist()

def summarize_bits(bits: list[int]) -> dict:
    arr = np.array(bits, dtype=int)
    return {
        "length": int(arr.size),
        "on_bits": int(arr.sum()),
        "on_ratio": float(arr.mean()),
    }

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def inc_query(): _db_inc("queries", 1)
def inc_download(): _db_inc("downloads", 1)

def cache_key(system: str, rid: int) -> str:
    return f"{system}__{rid}"

def get_cache(system: str, rid: int) -> dict:
    return st.session_state["calc_cache"].get(cache_key(system, rid), {})

def set_cache(system: str, rid: int, field: str, value):
    k = cache_key(system, rid)
    st.session_state["calc_cache"].setdefault(k, {})
    st.session_state["calc_cache"][k][field] = value

def availability_summary(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"n": 0, "class":"—", "ph_range":"—", "t_range":"—", "smiles_pct":"—", "ref_pct":"—"}

    n = len(df)
    cls = df["Chemical_class_27"].fillna("").astype(str).str.strip()
    cls = cls[cls != ""]
    class_top = cls.value_counts().index[0] if len(cls) else "—"

    phs = pd.to_numeric(df["Ph"], errors="coerce").dropna()
    ts = pd.to_numeric(df["T"], errors="coerce").dropna()
    ph_range = f"{phs.min():.2f} – {phs.max():.2f}" if len(phs) else "—"
    t_range = f"{ts.min():.1f} – {ts.max():.1f}" if len(ts) else "—"

    smiles_ok = df["Smiles"].fillna("").astype(str).str.strip()
    ref_ok = df["Ref"].fillna("").astype(str).str.strip()
    smiles_pct = f"{(smiles_ok!='').mean()*100:.1f}%"
    ref_pct = f"{(ref_ok!='').mean()*100:.1f}%"

    return {"n": n, "class": class_top, "ph_range": ph_range, "t_range": t_range, "smiles_pct": smiles_pct, "ref_pct": ref_pct}

def fmt_value(x, nd=6):
    if x is None:
        return "—"
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return "—"
    if s.lower() == "nan":
        return "nan"
    try:
        v = float(s)
        if np.isnan(v):
            return "nan"
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return s

# ===================== Load data =====================
data_map = load_data()

# ===================== Top bar =====================
st.markdown(
    """
<div class='topbar'>
  <div class='brand-badge'>RLAI</div>
  <h1>RadLogk-AI: Reaction Kinetics Database</h1>
</div>
""",
    unsafe_allow_html=True,
)

# ===================== 3-column layout =====================
col_left, col_mid, col_right = st.columns([1.05, 1.35, 1.60], gap="large")

# ===================== LEFT: Search (+ Developer hidden) =====================
with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Search</div>", unsafe_allow_html=True)

    system = st.selectbox(
        "Radical/oxidant system",
        list(RADICAL_FILES.keys()),
        index=list(RADICAL_FILES.keys()).index(st.session_state["last_system"])
        if st.session_state["last_system"] in RADICAL_FILES else 0
    )
    st.session_state["last_system"] = system

    q = st.text_input(
        "Chemical name or CAS",
        value=st.session_state.get("last_query", ""),
        placeholder="e.g., acetaminophen OR 71-55-6"
    )

    b1, b2 = st.columns([1, 1])
    with b1:
        do_search = st.button("Search", type="primary")
    with b2:
        do_clear = st.button("Clear")

    st.markdown(
        "<div class='small'>Name matching tolerates spaces/hyphens. CAS supports exact or partial matching.</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if do_clear:
        st.session_state["last_results"] = None
        st.session_state["last_query"] = ""
        st.rerun()

    if do_search:
        inc_query()
        st.session_state["last_query"] = q

        q0 = (q or "").strip()
        if q0 == "":
            st.session_state["last_results"] = None
        else:
            df_sys = data_map[system]
            q_norm = norm_text(q0)
            q_cas = norm_text(q0).replace(" ", "")

            idx_set = set()

            if is_cas_like(q0):
                mask = df_sys["_cas_norm"].str.contains(re.escape(q_cas), na=False)
                idx_set |= set(df_sys.loc[mask, "_rid"].tolist())
            else:
                mask1 = df_sys["_name_norm"].str.contains(re.escape(q_norm), na=False)
                idx_set |= set(df_sys.loc[mask1, "_rid"].tolist())

                q2 = re.sub(r"[\s\-]", "", q_norm)
                name2 = df_sys["_name_norm"].map(lambda x: re.sub(r"[\s\-]", "", x))
                mask1b = name2.str.contains(re.escape(q2), na=False)
                idx_set |= set(df_sys.loc[mask1b, "_rid"].tolist())

                mask2 = df_sys["_cas_norm"].str.contains(re.escape(q_cas), na=False)
                idx_set |= set(df_sys.loc[mask2, "_rid"].tolist())

            if len(idx_set) == 0:
                st.session_state["last_results"] = pd.DataFrame()
            else:
                res = df_sys[df_sys["_rid"].isin(sorted(idx_set))].copy()
                st.session_state["last_results"] = res

        st.rerun()

    # -------- Developer (hidden unless dev flag) --------
    dev_ui_enabled = qp_has_dev_flag() or (os.environ.get("RADLOGK_FORCE_DEV_UI", "").strip() == "1")
    if dev_ui_enabled:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Developer</div>", unsafe_allow_html=True)

        dev_key_input = st.text_input("Developer key", type="password", value="")
        if st.button("Unlock developer view"):
            expected = os.environ.get("RADLOGK_DEV_KEY", "")
            st.session_state["dev_unlocked"] = (expected != "" and dev_key_input == expected)

        if st.session_state.get("dev_unlocked", False):
            m = _db_get_all()
            a, b, c = st.columns(3)
            a.metric("Visits", m.get("visits", 0))
            b.metric("Queries", m.get("queries", 0))
            c.metric("Downloads", m.get("downloads", 0))
            st.markdown("<div class='small'>Aggregate counts only (no IP / no identifiers).</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='small'>Locked. Metrics are hidden for non-developers.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ===================== MID: Records + Results (all matched entries) =====================
with col_mid:
    res = st.session_state.get("last_results", None)

    # Records (stats + download)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Records</div>", unsafe_allow_html=True)

    if res is None:
        st.markdown("<div class='small'>Enter a chemical name or CAS and click Search.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if len(res) == 0:
            st.error("No matched records. Try a shorter keyword or verify the CAS/name.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success(f"{len(res)} matched record(s).")

            s = availability_summary(res)
            st.markdown(
                f"""
<span class="pill">Records: <b>{s['n']}</b></span>
<span class="pill">Class: <b>{s['class']}</b></span>
<span class="pill">pH range: <b>{s['ph_range']}</b></span>
<span class="pill">T range (°C): <b>{s['t_range']}</b></span>
<span class="pill">SMILES available: <b>{s['smiles_pct']}</b></span>
<span class="pill">Ref available: <b>{s['ref_pct']}</b></span>
""",
                unsafe_allow_html=True,
            )

            core_df = res[["_rid","Chemical compound","Cas","Smiles","Logk","Chemical_class_27","Ph","T","Ref"]].rename(
                columns={"_rid":"Record_ID"}
            )
            st.download_button(
                "Download results CSV (core fields)",
                data=to_csv_bytes(core_df),
                file_name=f"{st.session_state['last_system']}_query_results.csv",
                mime="text/csv",
                on_click=inc_download,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Results (cards, ALL matched entries)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)

            # tighter spacing: no extra blank area
            for i, (_, r) in enumerate(res.iterrows(), start=1):
                rid = int(r["_rid"])
                name = (r.get("Chemical compound","") or "").strip()
                cas = (r.get("Cas","") or "").strip()
                smiles = (r.get("Smiles","") or "").strip()
                logk = fmt_value(r.get("Logk",""), nd=9)
                cclass = (r.get("Chemical_class_27","") or "").strip() or "—"
                ph = fmt_value(r.get("Ph",""), nd=6)
                t = fmt_value(r.get("T",""), nd=6)
                ref = (r.get("Ref","") or "").strip()

                st.markdown(f"<div class='entry-head'>#{i} | Record #{rid}</div>", unsafe_allow_html=True)

                # optional structure (kept small)
                svg = rdkit_svg(smiles)
                if svg:
                    st.markdown(svg, unsafe_allow_html=True)

                # vertical-only fields
                def _ref_html(x: str) -> str:
                    if x and x.startswith("http"):
                        return f"<a href='{x}' target='_blank'>{x}</a>"
                    return x if x else "—"

                st.markdown(
                    f"""
<div class="vbox"><div class="vtitle">Chemical</div><div class="vvalue">{name if name else "—"}</div></div>
<div class="vbox"><div class="vtitle">CAS</div><div class="vvalue">{cas if cas else "—"}</div></div>
<div class="vbox"><div class="vtitle">Logk</div><div class="vvalue">{logk}</div></div>
<div class="vbox"><div class="vtitle">Class</div><div class="vvalue">{cclass}</div></div>
<div class="vbox"><div class="vtitle">pH</div><div class="vvalue">{ph}</div></div>
<div class="vbox"><div class="vtitle">T (°C)</div><div class="vvalue">{t}</div></div>
<div class="vbox"><div class="vtitle">Ref</div><div class="vvalue">{_ref_html(ref)}</div></div>
<div class="vbox"><div class="vtitle">SMILES</div><div class="vvalue">{smiles if smiles else "—"}</div></div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ===================== RIGHT: Calculations (use first matched record) =====================
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Calculations</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Calculations are performed on the first matched entry. Results are cached within the session.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    res = st.session_state.get("last_results", None)
    if res is None or (isinstance(res, pd.DataFrame) and len(res) == 0):
        st.markdown("<div class='card'><div class='small'>Search first to enable calculations.</div></div>", unsafe_allow_html=True)
    else:
        # first matched entry
        r0 = res.iloc[0]
        rid = int(r0["_rid"])
        smiles = (r0.get("Smiles","") or "").strip()
        name = (r0.get("Chemical compound","") or "").strip()
        cas = (r0.get("Cas","") or "").strip()
        system = st.session_state["last_system"]

        tabs = st.tabs(["2D Descriptors", "Morgan Fingerprint", "MACCS Fingerprint"])

        # ---------- 2D Descriptors ----------
        with tabs[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 2D Descriptors (11)")

            desc_sel = st.multiselect(
                "Descriptors",
                options=ALL_DESC_NAMES,
                default=ALL_DESC_NAMES,
                key=f"desc_sel_{system}_{rid}",
            )

            if st.button("Compute descriptors", key=f"btn_desc_{system}_{rid}"):
                try:
                    d = compute_descriptors(smiles, desc_sel)
                    set_cache(system, rid, "desc", {"selected": desc_sel, "values": d})
                    st.success("Descriptors computed.")
                except Exception as e:
                    st.error(str(e))

            pack = get_cache(system, rid).get("desc", None)
            if pack is not None:
                dvals = pack["values"]

                # compact two-column grid with larger text
                st.markdown("<div class='desc-grid'>", unsafe_allow_html=True)
                for k in dvals.keys():
                    v = fmt_value(dvals[k], nd=6)
                    st.markdown(
                        f"<div class='desc-box'><div class='desc-name'>{k}</div><div class='desc-val'>{v}</div></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                out_df = pd.DataFrame([{
                    "Record_ID": rid,
                    "System": system,
                    "Chemical compound": name,
                    "Cas": cas,
                    "Smiles": smiles,
                    **dvals
                }])

                st.download_button(
                    "Download descriptors CSV",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_Record{rid}_Descriptors.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_desc_{system}_{rid}",
                )
            else:
                st.caption("Compute to display descriptor values here.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Morgan ----------
        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Morgan Fingerprint")

            c1, c2 = st.columns(2)
            with c1:
                n_bits = st.number_input(
                    "nBits",
                    min_value=1,
                    max_value=8192,
                    value=32,
                    step=1,
                    key=f"m_bits_{system}_{rid}",
                )
            with c2:
                radius = st.number_input(
                    "radius",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key=f"m_rad_{system}_{rid}",
                )

            if st.button("Compute Morgan fingerprint", key=f"btn_morgan_{system}_{rid}"):
                try:
                    bits = compute_morgan_bits(smiles, int(n_bits), int(radius))
                    set_cache(system, rid, "morgan", {"nBits": int(n_bits), "radius": int(radius), "bits": bits})
                    st.success("Morgan fingerprint computed.")
                except Exception as e:
                    st.error(str(e))

            mpack = get_cache(system, rid).get("morgan", None)
            if mpack is not None:
                bits = mpack["bits"]
                info = summarize_bits(bits)
                st.markdown(
                    f"<div class='small'><b>length</b>={info['length']} &nbsp;|&nbsp; <b>on_bits</b>={info['on_bits']} &nbsp;|&nbsp; <b>on_ratio</b>={info['on_ratio']:.4f}</div>",
                    unsafe_allow_html=True
                )

                # wrapped display (no horizontal scroll)
                st.markdown(f"<div class='mono-wrap'>{','.join(map(str, bits))}</div>", unsafe_allow_html=True)

                nb = mpack["nBits"]
                rad = mpack["radius"]
                cols = [f"Morgan_{i}" for i in range(nb)]
                out_df = pd.DataFrame([[rid, system, name, cas, smiles] + bits],
                                      columns=["Record_ID","System","Chemical compound","Cas","Smiles"] + cols)

                st.download_button(
                    f"Download Morgan CSV (nBits={nb}, r={rad})",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_Record{rid}_Morgan_{nb}_r{rad}.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_morgan_{system}_{rid}",
                )
            else:
                st.caption("Compute to display Morgan bits here.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- MACCS ----------
        with tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### MACCS Fingerprint (167)")

            if st.button("Compute MACCS fingerprint", key=f"btn_maccs_{system}_{rid}"):
                try:
                    bits = compute_maccs_bits(smiles)
                    set_cache(system, rid, "maccs", {"bits": bits})
                    st.success("MACCS fingerprint computed.")
                except Exception as e:
                    st.error(str(e))

            kpack = get_cache(system, rid).get("maccs", None)
            if kpack is not None:
                bits = kpack["bits"]
                info = summarize_bits(bits)
                st.markdown(
                    f"<div class='small'><b>length</b>={info['length']} &nbsp;|&nbsp; <b>on_bits</b>={info['on_bits']} &nbsp;|&nbsp; <b>on_ratio</b>={info['on_ratio']:.4f}</div>",
                    unsafe_allow_html=True
                )

                # wrapped display (no horizontal scroll)
                st.markdown(f"<div class='mono-wrap'>{','.join(map(str, bits))}</div>", unsafe_allow_html=True)

                cols = [f"MACCS_{i}" for i in range(len(bits))]
                out_df = pd.DataFrame([[rid, system, name, cas, smiles] + bits],
                                      columns=["Record_ID","System","Chemical compound","Cas","Smiles"] + cols)

                st.download_button(
                    "Download MACCS CSV",
                    data=to_csv_bytes(out_df),
                    file_name=f"{system}_Record{rid}_MACCS.csv",
                    mime="text/csv",
                    on_click=inc_download,
                    key=f"dl_maccs_{system}_{rid}",
                )
            else:
                st.caption("Compute to display MACCS bits here.")
            st.markdown("</div>", unsafe_allow_html=True)

st.caption("© 2026 RadLogk-AI Dataset Team")