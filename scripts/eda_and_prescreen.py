"""22종 데이터셋에 대해 기초 EDA와 5.4절(변형 가능성 사전 선별) 카운트를 함께 수행한다.

순서는 연구계획서 5.7절을 따른다: 파싱 검증 -> 부모 분자 확정(원본 보존) ->
정준화 및 중복 확인 -> 골격 계산. 여기까지는 아직 분할이 아니라 데이터셋
전체에 대한 점검이므로 train_val/test를 합쳐서 봐도 분할 규율을 어기지 않는다.
실제 학습/시험 분할과 변형 생성은 이 스크립트의 범위가 아니다.

호변이성질체 열거와 양성자화 상태 열거는 분자당 시간이 걸리므로, 데이터셋이
큰 경우(1만 개 이상) 전체 실행에 수 분에서 수십 분이 걸릴 수 있다. 결과는
캐시하여 재계산을 방지한다.
"""

import json
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 열거 폭발을 막기 위한 상한 (7.2절: 분자당 상한을 반드시 설정)
MAX_TAUTOMERS = 20
MAX_PROTOMERS = 20

_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_tautomer_enumerator.SetMaxTautomers(MAX_TAUTOMERS)
# canSARchem_RDKit(Journal/canSAR chemistry registration and standardization
# pipeline.pdf)이 정준 대표 구조를 생성할 때 쓰는 설정과 동일하게 맞춘다.
# 이 단계에서 입체 정보를 지우지 않아야 B3(입체 표기) 축과 간섭하지 않는다.
_tautomer_enumerator.SetRemoveSp3Stereo(False)
_tautomer_enumerator.SetRemoveBondStereo(False)
_uncharger = rdMolStandardize.Uncharger()

try:
    from dimorphite_dl import protonate_smiles
    HAS_DIMORPHITE = True
except ImportError:
    HAS_DIMORPHITE = False


def load_dataset(name):
    """train_val과 test를 합쳐 데이터셋 하나로 반환한다. EDA/사전 선별 전용."""
    d = RAW_DIR / name
    tv = pd.read_csv(d / "train_val.csv")
    te = pd.read_csv(d / "test.csv")
    tv["_split"] = "train_val"
    te["_split"] = "test"
    return pd.concat([tv, te], ignore_index=True)


def has_salt(smiles):
    """SMILES에 조각(fragment)이 둘 이상이면 염/혼합물로 간주한다."""
    return "." in smiles


def get_parent(mol):
    """가장 큰 조각을 부모 분자로 취급한다 (원본은 보존, 별도 계산)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def has_stereo_spec(smiles):
    """SMILES 표기에 입체 정보가 명시되어 있는지 여부."""
    return "@" in smiles or "/" in smiles or "\\" in smiles


def count_tautomers(mol):
    try:
        enumerated = _tautomer_enumerator.Enumerate(mol)
        return min(len(enumerated), MAX_TAUTOMERS)
    except Exception:
        return None


def get_canonical_tautomer_smiles(mol):
    """대표(정준) 호변이성질체 하나를 정한다.

    canSARchem_RDKit과 동일하게 rdMolStandardize.TautomerEnumerator.Canonicalize를
    쓴다 (Sitzmann et al. 2010의 스코어링 규칙, RDKit 문서에 명시).
    주의: 이 함수는 "가장 안정적인" 형태를 보장하지 않는다. 입력 순서와 무관하게
    항상 같은 결과를 내는 정준화가 목적이며, 6.2절 분산 계산에는 쓰지 않는다.
    5.7절 중복 제거의 부모 분자 키, 7.3절 분자 카드의 기준점으로만 사용한다.
    """
    try:
        canonical = _tautomer_enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(canonical)
    except Exception:
        return None


def count_protomers(smiles):
    if not HAS_DIMORPHITE:
        return None
    try:
        results = protonate_smiles(smiles, ph_min=6.4, ph_max=8.4, precision=1.0)
        return min(len(set(results)), MAX_PROTOMERS)
    except Exception:
        return None


def profile_dataset(name, sample_n=None):
    df = load_dataset(name)
    if sample_n is not None:
        df = df.sample(n=min(sample_n, len(df)), random_state=0).reset_index(drop=True)

    n_total = len(df)
    n_dup_smiles_raw = n_total - df["Drug"].nunique()

    rows = []
    n_invalid = 0
    for smi in df["Drug"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            rows.append({
                "smiles": smi, "valid": False, "has_salt": None,
                "parent_smiles": None, "scaffold": None, "has_stereo": None,
                "n_tautomers": None, "n_protomers": None,
                "canonical_tautomer_smiles": None,
            })
            continue

        salt = has_salt(smi)
        parent_mol = get_parent(mol) if salt else mol
        parent_smi = Chem.MolToSmiles(parent_mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent_mol) if parent_mol else None
        stereo = has_stereo_spec(smi)
        n_taut = count_tautomers(parent_mol)
        n_proto = count_protomers(parent_smi)
        # 5.7절 중복 제거 키: 염만 뗀 parent_smiles는 호변이성질체가 다르면
        # 같은 화합물도 다른 문자열로 남는다. canSAR 순서(표준화 -> 정준
        # 호변이성질체 생성 -> 염 제거)를 참고해 정준 호변이성질체까지 반영한
        # 키를 별도로 둔다.
        canonical_taut_smi = get_canonical_tautomer_smiles(parent_mol)

        rows.append({
            "smiles": smi, "valid": True, "has_salt": salt,
            "parent_smiles": parent_smi, "scaffold": scaffold, "has_stereo": stereo,
            "n_tautomers": n_taut, "n_protomers": n_proto,
            "canonical_tautomer_smiles": canonical_taut_smi,
        })

    detail = pd.DataFrame(rows)
    valid = detail[detail["valid"]]

    n_parent_dup = 0
    n_parent_dup_taut_aware = 0
    if len(valid) > 0:
        n_parent_dup = len(valid) - valid["parent_smiles"].nunique()
        n_parent_dup_taut_aware = len(valid) - valid["canonical_tautomer_smiles"].nunique()

    summary = {
        "dataset": name,
        "n_total": n_total,
        "n_invalid_smiles": n_invalid,
        "n_dup_raw_smiles": int(n_dup_smiles_raw),
        "n_dup_parent_molecules": int(n_parent_dup),
        "n_dup_parent_molecules_tautomer_aware": int(n_parent_dup_taut_aware),
        "n_with_salt": int(valid["has_salt"].sum()) if len(valid) else 0,
        "pct_with_salt": round(100 * valid["has_salt"].mean(), 2) if len(valid) else None,
        "n_with_stereo": int(valid["has_stereo"].sum()) if len(valid) else 0,
        "pct_with_stereo": round(100 * valid["has_stereo"].mean(), 2) if len(valid) else None,
        "n_multi_tautomer": int((valid["n_tautomers"] >= 2).sum()) if len(valid) else 0,
        "pct_multi_tautomer": round(100 * (valid["n_tautomers"] >= 2).mean(), 2) if len(valid) else None,
        "n_multi_protomer": int((valid["n_protomers"] >= 2).sum()) if HAS_DIMORPHITE and len(valid) else None,
        "pct_multi_protomer": round(100 * (valid["n_protomers"] >= 2).mean(), 2) if HAS_DIMORPHITE and len(valid) else None,
        "n_unique_scaffolds": int(valid["scaffold"].nunique()) if len(valid) else 0,
    }

    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "molecule_profile.csv", index=False)
    return summary


def main():
    names = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir() and p.name not in ("_tdc_cache", "esol_pilot"))
    summaries = []
    for name in names:
        t0 = time.time()
        s = profile_dataset(name)
        s["seconds"] = round(time.time() - t0, 1)
        summaries.append(s)
        print(f"[{s['seconds']:>6.1f}s] {name:35s} "
              f"n={s['n_total']:>6d}  invalid={s['n_invalid_smiles']:>3d}  "
              f"salt={s['pct_with_salt']}%  stereo={s['pct_with_stereo']}%  "
              f"multi-taut={s['pct_multi_tautomer']}%  "
              f"multi-proto={s['pct_multi_protomer']}%")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "prescreen_summary.csv", index=False)
    summary_df.to_json(OUT_DIR / "prescreen_summary.json", orient="records", indent=2)
    print(f"\n요약 저장: {OUT_DIR / 'prescreen_summary.csv'}")


if __name__ == "__main__":
    main()
