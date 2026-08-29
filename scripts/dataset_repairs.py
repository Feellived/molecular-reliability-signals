"""담당1 분할 파일에 남아 있는 알려진 SMILES 오류를 적재 시점에 교정한다.

담당1의 splits.csv에는 AqSolDB에서 온 구형 N-oxide 표기 두 건이 파싱되지
않는 상태로 남아 있다. 담당2가 자신의 사본에서만 교정했기 때문에 공용
분할 파일에는 아직 반영되지 않았다.

교정값과 InChIKey 근거는 담당2의 scripts/repair_solubility_n_oxide.py에서
가져왔다. 여기서는 원본 파일을 고치지 않고 적재할 때만 바꾼다. 담당1이
공용 파일을 갱신하면 아래 검사에서 원문이 더 이상 일치하지 않으므로
이 교정은 자동으로 무동작이 된다.

두 행은 모두 train 분할이라 meta·test에서 만든 변형에는 영향이 없다.
"""

from __future__ import annotations

import pandas as pd

KNOWN_SMILES_REPAIRS = {
    "solubility_aqsoldb__9144": {
        "expected_source": "CC1=CC=C[NH+2]([O-])[CH-]1",
        "parent_smiles": "Cc1ccc[n+]([O-])c1",
        "inchi_key": "DMGGLIWGZFZLIY-UHFFFAOYSA-N",
    },
    "solubility_aqsoldb__9145": {
        "expected_source": "O=C(O)C1=C[NH+2]([O-])[CH-]C=C1",
        "parent_smiles": "O=C(O)c1ccc[n+]([O-])c1",
        "inchi_key": "FJCFFCXMEXZEIM-UHFFFAOYSA-N",
    },
}


def apply_known_repairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """parent_smiles의 알려진 오류를 교정하고 실제로 바꾼 row_uid를 돌려준다.

    원문이 예상과 다르면 건드리지 않는다. 공용 파일이 이미 갱신됐거나
    다른 값으로 바뀐 경우 조용히 덮어쓰지 않기 위해서다.
    """
    if "row_uid" not in frame.columns or "parent_smiles" not in frame.columns:
        return frame, []

    repaired: list[str] = []
    positions = {uid: index for index, uid in enumerate(frame["row_uid"])}
    for uid, repair in KNOWN_SMILES_REPAIRS.items():
        index = positions.get(uid)
        if index is None:
            continue
        if frame.iat[index, frame.columns.get_loc("parent_smiles")] != repair["expected_source"]:
            continue
        frame.iat[index, frame.columns.get_loc("parent_smiles")] = repair["parent_smiles"]
        repaired.append(uid)
    return frame, repaired
