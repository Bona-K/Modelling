# src/simulate.py
"""
Threshold Rule ABM (MVP)
- 모든 에이전트: t=0에 앵커에서 시작
- 규칙:
  1) 현재 이웃 중 '동종 비율' >= theta_same 이면, 동종 이웃으로 이동
  2) 아니면 확률 p_move_below로 랜덤 이웃 이동(보완 우선), 아니면 stay
  3) 위치가 앵커면 확률 p_stay_anchor로 stay
- 방문/구매 집계, 최종 이탈 노드 기록
"""
import argparse, json, random
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx

from src.model import Agent
from src.graph import build_anchor_center_grid, node_link_dump, node_link_load
from utils.metrics import compute_hhi, compute_gini, hop_dist_from_anchor, spillover_curve

# ---------- 내부 유틸 ----------

def pick_same_or_random(
    G: nx.Graph,
    current: int,
    agent_cat: str,
    theta_same: float,
    p_move_below: float,
    anti_backtrack: bool,
    prev_node: int,
    complements: Dict[str, List[str]]
) -> int:
    """임계값 기반 이동 대상 노드 선택."""
    neigh = list(G.neighbors(current))
    if not neigh:
        return current

    # 동종/타종 분리
    same = [n for n in neigh if G.nodes[n]["category"] == agent_cat]
    other = [n for n in neigh if G.nodes[n]["category"] != agent_cat]

    ratio_same = len(same) / len(neigh)

    # 1) 임계 초과 → 동종 중 랜덤
    if same and ratio_same >= theta_same:
        candidates = same
    else:
        # 2) 임계 미만 → p_move_below 확률로 랜덤 이동
        if random.random() < p_move_below and neigh:
            # 보완 우선(있으면)
            comp_targets = []
            for n in other:
                if G.nodes[n]["category"] in complements.get(agent_cat, []):
                    comp_targets.append(n)
            candidates = comp_targets if comp_targets else neigh
        else:
            # stay
            return current

    # 되돌림 억제
    if anti_backtrack and prev_node in candidates and len(candidates) > 1:
        candidates = [c for c in candidates if c != prev_node]

    return random.choice(candidates) if candidates else current


def maybe_purchase(node: int, G: nx.Graph, purchase_prob: Dict[str, float]) -> bool:
    cat = G.nodes[node]["category"]
    p = purchase_prob.get(cat, 0.1)
    return random.random() < p


# ---------- 시뮬레이터 ----------

def run_one_simulation(
    G: nx.Graph,
    params: Dict,
    rng_seed: int = 0
) -> Dict:
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    theta_same = float(params["theta_same"])
    p_move_below = float(params["p_move_below"])
    p_stay_anchor = float(params["p_stay_anchor"])
    anti_backtrack = bool(params.get("anti_backtrack", True))
    purchase_prob = params["purchase_prob"]

    dwell_min = int(params["dwell_ticks"]["min"])
    dwell_max = int(params["dwell_ticks"]["max"])
    n_agents = int(params["agents"])
    ticks = int(params["ticks"])

    # anchor 찾기
    anchor_id = None
    for n, d in G.nodes(data=True):
        if d.get("is_anchor", False):
            anchor_id = n
            break
    if anchor_id is None:
        raise ValueError("Anchor node not found (node attr is_anchor=True).")

    # 집계 컨테이너
    visits = {n: 0 for n in G.nodes()}
    sales = {n: 0 for n in G.nodes()}
    exit_nodes: List[int] = []

    # hop 거리 사전(스필오버 분석용)
    hopdist = hop_dist_from_anchor(G, anchor_id)

    # 에이전트 초기화
    agents = []
    for i in range(n_agents):
        dwell = random.randint(dwell_min, dwell_max)
        a = Agent(id=i, dwell_left=dwell, position=anchor_id, prev_position=None)
        agents.append(a)

    # tick 루프
    for t in range(ticks):
        for a in agents:
            if a.dwell_left <= 0:
                continue

            cur = a.position
            # 앵커에서 머무름 보너스
            if cur == anchor_id and random.random() < p_stay_anchor:
                pass  # stay
            else:
                agent_cat = G.nodes[cur]["category"]
                nxt = pick_same_or_random(
                    G, cur, agent_cat, theta_same, p_move_below,
                    anti_backtrack, a.prev_position,
                    params.get("complements", {})
                )
                a.prev_position = cur
                a.position = nxt
                cur = nxt

            # 방문/구매 집계
            visits[cur] += 1
            if maybe_purchase(cur, G, purchase_prob):
                sales[cur] += 1

            a.dwell_left -= 1

        # 현재 tick 끝난 뒤, dwell_left가 0이 된 에이전트 exit 노드 기록
        for a in agents:
            if a.dwell_left == 0:
                exit_nodes.append(a.position)

    # 지표
    hhi_sales = compute_hhi(sales)
    gini_sales = compute_gini(list(sales.values()))
    spill = spillover_curve(visits, hopdist)

    return dict(
        visits=visits,
        sales=sales,
        exit_nodes=exit_nodes,
        hop_spillover=spill,
        hhi_sales=hhi_sales,
        gini_sales=gini_sales
    )


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="data/params.json")
    ap.add_argument("--layout", default="data/layout.json")
    ap.add_argument("--generate-layout", action="store_true",
                    help="Create a 5x5 anchor-centered grid to data/layout.json")
    ap.add_argument("--side", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 파라미터
    with open(args.params, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 레이아웃 준비
    if args.generate_layout:
        G = build_anchor_center_grid(side=args.side, categories=params["categories"], seed=42)
        node_link_dump(G, args.layout)
        print(f"[OK] Generated layout -> {args.layout}")
    else:
        G = node_link_load(args.layout)

    # 반복 실행(씨드)
    all_runs = []
    for s in range(args.seeds):
        res = run_one_simulation(G, params, rng_seed=s)
        all_runs.append(res)

    # 간단 요약 출력
    print(f"[DONE] runs={len(all_runs)}  "
          f"mean HHI={np.mean([r['hhi_sales'] for r in all_runs]):.4f}  "
          f"mean Gini={np.mean([r['gini_sales'] for r in all_runs]):.4f}")

    # 필요 시 CSV/JSON 저장
    if args.out:
        import pandas as pd, os, json as _json
        os.makedirs(args.out, exist_ok=True)
        # 노드별 집계 저장(평균)
        nodes = list(G.nodes())
        def avg_dict(key):
            d = {n: np.mean([r[key][n] for r in all_runs]) for n in nodes}
            return pd.DataFrame({"node": list(d.keys()), key: list(d.values())})

        avg_visits = avg_dict("visits"); avg_sales = avg_dict("sales")
        avg_visits.to_csv(f"{args.out}/visits.csv", index=False)
        avg_sales.to_csv(f"{args.out}/sales.csv", index=False)
        # 스필오버 저장(첫 run)
        spill = all_runs[0]["hop_spillover"]
        pd.DataFrame({"hop": list(spill.keys()), "visits": list(spill.values())}) \
          .to_csv(f"{args.out}/spillover.csv", index=False)
        # 메타 저장
        with open(f"{args.out}/summary.json","w",encoding="utf-8") as f:
            _json.dump({
                "hhi_sales_mean": float(np.mean([r["hhi_sales"] for r in all_runs])),
                "gini_sales_mean": float(np.mean([r["gini_sales"] for r in all_runs]))
            }, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] results -> {args.out}")

if __name__ == "__main__":
    main()
