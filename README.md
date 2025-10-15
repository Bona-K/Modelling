# Anchor Store Modeling

We represent the shopping mall as a network (nodes = stores, edges = adjacency) and simulate footfall based on agent (customer) movement.

- **Step 0:** Random neighbor moves (baseline)
- **Step 1:** Same-category threshold (homophily) + anchor dwell probability (anchor quality) + quality-weighted fallback
- **Step 2:** Complementarity bias (`comp_bias`) to steer spillovers and cross-category flow
- **Step 3:** Performance frame with revenue outputs + summary KPIs for scenario comparison
- **Step 4:** Crowd interaction via Boids-style dynamics + affinity (“love”) term + diagnostic metrics

## Quickstart & Run
```bash
git clone https://github.com/Bona-K/Modelling.git
cd Modelling
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
Run these notebooks in order:
1. notebooks/step0_setup.ipynb → Run All → verify total footfall & saved CSV/figs
2. notebooks/step1_ThresholdRule.ipynb (tune theta_same, p_stay_anchor)
3. notebooks/step2_ComplementaryMovement.ipynb (tune comp_bias, etc.)
4. notebooks/step3_sales_model_comparison.ipynb (revenue & KPIs)
5. notebooks/step4_crowd_dynamics.ipynb (Boids + affinity + diagnostics)
Outputs: CSV under data/outputs/, figures under data/outputs/figs/ (or fig/).

##Repository Structure
├─ src/                      # Step-by-step simulation code used to run the models
│  └─ asm/                   # Simulator modules (core logic)
│     ├─ simulate_step0.py
│     ├─ simulate_step1.py
│     ├─ simulate_step2.py
│     ├─ simulate_step3.py
│     └─ simulate_step4.py
├─ notebooks/                # Jupyter notebooks per step; runnable analyses and results
│  ├─ step0_setup.ipynb
│  ├─ step1_ThresholdRule.ipynb
│  ├─ step2_ComplementaryMovement.ipynb
│  ├─ step3_sales_model_comparison.ipynb     # avoid spaces/& in filenames
│  └─ step4_crowd_dynamics.ipynb             # or step4_visible.ipynb if that’s your name
├─ utils/                    # Shared utilities (I/O, styling, seeding, etc.)
│  ├─ __init__.py
│  ├─ io.py
│  ├─ random_tools.py
│  └─ plotting.py
├─ data/
│  └─ outputs/               # Generated CSV/figs (recommend gitignore)
├─ fig/                      # Exported figures/images from notebooks
└─ report/                   # Final reports (e.g., .docx/.pdf)
