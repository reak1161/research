from graphviz import Digraph

def render_choice_tree(candidates, chosen_index, out_path="choice_tree", fmt="png"):
    """
    candidates: list[dict] 例:
      [{"id":"Q12","title":"lsのオプション","p":0.72,"domain":"ファイル管理","tier":2}, ...]
    chosen_index: int
    """
    g = Digraph("choice", format=fmt)
    g.attr(rankdir="TB", nodesep="0.35", ranksep="0.5")

    # root
    g.node("root", "LLM selection\n(choose 1)", shape="box", style="rounded,filled",
           fillcolor="#f2f2f2")

    # candidates
    for i, c in enumerate(candidates):
        label = (
            f'[{i}] {c.get("id","")}\n'
            f'{c.get("title","")}\n'
            f'P(correct)={c.get("p", None):.2f}  |  {c.get("domain","")}, tier {c.get("tier","")}'
        )

        is_chosen = (i == chosen_index)
        g.node(
            f"c{i}",
            label,
            shape="box",
            style="rounded,filled",
            fillcolor="#b6f0c1" if is_chosen else "#ffffff",
            penwidth="2" if is_chosen else "1",
        )
        g.edge("root", f"c{i}")

    # render
    g.render(out_path, cleanup=True)
    return f"{out_path}.{fmt}"

# ---- example ----
candidates = [
    {"id":"Q12","title":"cp と mv の違い", "p":0.68, "domain":"ファイル操作", "tier":2},
    {"id":"Q33","title":"標準入力とパイプ", "p":0.52, "domain":"入出力", "tier":3},
    {"id":"Q07","title":"chmod の数値表記", "p":0.30, "domain":"権限", "tier":2},
]
print(render_choice_tree(candidates, chosen_index=1, out_path="choice_tree"))
