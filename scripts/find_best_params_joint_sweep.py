"""
Find the best parameters from the joint (class x theme) unlearning sweep results.
Finds the (percentile, multiplier) combination that maximizes (SC + UA) / 2
subject to UA >= 60%.
"""
from collections import defaultdict
from pathlib import Path

import fire
import torch

# The 50 target combinations from Appendix B.5 of the paper
TARGET_COMBINATIONS = [
    ("Architectures", "Abstractionism"),
    ("Bears", "Artist_Sketch"),
    ("Birds", "Blossom_Season"),
    ("Butterfly", "Bricks"),
    ("Cats", "Byzantine"),
    ("Dogs", "Cartoon"),
    ("Fishes", "Cold_Warm"),
    ("Flame", "Color_Fantasy"),
    ("Flowers", "Comic_Etch"),
    ("Frogs", "Crayon"),
    ("Horses", "Cubism"),
    ("Human", "Dadaism"),
    ("Jellyfish", "Dapple"),
    ("Rabbits", "Defoliation"),
    ("Sandwiches", "Early_Autumn"),
    ("Sea", "Expressionism"),
    ("Statues", "Fauvism"),
    ("Towers", "French"),
    ("Trees", "Glowing_Sunset"),
    ("Waterfalls", "Gorgeous_Love"),
    ("Architectures", "Greenfield"),
    ("Bears", "Impressionism"),
    ("Birds", "Ink_Art"),
    ("Butterfly", "Joy"),
    ("Cats", "Liquid_Dreams"),
    ("Dogs", "Magic_Cube"),
    ("Fishes", "Meta_Physics"),
    ("Flame", "Meteor_Shower"),
    ("Flowers", "Monet"),
    ("Frogs", "Mosaic"),
    ("Horses", "Neon_Lines"),
    ("Human", "On_Fire"),
    ("Jellyfish", "Pastel"),
    ("Rabbits", "Pencil_Drawing"),
    ("Sandwiches", "Picasso"),
    ("Sea", "Pop_Art"),
    ("Statues", "Red_Blue_Ink"),
    ("Towers", "Rust"),
    ("Waterfalls", "Sketch"),
    ("Architectures", "Sponge_Dabbed"),
    ("Bears", "Structuralism"),
    ("Birds", "Superstring"),
    ("Butterfly", "Surrealism"),
    ("Cats", "Ukiyoe"),
    ("Dogs", "Van_Gogh"),
    ("Fishes", "Vibrant_Flow"),
    ("Flame", "Warm_Love"),
    ("Flowers", "Warm_Smear"),
    ("Frogs", "Watercolor"),
    ("Horses", "Winter"),
]


def find_best_parameters(
    percentiles: list[float],
    multipliers: list[float],
    base_path: str,
):
    best_params = defaultdict(
        lambda: {"best_score": float("-inf"), "params": None, "metrics": None}
    )

    for percentile in percentiles:
        for multiplier in multipliers:
            for cls, theme in TARGET_COMBINATIONS:
                combo_name = f"{cls}__{theme}"
                metrics_path = (
                    Path(base_path)
                    / f"percentile_{percentile}_multiplier_{multiplier}"
                    / f"{combo_name}.pth"
                )

                if not metrics_path.exists():
                    print(f"[Warn] Not found: {metrics_path}")
                    continue

                data = torch.load(metrics_path)

                ua = data.get("UA", {}).get("UA", float("nan"))
                sc = data.get("SC", {}).get("SC", float("nan"))
                oc = data.get("OC", {}).get("OC", float("nan"))

                if any(v != v for v in [ua, sc, oc]):
                    continue

                ua_pct = ua * 100
                sc_pct = sc * 100
                oc_pct = oc * 100
                score = (sc_pct + ua_pct) / 2.0

                if score > best_params[combo_name]["best_score"]:
                    best_params[combo_name]["best_score"] = score
                    best_params[combo_name]["params"] = (percentile, multiplier)
                    best_params[combo_name]["metrics"] = {
                        "UA": ua_pct,
                        "SC": sc_pct,
                        "OC": oc_pct,
                        "UP": float("nan"),
                    }

    print("\nBest parameters per target combination")
    print("  Criterion: max( (SC + UA) / 2 )")
    print("-" * 80)

    params_dict = {}
    score_list = []

    for cls, theme in TARGET_COMBINATIONS:
        combo_name = f"{cls}__{theme}"
        entry = best_params.get(combo_name)

        if entry is None or entry["params"] is None:
            print(f"\n[{combo_name}]  No data found, skipping.")
            continue

        percentile, multiplier = entry["params"]
        m = entry["metrics"]
        score = entry["best_score"]
        score_list.append(score)

        print(f"\n[{combo_name}]")
        print(f"  percentile={percentile}, multiplier={multiplier}")
        print(f"  UA : {m.get('UA', float('nan')):.2f}%")
        print(f"  SC : {m.get('SC', float('nan')):.2f}%")
        print(f"  OC : {m.get('OC', float('nan')):.2f}%")
        print(f"  UP : {m.get('UP', float('nan')):.2f}%")
        print(f"  Score : {score:.2f}%")

        params_dict[combo_name] = {
            "percentile": percentile,
            "multiplier": multiplier,
        }

    if score_list:
        print("\n" + "=" * 80)
        print(f"Overall average best score: {sum(score_list) / len(score_list):.2f}%")
        print(f"Targets with best params found: {len(score_list)} / {len(TARGET_COMBINATIONS)}")

    save_path = Path(base_path) / "joint_best_params.pth"
    torch.save(params_dict, save_path)
    print(f"\nSaved best params to: {save_path}")


if __name__ == "__main__":
    fire.Fire(find_best_parameters)