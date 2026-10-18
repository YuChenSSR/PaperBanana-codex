"""Final-image and evolution scans must see critic rounds past the old caps."""

from skill.run import extract_final_image_b64
from visualize.show_pipeline_evolution import collect_evolution_stages


def test_skill_final_image_prefers_round_past_the_old_cap():
    result = {
        "max_critic_rounds": 5,
        "target_diagram_critic_desc1_base64_jpg": "round-1",
        "target_diagram_critic_desc4_base64_jpg": "round-4",
        "target_diagram_desc0_base64_jpg": "planner",
    }

    assert extract_final_image_b64(result, "demo_planner_critic") == "round-4"


def test_skill_final_image_uses_available_round_without_configured_max():
    result = {
        "target_diagram_critic_desc4_base64_jpg": "round-4",
        "target_diagram_desc0_base64_jpg": "planner",
    }

    assert extract_final_image_b64(result, "demo_planner_critic") == "round-4"


def test_skill_final_image_still_falls_back_when_no_critic_image():
    result = {"target_diagram_stylist_desc0_base64_jpg": "stylist"}

    assert extract_final_image_b64(result, "demo_full") == "stylist"


def test_evolution_view_includes_critic_rounds_past_two():
    item = {
        "max_critic_rounds": "5",
        "target_diagram_desc0": "planned",
        "target_diagram_critic_desc0": "round-0",
        "target_diagram_critic_desc2": "round-2",
        "target_diagram_critic_desc4": "round-4",
    }

    rounds = [
        stage["round_idx"]
        for stage in collect_evolution_stages(item)
        if "round_idx" in stage
    ]

    assert rounds == [0, 2, 4]
    titles = [stage["title"] for stage in collect_evolution_stages(item)]
    assert titles[2] == "🔍 Critic Round 0"
    assert "🔍🔍🔍 Critic Round 2" in titles
    assert "🔍🔍🔍🔍🔍 Critic Round 4" in titles
