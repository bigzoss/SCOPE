import numpy as np

from scope.babyai_navigation import PublicLocalMap, delivery_tiles, matching_targets, ordered_mission
from scope.pddl_subgoals import avoid_repeated_inverse, relevant_state_facts
from scope.shop_purchase import record_candidate_review, review_extended_purchase, review_for_purchase_prompt, review_purchase


def test_shop_checks_submitted_fields_with_canonical_units_and_keeps_unknown_pending():
    panel = {"product_id": "123456789", "selected": {"尺寸": "95×205cm", "材质": "塑料", "数量": "2件装"},
             "unselected_fields": [], "displayed_price_cny": 12}
    goal = "尺寸为950*2050mm，材质要求纯棉，2件装。"
    review = review_extended_purchase(goal, panel)
    checks = {r["constraint"]: r["status"] for r in review["checks"]}
    assert checks == {"size": "confirmed", "material": "conflict", "pack_quantity": "confirmed",
                      "remaining_user_requirements": "pending"}
    panel["selected"] = {"尺寸": "205×95cm", "材质": "纯棉混纺"}
    checks = {r["constraint"]: r["status"] for r in review_extended_purchase(goal, panel)["checks"]}
    assert checks["size"] == checks["material"] == checks["pack_quantity"] == "pending"


def test_purchase_check_recognizes_public_shoe_size_and_does_not_equate_pack_units():
    panel = {"product_id": "123456789", "selected": {"鞋码": "40.5码", "材质": "不锈钢", "数量": "2件装"},
             "unselected_fields": [], "displayed_price_cny": 12}
    review = review_extended_purchase("鞋码需要40.5的，材质要合成金属的，2盒装。", panel)
    checks = {r["constraint"]: r["status"] for r in review["checks"]}
    assert checks == {"size": "confirmed", "material": "pending", "pack_quantity": "pending",
                      "remaining_user_requirements": "pending"}
    review = review_extended_purchase("鞋码需要40.5的，材质最好是不锈钢的。", panel)
    assert [row["constraint"] for row in review["checks"]] == ["size", "remaining_user_requirements"]
    pending = "材质要合成金属的，2盒装。"
    assert review_for_purchase_prompt(pending, panel) == review_purchase(pending, panel)
    assert review_for_purchase_prompt("鞋码需要40.5的。", panel)["checks"][0]["status"] == "confirmed"


def test_candidate_revisit_counts_only_same_variant_and_evidence():
    ledger = {}
    panel = {"product_id": "123456789", "selected": {"尺码": "L"},
             "unselected_fields": [], "displayed_price_cny": 12}
    review = review_extended_purchase("尺码为M。", panel)
    assert record_candidate_review(ledger, review, "obs_1")["visits"] == 1
    assert record_candidate_review(ledger, review, "obs_2", continuous=True)["visits"] == 1
    assert record_candidate_review(ledger, review, "obs_3")["visits"] == 2
    assert len(ledger) == 1


def test_babyai_map_routes_from_public_known_tiles_only():
    navigation = PublicLocalMap()
    navigation.tiles = {(1, 0): ("empty", "red", 0), (1, 1): ("key", "blue", 0)}
    assert navigation.next_steps(matching_targets(navigation.tiles, "get the blue key")) == (
        ("forward", "right"), True)
    image = np.zeros((3, 3, 3), dtype=int)
    image[1, 1] = (1, 1, 0)
    visible = navigation.project(image, 0, (0, 0), {0: "unseen", 1: "key"}, {0: "red", 1: "blue"})
    assert visible == {(1, 0): ("key", "blue", 0)}


def test_pddl_suppresses_only_repeated_reverse_without_an_observed_purpose():
    loop = {"repeated_inverse": True, "advances": set(), "unblocks": set()}
    unlock = {"repeated_inverse": True, "advances": set(), "unblocks": {"clear b1"}}
    alternative = {"repeated_inverse": False, "advances": set(), "unblocks": set()}
    assert avoid_repeated_inverse([loop, unlock, alternative]) == [unlock, alternative]
    assert avoid_repeated_inverse([loop]) == [loop]


def test_babyai_keeps_stage_order_and_requires_observed_delivery_space():
    stages = ordered_mission("put the red ball next to the blue box after you open the green door")
    assert stages == [("toggle", "the green door", None), ("deliver", "the red ball", "the blue box")]
    assert ordered_mission("open the green door and then pick up the red ball")[-1][0] == "pickup"
    tiles = {(1, 1): ("box", "blue", 0), (1, 2): ("empty", "red", 0), (0, 1): ("wall", "grey", 0)}
    assert delivery_tiles(tiles, {(1, 1)}) == {(1, 2)}


def test_pddl_compact_state_preserves_unrelated_blocks_that_can_support_the_goal():
    from collections import namedtuple
    Predicate = namedtuple("Predicate", "name")
    Literal = namedtuple("Literal", "predicate variables")
    def lit(name, *args):
        return Literal(Predicate(name), args)
    support = {lit("holding", "c"), lit("clear", "d"), lit("on", "d", "e"), lit("ontable", "e")}
    irrelevant = lit("object-type", "irrelevant")
    assert relevant_state_facts(support | {irrelevant}, {lit("on", "a", "b")}) == support
