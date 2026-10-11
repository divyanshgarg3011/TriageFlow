from app import tools


def test_customer_lookup():
    c = tools.get_customer("Aisha.Khan@example.com")
    assert c["found"] and c["tier"] == "gold" and c["first_name"] == "Aisha"
    assert tools.get_customer("nobody@example.com")["found"] is False


def test_order_has_computed_days():
    o = tools.get_order("ng-5530")
    assert o["found"] and o["days_since_delivery"] == 10


def test_change_of_mind_inside_window():
    r = tools.check_refund_eligibility("NG-5530", "change_of_mind")
    assert r["eligible"] and r["max_refund"] == 89.0 and r["return_shipping_fee"] == 0  # gold member


def test_change_of_mind_outside_window_standard_member():
    r = tools.check_refund_eligibility("NG-5539", "change_of_mind")
    assert not r["eligible"] and "store_credit_allowed" not in r


def test_change_of_mind_outside_window_gold_gets_store_credit():
    r = tools.check_refund_eligibility("NG-5571", "change_of_mind")
    assert not r["eligible"] and r["store_credit_allowed"] == 12.0


def test_defective_large_amount_needs_manager():
    r = tools.check_refund_eligibility("NG-5544", "defective")
    assert r["eligible"] and r["needs_manager_approval"] is True


def test_lost_parcel_and_delivered_parcel():
    assert tools.check_refund_eligibility("NG-5521", "not_delivered")["eligible"]
    assert not tools.check_refund_eligibility("NG-5560", "not_delivered")["eligible"]


def test_duplicate_charge():
    assert tools.check_refund_eligibility("NG-5560", "duplicate_charge")["max_refund"] == 149.0
    assert not tools.check_refund_eligibility("NG-5544", "duplicate_charge")["eligible"]


def test_kb_search_finds_the_right_policy():
    top = tools.search_knowledge_base("parcel lost tracking not updated")["results"][0]
    assert top["source"].startswith("shipping_and_delivery")
    top = tools.search_knowledge_base("power bank overheating")["results"][0]
    assert "overheating" in top["heading"].lower()
