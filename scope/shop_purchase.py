"""Purchase evidence from the rendered ShopSimulator storefront."""
from ast import literal_eval
from decimal import Decimal
import json
import math
import re
from urllib.parse import parse_qs, unquote, urlsplit

from .state_graph import _BUDGET, _COLOR

_DETAILS = re.compile(r"\nShopSimulator purchase panel: ([^\n]+)\Z")
_PRICE = re.compile(r"\s*价格\s*[:：]\s*[￥¥]?(\d+(?:\.\d+)?)(?:\s*元)?\s*")
_COLOR_FIELDS = {"color", "colour", "颜色", "颜色分类"}
_EXACT_COLORS = {"黑色", "白色", "红色", "蓝色", "灰色", "绿色", "银色", "金色", "透明"}
_PRODUCT_ID = re.compile(r"\d{8,}|[Bb][A-Za-z0-9]{9}")


def purchase_panel(html):
    """Read the public buy form and the radio controls it submits.

    The upstream template preselects radios from these form options in the browser.
    No catalog annotation, evaluator data or simulator session is consulted here.
    """
    button = html.select_one("button.purchase")
    form = button.find_parent("form") if button else None
    if form is None:
        return None
    action = urlsplit(form.get("action", ""))
    route = action.path.split("/", 4)
    if len(route) == 5 and route[1] == "done":
        product_id, encoded = route[3], unquote(route[4])
    elif action.path == "/":
        # The in-process TextEnv registers its form handlers at "/" and renders
        # the same public asin and options in the query rather than the path.
        query = parse_qs(action.query)
        product_id, encoded = (query.get(name, [""])[0] for name in ("asin", "options"))
    else:
        return None
    if not _PRODUCT_ID.fullmatch(product_id):
        return None
    groups = {}
    for radio in html.select('input[type="radio"][name][value]'):
        groups.setdefault(radio["name"], []).append(radio["value"])
    try:
        options = literal_eval(encoded) if len(encoded) <= 4096 else None
    except (ValueError, SyntaxError, RecursionError):
        options = None
    confirmed = (isinstance(options, dict) and
                 all(isinstance(value, str) and value in groups.get(field, [])
                     for field, value in options.items()))
    price_element = html.select_one("#product-price")
    price = _PRICE.fullmatch(price_element.get_text(" ", strip=True)) if price_element else None
    return {"product_id": product_id.upper(), "selected": options if confirmed else None,
            "unselected_fields": sorted(set(groups) - set(options)) if confirmed else None,
            "displayed_price_cny": float(price[1]) if price else None}


def append_panel(text, panel):
    return text if panel is None else text + "\nShopSimulator purchase panel: " + json.dumps(panel, ensure_ascii=False)


def read_panel(text):
    match = _DETAILS.search(text)
    if not match:
        return None, None
    try:
        panel = json.loads(match[1])
    except ValueError:
        return None, None
    if not isinstance(panel, dict) or not _PRODUCT_ID.fullmatch(str(panel.get("product_id", ""))):
        return None, None
    selected, fields, price = (panel.get(key) for key in ("selected", "unselected_fields", "displayed_price_cny"))
    if not {"selected", "unselected_fields", "displayed_price_cny"} <= panel.keys():
        return None, None
    if (selected is not None and (not isinstance(selected, dict) or
                                   not all(isinstance(v, str) for v in selected.values()))
        or fields is not None and (not isinstance(fields, list) or not all(isinstance(f, str) for f in fields))
        or price is not None and (type(price) not in (int, float) or not math.isfinite(price) or price < 0)):
        return None, None
    return panel, match


def exact_color(field, value):
    # "颜色分类" is also used for composite variants and unrelated specifications.
    return value if field.casefold() in _COLOR_FIELDS and value in _EXACT_COLORS else None


def review_purchase(goal, panel):
    if panel is None:
        return None
    checks = []
    selected = panel["selected"]
    cap = _BUDGET.search(goal)
    if cap:
        price = panel["displayed_price_cny"]
        ceiling = float(cap[1])
        # With unresolved selections the displayed base price may change.
        fixed = price is not None and panel["unselected_fields"] == []
        status = ("confirmed" if price <= ceiling else "conflict") if fixed else "pending"
        checks.append({"constraint":"maximum_price_cny", "required":ceiling, "observed":price, "status":status})
    color = _COLOR.search(goal)
    if color:
        observed = [value for field, value in (selected or {}).items() if field.casefold() in _COLOR_FIELDS]
        # Conservative exact matches only. Descriptive variants can include multiple attributes.
        status = "confirmed" if observed == [color[1]] else "pending"
        checks.append({"constraint":"color", "required":color[1], "observed":observed, "status":status})
    checks.append({"constraint":"remaining_user_requirements", "status":"pending"})
    return {"checks":checks, **panel}


def purchase_conflict(review):
    if review:
        for check in review["checks"]:
            if check["status"] == "conflict":
                return f"当前所选规格的公开价格 {check['observed']:g} 元超过明确上限 {check['required']:g} 元，请更换规格或商品"
    return None


_FIELD_ALIASES = {
    "size": {"size", "尺码", "鞋码", "尺寸", "规格尺寸", "大小"},
    "material": {"material", "材质", "材料", "面料"},
    "pack_quantity": {"pack", "pack quantity", "quantity", "数量", "包装数量", "件数", "包装规格"},
}
_DIMENSION = r"\d+(?:\.\d+)?(?:\s*[*×x]\s*\d+(?:\.\d+)?){0,2}\s*(?:mm|cm|毫米|厘米|公分|英寸)"
_REQUIRED = r"\s*(?:为|是|要求(?:为)?|需要|要|[:：])\s*"
_FIELD_END = r"(?:的)?(?=$|[\s，。；、,;.!])"
_GOAL_SLOTS = {
    "size": re.compile(r"(?:(?:尺码|尺寸)" + _REQUIRED + r"(" + _DIMENSION
                       + r"|[2-6]?xl|x{0,3}s|m|l|x{1,3}l)|鞋码" + _REQUIRED
                       + r"(\d+(?:\.\d+)?\s*码?)(?!\.\d))" + _FIELD_END, re.I),
    "material": re.compile(r"(?:材质|材料|面料)" + _REQUIRED
                           + r"(纯棉|全棉|棉|不锈钢|硅胶|陶瓷|玻璃|塑料|铝合金|合成金属|纯羊毛|羊毛|真皮)" + _FIELD_END, re.I),
    "pack_quantity": re.compile(r"(?:包装数量\s*(?:为|是|[:：])\s*(\d+)|(\d+\s*(?:件|个|只|包|片|盒)装))"),
}


def _normalized_slot(field, value):
    normalized = re.sub(r"\s+", "", value).casefold()
    if field == "size":
        normalized = normalized.replace("厘米", "cm").replace("公分", "cm").replace("毫米", "mm")
        dimensions = re.fullmatch(r"(\d+(?:\.\d+)?(?:[*×x]\d+(?:\.\d+)?){0,2})(mm|cm|英寸)", normalized)
        if dimensions:
            factor = {"mm": Decimal(1), "cm": Decimal(10), "英寸": Decimal("25.4")}[dimensions[2]]
            return "×".join(format((Decimal(part) * factor).normalize(), "f")
                            for part in re.split("[*×x]", dimensions[1])) + "mm"
        numeric = re.fullmatch(r"(\d+(?:\.\d+)?)码?", normalized)
        if numeric:
            return format(Decimal(numeric[1]).normalize(), "f") + "码"
        return normalized if re.fullmatch(r"(?:[2-6]?xl|x{0,3}s|m|l|x{1,3}l)", normalized) else None
    if field == "material":
        normalized = {"全棉": "纯棉"}.get(normalized, normalized)
        return normalized if _GOAL_SLOTS["material"].fullmatch("材质为" + normalized) else None
    boxes = re.fullmatch(r"(\d+)盒装", normalized)
    if boxes:
        return f"{int(boxes[1])}盒"
    count = re.fullmatch(r"(\d+)(?:件装|个装|只装|包装|片装|pc|pcs)?", normalized)
    return int(count[1]) if count else None


def review_extended_purchase(goal, panel):
    """Check only explicit user fields and the options the public form submits."""
    result = review_purchase(goal, panel)
    if result is None:
        return None
    checks = result["checks"][:-1]
    selected = panel["selected"] or {}
    for field, pattern in _GOAL_SLOTS.items():
        match = pattern.search(goal)
        if not match:
            continue
        required_raw = next(group for group in match.groups() if group is not None)
        required = _normalized_slot(field, required_raw)
        actual = [_normalized_slot(field, value) for key, value in selected.items()
                  if key.strip().casefold() in _FIELD_ALIASES[field]]
        value = actual[0] if len(actual) == 1 else None
        comparable = value is not None and required is not None
        if field == "material" and required in {"棉", "羊毛"} and value not in {required, "纯" + required}:
            comparable = False  # A composition not shown by the form stays pending.
        if field == "material" and "合成金属" in {required, value} and value != required:
            comparable = bool({required, value} & {"棉", "纯棉", "硅胶", "陶瓷", "玻璃", "塑料", "羊毛", "纯羊毛", "真皮"})
        if field == "size" and comparable:
            comparable = (value.endswith("mm") == required.endswith("mm") and
                          value.endswith("码") == required.endswith("码") and
                          value.count("×") == required.count("×"))
            if comparable and value != required and "×" in value:
                comparable = sorted(value.removesuffix("mm").split("×")) != sorted(required.removesuffix("mm").split("×"))
        if field == "pack_quantity" and comparable:
            comparable = type(value) is type(required)
        matches = required == value or field == "material" and value == "纯" + str(required)
        checks.append({"constraint": field, "required": required, "observed": value,
                       "status": ("confirmed" if matches else "conflict") if comparable else "pending"})
    return {**result, "checks": [*checks, result["checks"][-1]]}


def review_for_purchase_prompt(goal, panel):
    """Expose additional fields only when the public form supplies comparable values."""
    result = review_extended_purchase(goal, panel)
    if result is None:
        return None
    return {**result, "checks": [row for row in result["checks"]
                                if row["constraint"] not in _GOAL_SLOTS or row["status"] != "pending"]}


def record_candidate_review(ledger, review, observation_id, *, continuous=False):
    """Keep revisions by product and submitted variant; repeated views share a key."""
    if review is None:
        return None
    signature = json.dumps([review["product_id"], review["selected"], review["checks"]],
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    existing = ledger.pop(signature, None)
    row = {"product_id": review["product_id"], "selected": review["selected"], "last_observation_id": observation_id,
           "visits": (existing["visits"] if continuous else existing["visits"] + 1) if existing else 1,
           "conflicts": [x["constraint"] for x in review["checks"] if x["status"] == "conflict"],
           "pending": [x["constraint"] for x in review["checks"] if x["status"] == "pending"]}
    ledger[signature] = row
    if len(ledger) > 24:
        del ledger[next(iter(ledger))]
    return row
