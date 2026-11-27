"""The experience on Roblox's side, through Open Cloud with the person's own key: its settings, its Creator Store listings.

    rbx_universe_settings   read an experience's (universe's) and a place's settings; a change is built, scope-checked and
                            shown as the exact request (dry run) and sent only with confirm, dry_run false and full mode
    rbx_store_list          list Models on the Creator Store at a price: the same gate; reads of a product are free
    rbx_save_to_roblox      a Model in the open Studio saved to Roblox as a normal, non-package Model with no dialog
                            (AssetService:CreateAssetAsync from the plugin, IsPackage false, RB2006): the same gate, then
                            the new id read back (the Assets API's type, and the id loaded and compared with the source)
    rbx_plugin_upload       a Folder or script saved as a Plugin asset the same way (Enum.AssetType.Plugin)
    preflight()            what rbx_place_publish checks before a publish (and alone with preflight true): the self-test
                            flag off, config ids set, no key-shaped string in a script, a save layer that never falls
                            back to memory and passes granted only on UserOwnsGamePassAsync (save_audit.py,
                            rbx_save_audit), paid random items, devices, the game page;
                            it reads and never sets
    rbx_game_pass           list, read and create the experience's game passes; a create is never for sale, has no price
                            and no image, is a dry run by default, and is guarded so it is sent once
    rbx_store_art_check     the experience's icon and thumbnails checked on disk (size, RGB, count), a review sheet
                            written, and each claim's config value printed beside it (rbx_module_values)

WHY. Settings a game's publish needs (devices, voice, max players) live in Creator Hub pages, and the Open Cloud calls that
set them need scopes the key often lacks: a key with asset scopes reads a place, yet its PATCH answers 403 "Scope not
authorized.". So every write first asks the key what it may do (POST /api-keys/v1/introspect,
RB1307) and names the missing scope before anything is sent. Selling a model: only a Studio-saved Model can be listed
(an Open Cloud Model upload is a package, RB1111, RB1187); the products API answers 503 "being processed" for a fresh save
and then 200 (RB1186); PATCH ignores price and published (RB1102), so a listing is a POST and its read-back a GET.
A Studio save no longer needs Studio's dialog: CreateAssetAsync from a local plugin saves a Model with IsPackage false
and no prompt once Studio's CreateAssetAsync beta is on (RB2006; "not available yet" before, RB1306, RB1311). IsPackage
must be false on every call: a version made without it turns a listed Model into a package for good (RB2008), so the save
tool never makes a version - a changed model is saved as a new id.

Endpoints and scopes are Roblox's own (create.roblox.com/the internal notes.json, read 2026-11-20):
  GET   /cloud/v2/universes/{u}                    no scope listed
  PATCH /cloud/v2/universes/{u}?updateMask=        universe:write
  GET   /cloud/v2/universes/{u}/places/{p}         no scope listed
  PATCH /cloud/v2/universes/{u}/places/{p}?updateMask=   universe.place:write
  POST  /cloud/v2/creator-store-products           creator-store-product:write (BETA, 30 a minute)
  GET   /cloud/v2/creator-store-products/{id}      creator-store-product:read
A place's universe comes from apis.roblox.com/universes/v1/places/{placeId}/universe, a public read (the key is never sent).
Measured (RB1653): both GETs answer a key with no universe scope, and that key's introspect lists universe-places:write
(place publishing), which is neither universe:write nor universe.place:write. The scope names checked are openapi's;
introspect names its dotted scopes the same way (universe.place.luau-execution-session, RB1307), but a key that holds
these two has not been read yet, so a refusal says what the key holds.

GATED. Nothing here publishes, lists or changes a setting unless confirm is true, dry_run is false and the MifBridge panel
is in full mode; a price on private servers (privateServerPriceRobux earns Robux) is refused outright. The key is read only
through opencloud.py and never returned. Store art: icon 512 x 512, thumbnails 1920 x 1080, up to 10 and two or more for
Home Page personalization, 24-bit RGB (a palette review sheet posterized a game's art) - the internal notes, Store art and
publishing. NEVER RAISE: every failure is {"ok": False, ...}.
"""
import json
import re
import time
import urllib.parse

__all__ = ["rbx_universe_settings", "rbx_store_list", "rbx_store_art_check", "rbx_save_to_roblox", "rbx_plugin_upload",
           "rbx_game_pass"]

# Writable fields (openapi: not readOnly). displayName and description of a universe are set through its root place.
UNIVERSE_FIELDS = {"voiceChatEnabled": bool, "desktopEnabled": bool, "mobileEnabled": bool, "tabletEnabled": bool,
                   "consoleEnabled": bool, "vrEnabled": bool}
UNIVERSE_REFUSED = {"privateServerPriceRobux": "a private-server price earns Robux; MifBridge never sets one - the person "
                                               "sets it in Creator Hub (Monetization > Private Servers)",
                    "displayName": "a universe's name is its root place's: set displayName on the place",
                    "description": "a universe's description is its root place's: set description on the place"}
PLACE_FIELDS = {"displayName": str, "description": str, "serverSize": int}
SCOPE_UNIVERSE = ("universe", "write")          # openapi universe:write
SCOPE_PLACE = ("universe.place", "write")       # openapi universe.place:write
KEY_STEP = ("Creator Hub > Open Cloud > API Keys > this key > Access Permissions: add %s with the %s operation for this "
            "experience, then Save.")


def _T():
    from mif_roblox import tools
    return tools


def _OC():
    from mif_roblox import opencloud
    return opencloud


def _no(msg, **k):
    return dict({"ok": False, "ran": False, "error": msg}, **k)


def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _get(path):
    """(status, body) of one Open Cloud GET with the key; body redacted. (None, refusal) without a key."""
    OC = _OC()
    key, err = OC.read_key()
    if err:
        return None, err
    st, out, _w = OC._call("GET", path, key, tries=3)
    return st, OC.redact(out, key)


def universe_of_place(place_id):
    """The universe a place belongs to: GET apis.roblox.com/universes/v1/places/{id}/universe, public, no key. ->
    (universe id, None) or (None, error)."""
    OC = _OC()
    try:
        st, _h, raw = OC.REQUEST("GET", "%s/universes/v1/places/%d/universe" % (OC.BASE, int(place_id)),
                                 {"Accept": "application/json"}, None, 30)
    except OSError as exc:
        return None, "the place's universe did not read: %s" % exc
    body = OC._json(raw)
    uid = body.get("universeId") if isinstance(body, dict) else None
    if st != 200 or not _pos_int(uid):
        return None, "GET /universes/v1/places/%s/universe answered %s: %s" % (place_id, st, json.dumps(body)[:200])
    return uid, None


def scope_check(needed, universe_id, info=None):
    """Does the key hold scope name:operation over universe_id? From introspect (RB1307): a scope's universeIds list,
    "*" meaning all. -> {ok, scope, holds, covers, keyScopes, steps?}."""
    OC = _OC()
    info = info if info is not None else OC.introspect()
    name, op = needed
    out = {"scope": "%s:%s" % (name, op)}
    if not info.get("ok"):
        return dict(out, ok=False, holds=None, error="the key's scopes did not read: %s" % info.get("error"))
    holds, res = OC.has_scope(info, name, op)
    ids = [str(x) for x in ((res or {}).get("universeIds") or [])]
    covers = holds and ("*" in ids or str(universe_id) in ids or not res)
    out.update(holds=holds, covers=bool(covers), keyScopes=sorted("%s:%s" % (s["name"], o) for s in info["scopes"]
                                                                  for o in s["operations"]))
    if not covers:
        out.update(ok=False, refusedRule="key-scope",
                   error="the key has no %s:%s%s, which this change needs" % (
                       name, op, "" if not holds else " for universe %s (it covers %s)" % (universe_id, ", ".join(ids))),
                   steps=[KEY_STEP % ("%s (%s:%s)" % ("Universe" if name == "universe" else "Universe Places",
                                                      name, op), op)])
    else:
        out["ok"] = True
    return out


def _validate(fields, allowed, refused):
    for k, v in fields.items():
        if k in refused:
            return refused[k], "robux" if "Robux" in refused[k] else "field"
        if k not in allowed:
            return "%s is not a field this tool sets; it sets %s" % (k, ", ".join(sorted(allowed))), "field"
        t = allowed[k]
        if t is bool and not isinstance(v, bool):
            return "%s is true or false" % k, "field"
        if t is int and not _pos_int(v):
            return "%s is a positive whole number" % k, "field"
        if t is str and not isinstance(v, str):
            return "%s is text" % k, "field"
    if "serverSize" in fields and not 1 <= fields["serverSize"] <= 700:
        return "serverSize is 1-700 (Roblox's Maximum Visitor Count range)", "field"
    return None, None


def _request(path, fields):
    return {"method": "PATCH", "path": "%s?updateMask=%s" % (path, ",".join(sorted(fields))),
            "contentType": "application/json", "body": {k: fields[k] for k in sorted(fields)}}


def rbx_universe_settings(universe_id: int = None, place_id: int = None, set_universe: dict = None,
                          set_place: dict = None, confirm: bool = False, dry_run: bool = True, allow_ids: list = None) -> dict:
    "Roblox Open Cloud: an experience's settings with the person's own key (read from their key file, never shown). Reads (free): the universe (GET /cloud/v2/universes/{id}: devices desktopEnabled/mobileEnabled/tabletEnabled/consoleEnabled/vrEnabled, voiceChatEnabled, visibility, ageRating, privateServerPriceRobux, rootPlace) and a place (GET .../places/{id}: displayName, description, serverSize), place_id alone finding its universe (a public read) and universe_id alone reading its root place. Changes: set_universe {voiceChatEnabled, desktopEnabled, mobileEnabled, tabletEnabled, consoleEnabled, vrEnabled} and set_place {displayName, description, serverSize 1-700} are checked against the key's scopes first (universe:write, universe.place:write, over this universe - the missing one named with the step that adds it) and, by default (dry_run true), only answered as the exact PATCH requests they would send, with the values now and after. A change is sent only with confirm true, dry_run false, the MifBridge panel in full mode, and the universe (and, for set_place, the place) on the person's write-target list (roblox_write_targets.json) or named in allow_ids (refusedRule write-target; the dry run says writeTarget), then read back field by field. privateServerPriceRobux is refused (it earns Robux; the person sets it in Creator Hub)."
    OC = _OC()
    for name, v in (("universe_id", universe_id), ("place_id", place_id)):
        if v is not None and not _pos_int(v):
            return _no("%s is a positive whole number" % name)
    if universe_id is None and place_id is None:
        return _no("give universe_id, place_id, or both")
    for name, v in (("set_universe", set_universe), ("set_place", set_place)):
        if v is not None and not (isinstance(v, dict) and v):
            return _no("%s is a non-empty object of fields to set" % name)
    if not isinstance(confirm, bool) or not isinstance(dry_run, bool):
        return _no("confirm and dry_run are true or false")
    err, rule = _validate(set_universe or {}, UNIVERSE_FIELDS, UNIVERSE_REFUSED)
    if not err:
        err, rule = _validate(set_place or {}, PLACE_FIELDS, {})
    if err:
        return _no(err, refusedRule=rule)
    try:
        out = {}
        if universe_id is None:
            universe_id, uerr = universe_of_place(place_id)
            if uerr:
                return dict(_no(uerr), ran=True)
            out["universeFrom"] = "place %d (public read)" % place_id
        out["universeId"] = universe_id
        upath = "/cloud/v2/universes/%d" % universe_id
        st, uni = _get(upath)
        if st is None:
            return uni
        if st != 200 or not isinstance(uni, dict):
            return dict(out, ok=False, ran=True, status=st, answer=uni,
                        error="GET %s answered %s: %s" % (upath, st, json.dumps(uni)[:300]))
        out["universe"] = uni
        if place_id is None:
            root = str(uni.get("rootPlace") or "")
            place_id = int(root.rsplit("/", 1)[-1]) if root.rsplit("/", 1)[-1].isdigit() else None
            out["placeFrom"] = "the universe's rootPlace"
        place = None
        ppath = "%s/places/%d" % (upath, place_id) if place_id else None
        if ppath:
            pst, place = _get(ppath)
            if pst != 200 or not isinstance(place, dict):
                return dict(out, ok=False, ran=True, status=pst, answer=place,
                            error="GET %s answered %s: %s (is place %s in universe %s?)" % (
                                ppath, pst, json.dumps(place)[:300], place_id, universe_id))
            out["placeId"], out["place"] = place_id, place
        if not set_universe and not set_place:
            out.update(ok=True, ran=True, verified=True)
            return out
        if set_place and not ppath:
            return dict(out, ok=False, ran=True, error="set_place needs a place: the universe named no rootPlace")
        info = OC.introspect()
        plan = []
        if set_universe:
            plan.append(("universe", upath, set_universe, uni, scope_check(SCOPE_UNIVERSE, universe_id, info)))
        if set_place:
            plan.append(("place", ppath, set_place, place, scope_check(SCOPE_PLACE, universe_id, info)))
        out["requests"] = [dict(_request(p, f), target=t, scope=sc,
                                change={k: {"now": (cur or {}).get(k), "to": f[k]} for k in sorted(f)})
                           for t, p, f, cur, sc in plan]
        missing = [r["scope"] for r in out["requests"] if not r["scope"].get("ok")]
        tgate = OC.write_target_check(universe_id, place_id if set_place else False, allow_ids)
        if tgate is not None and "notAllowed" not in tgate:   # a malformed allow_ids: its own refusal, nothing listed
            return dict(out, **tgate)
        out["writeTarget"] = "listed" if tgate is None else {k: tgate[k] for k in ("notAllowed", "targetsFile", "error")}
        if dry_run or not confirm:
            out.update(ok=not missing, ran=True, dryRun=True, sent=False,
                       note="nothing was sent: these are the exact requests a change would make. It is sent with "
                            "confirm true and dry_run false once the person agreed, with the panel in full mode")
            if missing:
                out.update(refusedRule="key-scope", error="; ".join(m.get("error") or m["scope"] for m in missing),
                           steps=[s for m in missing for s in m.get("steps") or []])
            return out
        if missing:
            return dict(out, ok=False, ran=False, refusedRule="key-scope", sent=False,
                        error="; ".join(m.get("error") or m["scope"] for m in missing),
                        steps=[s for m in missing for s in m.get("steps") or []])
        mode, _st = _T()._plugin_mode()
        if mode != "full":
            return dict(out, ok=False, ran=False, refusedRule="full-mode", mode=mode, sent=False,
                        error="a setting change writes to the person's account; the MifBridge panel must be in full mode "
                              "(it is %s)" % mode)
        if tgate is not None:
            return dict(out, **tgate)
        key, kerr = OC.read_key()
        if kerr:
            return kerr
        good = True
        for req, (t, p, f, _cur, _sc) in zip(out["requests"], plan):
            st, ans, _w = OC._call("PATCH", req["path"], key, json.dumps(req["body"]).encode("utf-8"), "application/json",
                                   tries=2)
            req["status"], req["answer"] = st, OC.redact(ans, key)
            if st != 200:
                good = False
                req["error"] = "PATCH answered %s: %s" % (st, json.dumps(req["answer"])[:300])
                if st in (401, 403):
                    req["refusedRule"] = "key-scope"
                continue
            bst, back = _get(p)
            req["readBack"] = {k: (back or {}).get(k) for k in f}
            req["verified"] = bst == 200 and all(req["readBack"][k] == f[k] for k in f)
            good = good and req["verified"]
        out.update(ok=good, ran=True, sent=True, verified=good, dryRun=False)
        if not good:
            out["error"] = "; ".join(r.get("error") or "%s read back %s" % (r["target"], r.get("readBack"))
                                     for r in out["requests"] if not r.get("verified"))
        return out
    except Exception as exc:  # noqa: BLE001 - never raise, never let the key out
        return {"ok": False, "ran": None, "error": "universe settings failed: %s" % OC.exc_text(exc)}


# ---- the Creator Store: Models listed at a price (POST /cloud/v2/creator-store-products, BETA)
PRODUCTS = "/cloud/v2/creator-store-products"
SCOPE_PRODUCT_WRITE = ("creator-store-product", "write")
MIN_GAP = 2.0                 # s between POSTs: Roblox's limit is 30 a minute per key owner
PROCESSING_WAIT = 120.0       # s of 503 "being processed" retries (measured about 20 s for a fresh Studio save, RB1186)
PROCESSING_EVERY = 10.0
RATE_WAIT = 65.0              # s after a 429 before the same POST again
RATE_TRIES = 2
PRICE_MIN, PRICE_MAX = 2.99, 49.99   # a paid Model's price range
SELLER_STEP = ("the seller steps no tool can do: ID verification, 2-Step Verification and a Stripe seller account, until "
               "Creator Hub > Settings > Eligibility > Priced assets says Approved for distribution (RB1101, RB1102)")
PACKAGE_STEP = ("a Model made by an Open Cloud upload is a package, and a package cannot be distributed (RB1111, RB1187): "
                "load it in Studio, save it as a new normal Model with rbx_save_to_roblox (CreateAssetAsync, IsPackage false, "
                "RB2006) or Studio's Save to Roblox (Development Item, Model, RB1186), and list that id")
STUDIO_ROUTE = "studio"     # an uploads.jsonl row rbx_save_to_roblox / rbx_plugin_upload wrote (CreateAssetAsync)


def money(price):
    """basePrice for a USD price: significand x 10^-9, and 0/0 for free (Roblox's Money format; the exponent must be -9)."""
    if price == 0:
        return {"currencyCode": "USD", "quantity": {"significand": 0, "exponent": 0}}
    return {"currencyCode": "USD", "quantity": {"significand": int(round(price * 1e9)), "exponent": -9}}


def price_of(m):
    """A Money answer as a float in USD, or None."""
    try:
        q = (m or {}).get("quantity") or {}
        return round(int(q.get("significand", 0)) * 10 ** int(q.get("exponent", 0)), 2)
    except (TypeError, ValueError):
        return None


def _uploaded_by_open_cloud(asset_id):
    """The uploads.jsonl row when rbx_asset_upload made this id (Open Cloud), else None."""
    OC = _OC()
    for r in OC._rows():
        if str(r.get("assetId")) == str(asset_id) and r.get("made") == "yes" and r.get("route") != STUDIO_ROUTE:
            return r
    return None


def _saved_by_studio(asset_id):
    """The uploads.jsonl row when rbx_save_to_roblox made this id (CreateAssetAsync, IsPackage false), else None."""
    OC = _OC()
    for r in OC._rows():
        if str(r.get("assetId")) == str(asset_id) and r.get("made") == "yes" and r.get("route") == STUDIO_ROUTE:
            return r
    return None


def _product_state(key, asset_id):
    OC = _OC()
    st, body, _w = OC._call("GET", "%s/%d" % (PRODUCTS, asset_id), key, tries=2)
    return st, OC.redact(body, key)


def _listed(body, price, published):
    return (isinstance(body, dict) and price_of(body.get("basePrice")) == round(price, 2)
            and bool(body.get("published")) == published and (not published or body.get("purchasable") is True))


def rbx_store_list(asset_ids: list, price_usd: float, published: bool = True, confirm: bool = False,
                   dry_run: bool = True) -> dict:
    "Roblox Open Cloud: list Models on the Creator Store at a price with the person's own key (read from their key file, never shown) - POST /cloud/v2/creator-store-products {modelAssetId, basePrice USD significand x 10^-9, published} (Roblox's BETA products API; the product id is the model's asset id). Before anything: each id is read (GET /assets/v1/assets/{id}: a Model, and on this machine's upload log = made by an Open Cloud upload, which is a package and cannot be distributed - refused with the rbx_save_to_roblox step, RB1111, RB1187; one rbx_save_to_roblox made is named a normal Model, RB2006), its product is read (GET .../creator-store-products/{id}: already listed at this price answers nothing to do; an existing product at another price is refused, since PATCH ignores price and published, RB1102), and the key's creator-store-product:write scope is checked (missing: refused with the step). By default (dry_run true) the answer is the exact POST each id would send. A listing is sent only with confirm true, dry_run false and the MifBridge panel in full mode: 2 s apart (30 a minute), a 503 'being processed' retried for up to 120 s (a fresh Studio save, RB1186), a 429 (the rate) waited out 65 s and sent again, at most twice, a 400 FAILED_PRECONDITION answered with the seller steps, then read back: price, published, purchasable. price_usd is 0 (free) or 2.99-49.99."
    OC = _OC()
    if not (isinstance(asset_ids, list) and asset_ids and all(_pos_int(a) for a in asset_ids)):
        return _no("asset_ids is a non-empty list of Model asset ids (positive whole numbers)")
    if len(set(asset_ids)) != len(asset_ids) or len(asset_ids) > 30:
        return _no("asset_ids holds each id once, at most 30 (Roblox's 30 a minute)")
    if isinstance(price_usd, bool) or not isinstance(price_usd, (int, float)) or not (
            price_usd == 0 or PRICE_MIN <= price_usd <= PRICE_MAX) or round(price_usd, 2) != price_usd:
        return _no("price_usd is 0 (free) or %.2f-%.2f in whole cents" % (PRICE_MIN, PRICE_MAX))
    if not all(isinstance(v, bool) for v in (published, confirm, dry_run)):
        return _no("published, confirm and dry_run are true or false")
    key, err = OC.read_key()
    if err:
        return err
    try:
        info = OC.introspect()
        holds, _res = OC.has_scope(info, *SCOPE_PRODUCT_WRITE) if info.get("ok") else (None, None)
        rows, todo = [], []
        for aid in asset_ids:
            row = {"assetId": aid}
            ast = OC.asset_state(aid)
            row["assetType"] = ast.get("assetType")
            up = _uploaded_by_open_cloud(aid)
            pst, prod = _product_state(key, aid)
            row["product"] = {"status": pst, "body": prod}
            if not ast.get("ok"):
                row.update(status="refused", refusedRule="not-found", error="the asset did not read: %s" % ast.get("error"))
            elif ast.get("assetType") != "Model":
                row.update(status="refused", refusedRule="type",
                           error="asset %d is a %s; this lists Models" % (aid, ast.get("assetType")))
            elif up:
                row.update(status="refused", refusedRule="package", uploadedAt=up.get("at"),
                           error="asset %d was made by an Open Cloud upload on %s: %s" % (aid, up.get("at"), PACKAGE_STEP))
            elif pst in (401, 403):
                row.update(status="refused", refusedRule="key-scope",
                           error="the products API answered %s: the key has no creator-store-product:read" % pst,
                           steps=[KEY_STEP % ("Creator Store Product (creator-store-product)", "read and write")])
            elif pst == 200 and _listed(prod, price_usd, published):
                row.update(status="listed", note="already a product at $%.2f, published %s: nothing to send"
                           % (price_usd, published))
            elif pst == 200:
                row.update(status="refused", refusedRule="exists",
                           error="a product already exists for %d at $%s, published %s; Roblox's PATCH answers 200 and "
                                 "changes neither price nor published (RB1102), so the person changes it on the model's "
                                 "Creator Hub page" % (aid, price_of((prod or {}).get("basePrice")), (prod or {}).get("published")))
            elif pst != 404:
                row.update(status="refused", refusedRule="read",
                           error="the product read answered %s: %s" % (pst, json.dumps(prod)[:200]))
            else:
                row.update(status="to-list", request={"method": "POST", "path": PRODUCTS, "contentType": "application/json",
                                                      "body": {"modelAssetId": str(aid), "basePrice": money(price_usd),
                                                               "published": published}})
                saved = _saved_by_studio(aid)
                row["packageCheck"] = (
                    "saved by rbx_save_to_roblox on %s (CreateAssetAsync, IsPackage false): a normal Model (RB2006)"
                    % saved.get("at") if saved else
                    "not made by an Open Cloud upload on this machine (its uploads.jsonl); no API says "
                    "whether a Model is a package (RB1111, RB1654), so an approved seller's POST answering "
                    "400 FAILED_PRECONDITION means it is one")
                todo.append(row)
            rows.append(row)
        out = {"priceUsd": price_usd, "published": published, "items": rows,
               "scope": {"scope": "creator-store-product:write", "holds": holds}}
        refused = [r for r in rows if r["status"] == "refused"]
        if todo and holds is not True:
            out.update(ok=False, ran=True, refusedRule="key-scope", sent=False,
                       error="the key has no creator-store-product:write%s" % (
                           "" if info.get("ok") else " (its scopes did not read: %s)" % info.get("error")),
                       steps=[KEY_STEP % ("Creator Store Product (creator-store-product)", "write")])
            return out
        if dry_run or not confirm or not todo:
            out.update(ok=not refused, ran=True, dryRun=bool(todo), sent=False,
                       note=("nothing was sent: these are the exact requests. They are sent with confirm true and dry_run "
                             "false once the person agreed, with the panel in full mode") if todo else
                       "nothing to send")
            if refused:
                out["error"] = "; ".join(r["error"] for r in refused)
            return out
        mode, _st = _T()._plugin_mode()
        if mode != "full":
            return dict(out, ok=False, ran=False, refusedRule="full-mode", mode=mode, sent=False,
                        error="a listing sells on the person's account; the MifBridge panel must be in full mode (it is %s)"
                              % mode)
        last = 0.0
        for row in todo:
            wait = MIN_GAP - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            body = json.dumps(row["request"]["body"]).encode("utf-8")
            t0, tries, rated = time.time(), 0, 0
            while True:
                last = time.time()
                st, ans, _w = OC._call("POST", PRODUCTS, key, body, "application/json", tries=1)
                tries += 1
                if st == 503 and time.time() - t0 + PROCESSING_EVERY <= PROCESSING_WAIT:
                    time.sleep(max(PROCESSING_EVERY, MIN_GAP))  # a retry is a POST too: still 30 a minute
                    continue
                # 429: the products API refused it for the rate, so nothing was listed. Measured: a second group of 10
                # sent straight after a first answered 429, and 65 s later it went through.
                if st == 429 and rated < RATE_TRIES:
                    time.sleep(RATE_WAIT)
                    rated += 1
                    continue
                break
            row.update(postStatus=st, answer=OC.redact(ans, key), tries=tries, sent=True)
            if rated:
                row["rateWaits"] = rated
            if st != 200:
                msg = json.dumps(row["answer"])[:300]
                row.update(status="failed", error="POST answered %s after %d tr%s: %s" % (st, tries, "y" if tries == 1 else
                                                                                       "ies", msg))
                if not isinstance(st, int) or st in (0, 408) or st >= 500:
                    # no answer, a timeout or a server error: the listing may have landed
                    row.update(status="unknown", outcomeUnknown=True,
                               error=row["error"] + ". Whether it was listed is unknown: read the product (rbx_store_list "
                                                    "dry run) before sending it again")
                    continue
                if st == 400 and "FAILED_PRECONDITION" in msg:
                    row.update(refusedRule="seller-or-package",
                               steps=[SELLER_STEP, "if the seller is approved: " + PACKAGE_STEP])
                elif st in (401, 403):
                    row["refusedRule"] = "key-scope"
                continue
            bst, back = _product_state(key, row["assetId"])
            row["readBack"] = {"status": bst, "price": price_of((back or {}).get("basePrice")),
                               "published": (back or {}).get("published"), "purchasable": (back or {}).get("purchasable"),
                               "restrictions": (back or {}).get("restrictions")}
            row["status"] = "listed" if bst == 200 and _listed(back, price_usd, published) else "unverified"
        bad = [r for r in rows if r["status"] not in ("listed",)]
        out.update(ok=not bad, ran=True, sent=True, verified=not bad, dryRun=False)
        if any(r.get("outcomeUnknown") for r in rows):
            out["outcomeUnknown"] = True
        if bad:
            out["error"] = "; ".join("%s: %s" % (r["assetId"], r.get("error") or r["status"]) for r in bad)
        return out
    except Exception as exc:  # noqa: BLE001 - never raise, never let the key out
        return OC.redact({"ok": False, "ran": None, "error": "store listing failed: %s" % OC.exc_text(exc, key)}, key)


# ---- saving from Studio with no dialog: AssetService:CreateAssetAsync from the plugin (RB2006)
CREATE_TIMEOUT = 120            # s for the CreateAssetAsync call (it yields while Roblox makes the asset)
READ_TRIES, READ_EVERY = 6, 5.0  # the new id read back: the Assets API and LoadAsset, up to 6 tries 5 s apart
TREE_LIMIT = 2000               # nodes the dry run lists; a bigger tree is refused rather than described in part
TREE_DEPTH = 20                 # rbx_tree's deepest; the source is read one level less (LoadAsset adds a wrapper)
NAME_MAX, DESC_MAX = 50, 1000   # Creator Hub's limits on an asset's name and description
NOT_AVAILABLE = "not available yet"
BETA_STEP = ("turn on Studio's CreateAssetAsync beta: File > Beta Features, tick the entry for AssetService:CreateAssetAsync "
             "(Studio's key Studio.BetaFeatures.CreateAssetAsyncTitle), Save - measured to take effect in the running "
             "Studio (RB2006). It is stored per Windows user in HKCU\\Software\\Roblox\\RobloxStudio, value "
             "BetaFeatureInformation, allBetaFeatures.CreateAssetAsync.enabled true (RB1421)")
SAVE_KINDS = {
    "Model": {"classes": ("Model",), "params": lambda n, d: {"Name": n, "Description": d, "IsPackage": False},
              "what": "a normal (non-package) Model"},
    "Plugin": {"classes": ("Folder", "Script", "ModuleScript", "LocalScript", "Model"),
               "params": lambda n, d: {"Name": n, "Description": d}, "what": "a Plugin asset"},
}


def _flat(node, prefix=""):
    """[(relative path, class)] for every node under a tree node (rbx_tree), the node itself first as ''."""
    rows = [(prefix, node.get("className"))]
    for c in node.get("children") or []:
        rows += _flat(c, (prefix + "." if prefix else "") + str(c.get("name")))
    return rows


def _describe(T, target):
    """What a save would send: the object's path, class and every descendant (path and class), counted by class.
    -> (description, refusal)."""
    st = T.rbx_status()
    if not st.get("ok") or not (st.get("plugin") or {}).get("connected", True):
        return None, _no("nothing was saved: Roblox Studio is not running or the MifBridge plugin is not connected (%s). "
                         "Launch it with rbx_launch first" % (st.get("error") or "no answer"), refusedRule="no-studio")
    tree = T.rbx_tree(root=target, depth=TREE_DEPTH - 1, limit=TREE_LIMIT)
    if not tree.get("ok") or not isinstance(tree.get("root"), dict):
        return None, _no("the target did not read: %s" % (tree.get("error") or "no tree"), refusedRule="not-found")
    if tree.get("truncated"):
        return None, _no("the target holds %d or more objects: a save is described in full before it is sent, so this "
                         "refuses one larger than %d" % (TREE_LIMIT, TREE_LIMIT), refusedRule="too-large")
    rows = _flat(tree["root"])
    counts = {}
    for _p, c in rows[1:]:
        counts[c] = counts.get(c, 0) + 1
    return {"path": target if isinstance(target, str) else json.dumps(target), "className": tree["root"].get("className"),
            "name": tree["root"].get("name"), "descendants": len(rows) - 1, "byClass": dict(sorted(counts.items())),
            "objects": ["%s (%s)" % (p, c) for p, c in rows[1:]], "rows": rows,
            "placeName": (st.get("plugin") or {}).get("placeName")}, None


def _studio_row(name, kind, out):
    OC = _OC()
    try:
        OC.log_upload(None, name, out, asset_type=kind, route=STUDIO_ROUTE)
    except TypeError:  # an opencloud.py without route: never fail the save for the log
        pass


def _compare(tree, desc):
    """The loaded copy (rbx_tree of what LoadAsset gave) against the source's rows: LoadAsset wraps the saved object in
    a Model of its own, so the saved object is the wrapper's one child. A tree that did not read fails, never compares
    as empty."""
    if not tree.get("ok") or not isinstance(tree.get("root"), dict):
        return {"ok": False, "error": "the loaded copy did not read: %s" % (tree.get("error") or "no tree")}
    root = tree["root"]
    kids = root.get("children") or []
    inner = kids[0] if len(kids) == 1 else None
    got = _flat(inner) if inner else []
    want = desc["rows"]
    links = [p for p, c in got if c == "PackageLink"]
    same = bool(inner) and inner.get("className") == desc["className"] and sorted(got[1:]) == sorted(want[1:])
    out = {"ok": True, "wrapper": root.get("className"), "wrapperChildren": len(kids),
           "savedAs": inner and inner.get("name"), "className": inner and inner.get("className"),
           "descendants": len(got) - 1 if got else 0, "sameAsSource": same, "packageLinks": links}
    if not same:
        out["diff"] = {"source": sorted("%s (%s)" % r for r in set(want[1:]) - set(got[1:]))[:20],
                       "loaded": sorted("%s (%s)" % r for r in set(got[1:]) - set(want[1:]))[:20]}
    return out


def _read_back(T, OC, kind, aid, desc):
    """The new id read back: the Assets API's assetType (needs the key's asset:read), and for a Model the id loaded
    with InsertService:LoadAsset into Workspace.MifScratch, compared object for object with the source, no PackageLink,
    then removed. -> {verified, ...}."""
    out = {"assetApi": None, "loaded": None}
    key, kerr = OC.read_key()
    for i in range(READ_TRIES):
        if kerr:
            out["assetApi"] = {"read": False, "error": (kerr or {}).get("error") or "no key"}
            break
        a = OC.asset_state(aid)
        out["assetApi"] = {k: a.get(k) for k in ("ok", "assetType", "state", "moderationState", "error", "status")}
        if a.get("ok"):
            break
        time.sleep(READ_EVERY)
    if kind == "Model":
        made_scratch = False
        if not T.rbx_find(name="MifScratch", root="Workspace").get("matches"):
            made_scratch = bool(T.rbx_create("Folder", "Workspace", name="MifScratch").get("ok"))
        try:
            loaded = {}
            for i in range(READ_TRIES):
                loaded = T.rbx_call("InsertService", "LoadAsset", [aid], timeout=60)
                if isinstance(loaded.get("value"), dict) and loaded["value"].get("handle"):
                    break
                time.sleep(READ_EVERY)
            h = (loaded.get("value") or {}).get("handle") if isinstance(loaded.get("value"), dict) else None
            if not h:
                out["loaded"] = {"ok": False, "error": loaded.get("error") or "LoadAsset returned %r" % (loaded.get("value"),)}
            else:
                path, perr = T._place_handle(h, ["Workspace", "MifScratch"], "MifSavedCheck%d" % aid)
                if perr:
                    T.rbx_release_handle(h)
                    out["loaded"] = {"ok": False, "error": perr}
                else:
                    tree = T.rbx_tree(root=path, depth=TREE_DEPTH, limit=TREE_LIMIT + 10)
                    out["loaded"] = _compare(tree, desc)
                    out["loaded"]["removed"] = bool(T.rbx_destroy(path).get("ok"))
        finally:
            if made_scratch:   # the folder this call made is not left in the person's place
                out["scratchRemoved"] = bool((T.rbx_destroy("Workspace.MifScratch") or {}).get("ok"))
    api_ok = (out["assetApi"] or {}).get("assetType") == kind
    if kind == "Model":
        ld = out["loaded"] or {}
        out["verified"] = bool(api_ok or (out["assetApi"] or {}).get("read") is False) and bool(
            ld.get("ok") and ld.get("sameAsSource") and not ld.get("packageLinks"))
    else:
        out["verified"] = api_ok
    return out


def _create_asset(kind, target, name, description, confirm, dry_run, again):
    T, OC = _T(), _OC()
    spec = SAVE_KINDS[kind]
    if not isinstance(name, str) or not name.strip() or len(name) > NAME_MAX:
        return _no("name is the asset's name: 1-%d characters" % NAME_MAX)
    if not isinstance(description, str) or len(description) > DESC_MAX:
        return _no("description is text, at most %d characters" % DESC_MAX)
    if not all(isinstance(v, bool) for v in (confirm, dry_run, again)):
        return _no("confirm, dry_run and again are true or false")
    if not (isinstance(target, str) and target.strip()):
        return _no("target is the path of the object to save, e.g. Workspace.MifScratch.MyModel")
    try:
        desc, refusal = _describe(T, target)
        if refusal:
            return refusal
        public = {k: v for k, v in desc.items() if k != "rows"}
        call = {"call": "AssetService:CreateAssetAsync", "object": target, "assetType": "Enum.AssetType.%s" % kind,
                "requestParameters": spec["params"](name, description)}
        out = {"kind": kind, "saves": public, "request": call}
        if desc["className"] not in spec["classes"]:
            return dict(out, ok=False, ran=False, refusedRule="type",
                        error="%s is a %s; %s saves %s" % (target, desc["className"], "rbx_save_to_roblox" if kind == "Model"
                                                          else "rbx_plugin_upload", " or ".join(spec["classes"])))
        links = [p for p, c in desc["rows"] if c == "PackageLink"]
        if links:
            return dict(out, ok=False, ran=False, refusedRule="package-link",
                        error="%s holds a PackageLink (%s): whether the saved asset would become a package is not measured, "
                              "so remove the link first" % (target, ", ".join(p or "(itself)" for p in links)))
        scripts = [p for p, c in desc["rows"] if c in ("Script", "LocalScript", "ModuleScript")]
        if scripts and kind == "Model":
            out["note"] = ("it holds %d script(s), saved with it and readable by anyone who gets the model: %s"
                           % (len(scripts), ", ".join(scripts[:10])))
        damaged = OC.log_damage()
        if damaged:
            return dict(out, **damaged)
        prior = [r for r in OC._rows() if r.get("route") == STUDIO_ROUTE and r.get("made") in ("yes", "unknown")
                 and r.get("displayName") == name and r.get("assetType") == kind]
        if prior and not again:
            p = prior[-1]
            return dict(out, ok=False, ran=False, refusedRule="already-saved", earlier={k: p.get(k) for k in
                                                                                       ("at", "assetId", "made")},
                        error="a %s named %r was already saved from Studio on %s (%s): each save is a new permanent asset, "
                              "so this refuses a second one unless again is true" % (kind, name, p.get("at"),
                                                                                     p.get("assetId") or "outcome unknown"))
        if dry_run or not confirm:
            return dict(out, ok=True, ran=True, dryRun=True, sent=False,
                        note=(out.get("note", "") + "; " if out.get("note") else "") +
                        "nothing was saved: this is exactly what would be. It is saved, as %s, permanent and moderated, "
                        "with confirm true and dry_run false once the person agreed, with the panel in full mode"
                        % spec["what"])
        mode, _st = T._plugin_mode()
        if mode != "full":
            return dict(out, ok=False, ran=False, refusedRule="full-mode", mode=mode, sent=False,
                        error="a save makes a permanent asset on the person's account; the MifBridge panel must be in full "
                              "mode (it is %s)" % mode)
        t0 = time.time()
        r = T.rbx_call("AssetService", "CreateAssetAsync", [target, kind, call["requestParameters"]], timeout=CREATE_TIMEOUT)
        out["seconds"] = round(time.time() - t0, 2)
        if not r.get("ok"):
            err = str(r.get("error") or "")
            if r.get("outcomeUnknown"):
                _studio_row(name, kind, {"made": "unknown", "outcomeUnknown": True})
                return dict(out, ok=False, ran=True, sent=True, made="unknown", outcomeUnknown=True,
                            error="the call did not answer in time: whether an asset was made is unknown. Check Creator "
                                  "Hub (Creations) before saving again; a second save needs again true")
            if NOT_AVAILABLE in err:
                return dict(out, ok=False, ran=True, sent=True, made="no", refusedRule="beta-off", answer=err,
                            steps=[BETA_STEP], error="Studio answered %r: its CreateAssetAsync beta is off (RB1306, RB1311)"
                                                     % err)
            return dict(out, ok=False, ran=True, sent=True, made="no", refusedRule=r.get("refusedRule") or "failed",
                        answer=err, error="CreateAssetAsync failed: %s" % err)
        vals = r.get("values") or [r.get("value")]
        result = vals[0] if vals else None
        try:
            aid = int(vals[1]) if len(vals) > 1 and not isinstance(vals[1], bool) else None
        except (TypeError, ValueError):
            aid = None
        out["answer"] = vals
        if result != "Success" or not _pos_int(aid):
            return dict(out, ok=False, ran=True, sent=True, made="no", refusedRule="failed",
                        error="CreateAssetAsync answered %r, not Success with an id" % (vals,))
        aid = int(aid)
        _studio_row(name, kind, {"assetId": str(aid)})
        out.update(assetId=aid, made="yes", sent=True, ran=True)
        out["readBack"] = _read_back(T, OC, kind, aid, desc)
        out.update(ok=bool(out["readBack"].get("verified")), verified=bool(out["readBack"].get("verified")))
        if not out["ok"]:
            out["error"] = "saved as %d, but the read-back did not confirm %s: %s" % (
                aid, spec["what"], json.dumps(out["readBack"], default=str)[:400])
        return out
    except Exception as exc:  # noqa: BLE001 - never raise
        return {"ok": False, "ran": None, "error": "saving to Roblox failed: %s" % OC.exc_text(exc)}


def rbx_save_to_roblox(target: str, name: str, description: str = "", confirm: bool = False, dry_run: bool = True,
                       again: bool = False) -> dict:
    "Roblox Studio: save a Model in the open place to Roblox as a NEW normal (non-package) Model on the person's account, with no dialog - AssetService:CreateAssetAsync(model, Enum.AssetType.Model, {Name, Description, IsPackage = false}) from the MifBridge plugin (RB2006; the Model is sellable, unlike an Open Cloud Model upload, which is a package, RB1111, RB1187). By default (dry_run true) the answer is exactly what would be saved: the model's path, class and every descendant by path and class, counts by class, the scripts it carries, and the exact call. A save is permanent and moderated, so it is sent only with confirm true, dry_run false and the MifBridge panel in full mode. Refused before anything is sent: a target that is not a Model (refusedRule type), one holding a PackageLink (package-link: unmeasured whether the asset would become a package), one over 2000 objects (too-large: it could not be described in full), and a name already saved from this machine (already-saved, its id and time; again true saves a second one). Studio's answer 'not available yet' is refusedRule beta-off with the step that turns on Studio's CreateAssetAsync beta (RB1306, RB1311). Read back after a save: the Assets API's assetType for the new id (the person's key, never shown), and the id loaded with InsertService:LoadAsset into Workspace.MifScratch, compared object for object with the source, checked for a PackageLink, and removed; verified is true only when both agree. Never makes a version of an existing asset: CreateAssetVersionAsync without IsPackage false turns a listed Model into a package for good (RB2008). The save is logged in the MifBridge home's uploads.jsonl, so rbx_store_list names the id a normal Model. List it with rbx_store_list."
    return _create_asset("Model", target, name, description, confirm, dry_run, again)


def rbx_plugin_upload(target: str, name: str, description: str = "", confirm: bool = False, dry_run: bool = True,
                      again: bool = False) -> dict:
    "Roblox Studio: save a Folder, Script, LocalScript, ModuleScript or Model in the open place to Roblox as a NEW Plugin asset on the person's account, with no dialog - AssetService:CreateAssetAsync(object, Enum.AssetType.Plugin, {Name, Description}) from the MifBridge plugin. The same gate as rbx_save_to_roblox: by default a dry run that lists every object that would be saved and the exact call; sent only with confirm true, dry_run false and full mode; already-saved, package-link, too-large and beta-off refusals the same. Read back: the Assets API's assetType for the new id is Plugin (the person's key, never shown). The asset is the person's own and is not put on the Creator Store by this."
    return _create_asset("Plugin", target, name, description, confirm, dry_run, again)


def module_value(T, module, keypath=None):
    """A ModuleScript's returned table (or one dotted key of it) through rbx_module_values. -> (value, error)."""
    fn = getattr(T, "TOOLS", {}).get("rbx_module_values")
    if fn is None:
        return None, "rbx_module_values is not on this MifBridge build, so config values cannot be read"
    r = fn(module)
    if not r.get("ok"):
        return None, r.get("error") or "rbx_module_values answered no value"
    val = r.get("value", r.get("values"))
    for k in [k for k in (keypath or "").split(".") if k]:
        if not isinstance(val, dict) or k not in val:
            return None, None if isinstance(val, dict) else "%s is not a table at %s" % (module, k)
        val = val[k]
    return val, None


# ---- store art: the icon and thumbnails checked, one review sheet written
ICON_SIZE = (512, 512)
THUMB_SIZE = (1920, 1080)
THUMB_MAX, THUMB_MIN_PERSONALIZE = 10, 2


def image_info(path):
    """(info, None) or (None, error) from the file's own header, standard library only: PNG IHDR or a JPEG SOF marker.
    info {format, width, height, mode}: mode RGB, RGBA, palette, grey, grey+alpha or CMYK."""
    import struct
    try:
        with open(path, "rb") as fh:
            data = fh.read(1 << 20)
    except OSError as exc:
        return None, "%s did not open: %s" % (path, exc.strerror)
    if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
        mode = {0: "grey", 2: "RGB", 3: "palette", 4: "grey+alpha", 6: "RGBA"}.get(ctype, "type %d" % ctype)
        return {"format": "PNG", "width": w, "height": h, "mode": mode, "bitDepth": depth}, None
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                i += 1 if marker == 0xFF else 2
                continue
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                comps = data[i + 9]
                return {"format": "JPEG", "width": w, "height": h,
                        "mode": {1: "grey", 3: "RGB", 4: "CMYK"}.get(comps, "%d components" % comps)}, None
            i += 2 + seg
        return None, "%s is a JPEG with no frame header in its first MB" % path
    return None, "%s is not a PNG or a JPEG" % path


def _check_image(path, want, role):
    info, err = image_info(path)
    if err:
        return {"path": path, "role": role, "ok": False, "problems": [err]}
    probs = []
    if (info["width"], info["height"]) != want:
        probs.append("%d x %d, not %d x %d" % (info["width"], info["height"], want[0], want[1]))
    if info["mode"] not in ("RGB", "RGBA"):
        probs.append("%s colour, not 24-bit RGB (a palette image posterizes the art)" % info["mode"])
    if info.get("bitDepth") not in (None, 8):
        probs.append("%d bits a channel, not 8" % info["bitDepth"])
    return dict(info, path=path, role=role, ok=not probs, problems=probs)


def review_sheet(icon, thumbs, out_path):
    """One sheet: the icon at full size, then each thumbnail at half size (960 x 540) two to a row, written as a 24-bit RGB
    PNG (lossless). Needs Pillow to decode JPEGs and resample. -> (info, None) or (None, error)."""
    try:
        from PIL import Image  # noqa: PLC0415 - optional: only the sheet needs it
    except ImportError:
        return None, "the review sheet needs Pillow (pip install Pillow); the checks above ran without it"
    tw, th, pad = 960, 540, 16
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    width = pad + cols * (tw + pad)
    height = pad + ICON_SIZE[1] + pad + rows * (th + pad)
    sheet = Image.new("RGB", (width, height), (24, 24, 28))
    with Image.open(icon) as im:
        sheet.paste(im.convert("RGB").resize(ICON_SIZE, Image.LANCZOS), (pad, pad))
    for n, t in enumerate(thumbs):
        with Image.open(t) as im:
            x = pad + (n % cols) * (tw + pad)
            y = pad + ICON_SIZE[1] + pad + (n // cols) * (th + pad)
            sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
    sheet.save(out_path, "PNG")
    info, err = image_info(out_path)
    return (dict(info, path=out_path), None) if not err else (None, err)


def rbx_store_art_check(icon: str, thumbnails: list, sheet: str = None, claims: list = None,
                        config_module: str = None) -> dict:
    "Roblox: an experience's store art checked before it goes up, on files on this machine: the icon 512 x 512 and each thumbnail 1920 x 1080 (read from the file's own PNG or JPEG header), 24-bit RGB or RGBA and never a palette or grey image (a palette review sheet posterized the art), 1 to 10 thumbnails and a warning under 2 (Home Page personalization needs two or more). Then one review sheet - the icon at full size and every thumbnail at half size - written to sheet (default store_review_sheet.png beside the icon) as a lossless 24-bit RGB PNG (needs Pillow). claims [{claim, image, key}] with config_module (a ModuleScript path in the open Studio) reads each key's value through rbx_module_values and prints it beside the claim it backs, so a number on an image is checked against the game's one table; a key that does not read fails. Writes only the sheet file."
    import os
    if not isinstance(icon, str) or not icon:
        return _no("icon is the path of the icon image")
    if not (isinstance(thumbnails, list) and all(isinstance(t, str) and t for t in thumbnails)):
        return _no("thumbnails is a list of image paths")
    if not thumbnails:
        return _no("no thumbnails given: an experience needs at least one, and two or more for personalization")
    if len(thumbnails) > THUMB_MAX:
        return _no("%d thumbnails: Roblox takes up to %d" % (len(thumbnails), THUMB_MAX))
    if claims is not None and not (isinstance(claims, list) and claims and all(
            isinstance(c, dict) and isinstance(c.get("claim"), str) and isinstance(c.get("key"), str) for c in claims)):
        return _no("claims is a list of {claim, image?, key}")
    if claims and not (isinstance(config_module, str) and config_module):
        return _no("claims need config_module, the ModuleScript the keys are read from")
    rows = [_check_image(icon, ICON_SIZE, "icon")] + [_check_image(t, THUMB_SIZE, "thumbnail") for t in thumbnails]
    out = {"images": rows, "thumbnailCount": len(thumbnails), "warnings": []}
    if len(thumbnails) < THUMB_MIN_PERSONALIZE:
        out["warnings"].append("one thumbnail: Home Page personalization tests need two or more")
    bad = [r for r in rows if not r["ok"]]
    if claims:
        T = _T()
        known = {os.path.basename(p) for p in [icon] + thumbnails}
        out["claims"] = []
        for c in claims:
            v, err = module_value(T, config_module, c["key"])
            row = {"claim": c["claim"], "image": c.get("image"), "key": c["key"], "value": v}
            if err or v is None:
                row["error"] = err or "%s holds no value at %s" % (config_module, c["key"])
            if c.get("image") and os.path.basename(c["image"]) not in known:
                row["error"] = (row.get("error", "") + "; " if row.get("error") else "") + \
                    "image %s is not one of the images checked" % c["image"]
            out["claims"].append(row)
    sheet = sheet or os.path.join(os.path.dirname(os.path.abspath(icon)), "store_review_sheet.png")
    if not bad:
        info, err = review_sheet(icon, thumbnails, sheet)
        out["sheet"] = info if info else {"path": sheet, "error": err}
    else:
        out["sheet"] = {"path": sheet, "error": "not written: an image failed its check"}
    failed = [("%s %s: %s" % (r["role"], r["path"], "; ".join(r["problems"]))) for r in bad]
    failed += ["claim %r: %s" % (c["claim"], c["error"]) for c in out.get("claims") or [] if c.get("error")]
    if (out["sheet"] or {}).get("error") and not bad:
        failed.append("sheet: %s" % out["sheet"]["error"])
    out.update(ok=not failed, ran=True, verified=not failed)
    if failed:
        out["error"] = "; ".join(failed)
    return out


# ---- the publish preflight: what a game's first publish needed checked by hand
KEY_SHAPES = [
    # (kind, pattern). A match is reported by script, line and kind - never its text.
    ("x-api-key header", r"""["']x-api-key["']"""),
    ("Roblox cookie", r"_\|WARNING:-DO-NOT-SHARE-THIS"),
    ("JWT-shaped token", r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
    ("long secret-shaped token", r"(?<![A-Za-z0-9+/=_-])(?=[A-Za-z0-9+/=_-]{40,})(?=[^\s'\"]*[0-9])(?=[^\s'\"]*[a-z])"
                                 r"(?=[^\s'\"]*[A-Z])[A-Za-z0-9+/=_-]{40,}(?![A-Za-z0-9+/=_-])"),
]
POLICY_RESTRICTED = "ArePaidRandomItemsRestricted"


def strip_luau_comments(src):
    """Luau source with its comments (-- to the end of the line, --[[ ]] and --[==[ ]==] blocks) blanked, strings kept:
    a name inside a comment is not code that reads it. A -- inside a quoted or [[ ]] string is not a comment."""
    import re
    if not isinstance(src, str):
        return ""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "-" and src.startswith("--", i):
            m = re.match(r"\[(=*)\[", src[i + 2:i + 2 + 64])
            if m:
                close = "]" + m.group(1) + "]"
                j = src.find(close, i + 2 + m.end())
                i = n if j < 0 else j + len(close)
            else:
                j = src.find("\n", i)
                i = n if j < 0 else j
            out.append(" ")
            continue
        if c in "\"'`":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if c == "[":
            m = re.match(r"\[(=*)\[", src[i:i + 64])
            if m:
                close = "]" + m.group(1) + "]"
                j = src.find(close, i + m.end())
                j = n if j < 0 else j + len(close)
                out.append(src[i:j])
                i = j
                continue
        out.append(c)
        i += 1
    return "".join(out)


def scan_secrets(sources, own_key=None):
    """[{script, line, kind}] for every key-shaped string in the scripts' sources (sources: [{path, source}]). own_key,
    the person's Open Cloud key, is matched exactly and named as theirs; it never leaves this function."""
    import re
    rows = []
    pats = [(k, re.compile(p)) for k, p in KEY_SHAPES]
    for s in sources:
        src = s.get("source") or ""
        path = ".".join(s["path"]) if isinstance(s.get("path"), list) else str(s.get("path"))
        for i, line in enumerate(src.split("\n"), 1):
            if own_key and own_key in line:
                rows.append({"script": path, "line": i, "kind": "the person's own Open Cloud key"})
                continue
            for kind, rx in pats:
                if rx.search(line):
                    rows.append({"script": path, "line": i, "kind": kind})
                    break
    return rows


# A secret read: HttpService:GetSecret("name"). The name counts only when a quoted literal is the whole argument; a
# variable, a join or a bare reference is a read whose name is not known here.
SECRET_READ = (r"""[:.]\s*GetSecret\b\s*(\(\s*(?:[A-Za-z_][\w.]*\s*,\s*)?)?"""
               r"""(?:"([^"\\\n]*)"|'([^'\\\n]*)'|\[\[([^\]\n]*)\]\])?\s*(\))?""")


def scan_secret_reads(sources):
    """([{script, line, name}], [{script, line}]) for every HttpService:GetSecret in the scripts' sources (sources:
    [{path, source}]): the reads whose name is a literal, and the reads whose name is computed (unresolved, never
    guessed). A read after a line comment's -- is skipped; a block comment is not followed."""
    import re
    rx = re.compile(SECRET_READ)
    reads, unresolved = [], []
    for s in sources:
        if not isinstance(s, dict):
            continue
        src = s.get("source") if isinstance(s.get("source"), str) else ""
        path = ".".join(str(p) for p in s["path"]) if isinstance(s.get("path"), list) else str(s.get("path"))
        for i, line in enumerate(src.split("\n"), 1):
            for m in rx.finditer(line):
                if "--" in line[:m.start()]:
                    continue
                opened, closed = m.group(1), m.group(5)
                name = next((g for g in m.group(2, 3, 4) if g is not None), None)
                if name is not None and (closed if opened else True):
                    reads.append({"script": path, "line": i, "name": name})
                else:
                    unresolved.append({"script": path, "line": i})
    return reads, unresolved


def _flag_value(T, flag):
    """A self-test flag's value: "Path" (a ValueBase's Value), "Path@Attribute" (an attribute) or "Module:key.sub" (a
    ModuleScript's returned value, through rbx_module_values). -> (value, how, error)."""
    if "@" in flag:
        path, attr = flag.rsplit("@", 1)
        r = T.rbx_call(path, "GetAttribute", [attr])
        return (r.get("value"), "attribute %s of %s" % (attr, path), None) if r.get("ok") else (None, None, r.get("error"))
    if ":" in flag:
        mod, keypath = flag.split(":", 1)
        got, err = module_value(T, mod, keypath)
        return got, "%s of ModuleScript %s" % (keypath, mod), err
    r = T.rbx_get_props(flag, ["Value"])
    if not r.get("ok"):
        return None, None, r.get("error")
    return (r.get("values") or {}).get("Value"), "Value of %s" % flag, None


def _id_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and float(v).is_integer()


def _entries(val):
    """A table's entries in order as (label, value): a list by position, a dict keyed "1".."n" (a Luau array as the
    plugin sends it) by number, any other dict by key."""
    if isinstance(val, list):
        return [("[%d]" % (i + 1), v) for i, v in enumerate(val)]
    if isinstance(val, dict):
        if val and all(isinstance(k, str) and k.isdigit() for k in val):
            return [("[%s]" % k, val[k]) for k in sorted(val, key=int)]
        return [(".%s" % k, val[k]) for k in sorted(val)]
    return []


def id_leaves(table, keypath):
    """Every value an id key names in a config table, as [(path, value)]. keypath is dotted; a * segment is every entry
    of the table there (a list of passes or products), so 'Store.Passes.*.Id' reaches the id inside each entry (found
    2026-11-23: a store list's ids sat inside array entries and the check could only read top-level keys). A key that
    is missing reads None; a * over an empty or missing table reads one None, so a list with nothing in it is unset."""
    segs = [s for s in keypath.split(".") if s]
    out = []

    def walk(val, i, path):
        if i == len(segs):
            out.append((path, val))
            return
        s = segs[i]
        if s == "*":
            items = _entries(val)
            if not items:
                out.append((path + ".*", None))
            for label, v in items:
                walk(v, i + 1, path + label)
            return
        nxt = val.get(s) if isinstance(val, dict) else None
        walk(nxt, i + 1, (path + "." + s) if path else s)
    walk(table, 0, "")
    return out


RUNTIME_UI = re.compile(r"\bScreenGui\b")


def runtime_ui_scripts(sources):
    """Client-side scripts (LocalScript, ModuleScript) whose code - comments stripped - names ScreenGui: a UI built at
    run time, which leaves StarterGui empty in Edit."""
    return [".".join(s.get("path") or []) for s in sources or []
            if s.get("className", "LocalScript") in ("LocalScript", "ModuleScript")
            and RUNTIME_UI.search(strip_luau_comments(s.get("source") or ""))]


def console_ready(put, T, sources, play):
    """consoleReady. The static controller audit first; when it found no ScreenGui under StarterGui and client scripts
    build one at run time, the audit's play half judges the play client's PlayerGui instead. A play run that is not
    possible (play false: a dry run; the lock busy, no client) answers unknown with why - never a pass, and never a
    fail for a UI that was not looked at."""
    audit = T.TOOLS["rbx_controller_audit"]

    def brief(a):
        return {k: a.get(k) for k in ("ok", "error", "unreachable", "notSelectable", "deadLinks", "editTree") if k in a}
    a = audit()
    empty = not a.get("ok") and "no ScreenGui under" in str(a.get("error") or "")
    makers = runtime_ui_scripts(sources) if empty else []
    if not makers:
        put("consoleReady", "pass" if a.get("ok") else "fail", {"consoleEnabled": True, "audit": brief(a)},
            None if a.get("ok") else "Console is on and rbx_controller_audit fails",
            None if a.get("ok") else "fix what the audit names, or turn Console off")
        return
    read = {"consoleEnabled": True, "runtimeUi": makers[:20], "audit": brief(a)}
    if not play:
        put("consoleReady", "unknown", dict(read, unknown="runtime UI"),
            "unknown: runtime UI - StarterGui holds no ScreenGui and %d client script(s) build one at run time; a dry "
            "run starts no play test, so the play client's PlayerGui was not audited" % len(makers),
            "run the preflight without dry_run, or rbx_controller_audit(play=true); or the person accepts consoleReady")
        return
    p = audit(play=True, play_wait=8)
    pr = p.get("play") or {}
    gui = pr.get("playerGui") or {}
    read["playAudit"] = dict(brief(p), uiSettled=pr.get("uiSettled"), playerGui={"screenGuis": len(gui.get("screenGuis") or []),
                                                  "elements": gui.get("elements"), "truncated": gui.get("truncated")})
    if pr.get("uiSettled") is False and gui.get("screenGuis"):
        # the game's UI was still being built when the probe's settle deadline passed: what it read is a partial tree,
        # so it is neither a pass nor a fail (found 2026-11-22: one run of three on an unchanged place failed this way)
        put("consoleReady", "unknown", dict(read, unknown="UI not built yet"),
            "unknown: UI not built yet - the play client's UI was still changing after %s s, so it was not judged" % (
                round(pr.get("uiSettleSeconds") or 0, 1)),
            "run the preflight again; or the person accepts consoleReady")
    elif p.get("ok"):
        put("consoleReady", "pass", read)
    elif pr.get("error") or not pr:
        put("consoleReady", "unknown", dict(read, unknown="runtime UI"),
            "unknown: runtime UI - the UI is built at run time and the play half could not run: %s" % (
                pr.get("error") or p.get("error")),
            "free the play lock or reconnect the Studio and run the preflight again; or the person accepts consoleReady")
    else:
        put("consoleReady", "fail", read, "Console is on and the play client's PlayerGui fails rbx_controller_audit",
            "fix what the audit names, or turn Console off")


def _norm_source(s):
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


SCAN_PAGE_CALLS = 400        # script_scan calls one paged read may make before it stops and says the scan was cut


def scan_all_sources(T, root=None, max_calls=SCAN_PAGE_CALLS):
    """Every script's source under root (a path, [] for the whole place) from the connected Studio, past the plugin's
    4 MB cap on one script_scan answer. A scan that comes back truncated is paged by subtree: the plugin sends sources in
    full-name order and stops before the script that passes 4 MB, so the root's own source (it sorts first) is kept from
    that answer and each child (OPS.tree, depth 1) is scanned on its own, recursively, until every answer is whole.
    -> {ok, sources, scanned, calls, paged, truncated, cut}: truncated is true only when a part could still not be read
    whole - a single script over 4 MB, a parent with two children of one name (a path names only the first), a node with
    more children than one tree answer lists, or max_calls reached - and cut names each such path. At the top level a
    service whose scan errors is skipped, as the plugin's whole-place scan skips one whose descendants it cannot read
    (skipped names them); deeper, an error is a cut. ok false only when the first scan fails."""
    calls = {"n": 0}
    sources, cut, skipped = {}, [], []

    def scan(parts):
        calls["n"] += 1
        return T._rbx("script_scan", root=list(parts))

    first = scan(root or [])
    if not first.get("ok"):
        return dict(first, ok=False)
    scanned = first.get("scanned")

    def take(ans):
        for x in ans.get("sources") or []:
            if isinstance(x, dict) and isinstance(x.get("path"), list):
                sources[tuple(x["path"])] = x

    def walk(parts, ans):
        if ans.get("truncated") is not True:
            take(ans)
            return
        # the node itself sorts before its descendants, so a node that is a script is in the cut answer unless it alone
        # passes the cap
        for x in ans.get("sources") or []:
            if isinstance(x, dict) and tuple(x.get("path") or ()) == tuple(parts):
                sources[tuple(parts)] = x
        if calls["n"] >= max_calls:
            cut.append({"path": ".".join(parts) or "game", "why": "the paged scan reached %d calls" % max_calls})
            return
        t = T._rbx("tree", root=list(parts), depth=1, limit=5000)
        node = (t or {}).get("root") if (t or {}).get("ok") else None
        kids = (node or {}).get("children")
        if not isinstance(kids, list):
            cut.append({"path": ".".join(parts) or "game", "why": "its children could not be listed: %s" % (
                (t or {}).get("error") or "no tree answer")})
            return
        if node.get("truncated") or (isinstance(node.get("childCount"), int) and node["childCount"] > len(kids)):
            cut.append({"path": ".".join(parts) or "game", "why": "it has more children than one listing holds"})
            return
        names = [k.get("name") for k in kids if isinstance(k, dict)]
        dup = sorted({n for n in names if names.count(n) > 1})
        if not kids:
            cut.append({"path": ".".join(parts) or "game", "why": "a single script holds more than 4 MB of source"})
            return
        if dup:
            cut.append({"path": ".".join(parts) or "game", "why": "two or more children share a name (%s), and a path "
                        "names only the first" % ", ".join(str(n) for n in dup[:5])})
        for k in kids:
            if not isinstance(k, dict) or k.get("name") in dup:
                continue
            if not parts and k.get("className") == "CoreGui":
                continue                     # the plugin's whole-place scan never reads CoreGui
            sub = list(parts) + [k.get("name")]
            if calls["n"] >= max_calls:
                cut.append({"path": ".".join(sub), "why": "the paged scan reached %d calls" % max_calls})
                continue
            a = scan(sub)
            if not a.get("ok"):
                if not parts:
                    skipped.append(".".join(sub))
                else:
                    cut.append({"path": ".".join(sub), "why": "its scan failed: %s" % (a.get("error") or "no answer")})
                continue
            walk(sub, a)

    walk(list(root or []), first)
    # the first answer counts every script under root even when cut (the plugin counts before it sends): once that many
    # were read whole, a part left out (two of Studio's own services share a name, FilteredSelection, live 2026-11-26)
    # held none of them, so it is not a cut
    spared = []
    if cut and isinstance(scanned, int) and not isinstance(scanned, bool) and len(sources) >= scanned:
        spared, cut = cut, []
    return {"ok": True, "sources": [sources[k] for k in sorted(sources, key=".".join)], "scanned": scanned,
            "calls": calls["n"], "paged": calls["n"] > 1, "truncated": bool(cut), "cut": cut[:50],
            **({"skipped": skipped} if skipped else {}),
            **({"notNeeded": spared[:50], "notNeededWhy": "every script the first scan counted was read whole"}
               if spared else {})}


def file_current(put, T, file_sources):
    """fileCurrent: every script in the saved file against the same script in the open Studio, by path and source (line
    endings set aside). A script that differs, or is in one and not the other, fails: the file is older than the place
    the other checks read, so the person saves first. Studio's scan stops at 4 MB of source a call, so the read is paged
    by subtree (scan_all_sources); only a part that still cannot be read whole (a single script over 4 MB, two children
    of one name) leaves the check unread, never pass, and names that part. Properties outside scripts (an attribute
    flag, a part) are not compared, and the read says so."""
    if file_sources is None:
        put("fileCurrent", "unread", None, "the file's scripts could not be read, so the file was not compared with the "
                                           "open place", "check the file opens in Studio, save it, run the preflight again")
        return
    s = scan_all_sources(T)
    if not s.get("ok"):
        put("fileCurrent", "unread", None, "the open place's scripts could not be read: %s" % (s.get("error") or "no answer"),
            "reconnect the Studio and run the preflight again")
        return
    in_file = {tuple(x.get("path") or ()): _norm_source(x.get("source")) for x in file_sources if isinstance(x, dict)}
    in_studio = {tuple(x.get("path") or ()): _norm_source(x.get("source")) for x in (s.get("sources") or [])
                 if isinstance(x, dict)}
    cut = s.get("truncated") is True
    differ = sorted(".".join(p) for p in in_file if p in in_studio and in_file[p] != in_studio[p])
    only_studio = sorted(".".join(p) for p in in_studio if p not in in_file)
    only_file = [] if cut else sorted(".".join(p) for p in in_file if p not in in_studio)
    read = {"compared": len([p for p in in_file if p in in_studio]), "fileScripts": len(in_file),
            "studioScripts": s.get("scanned"), "differ": differ[:50], "onlyInStudio": only_studio[:50],
            "onlyInFile": only_file[:50], "truncated": cut, "scanCalls": s.get("calls"), "paged": s.get("paged"),
            "notCompared": "properties outside scripts (attributes, values, parts) are not compared"}
    if cut:
        read["cutAt"] = s.get("cut")
    if not in_file and not in_studio:
        put("fileCurrent", "unread", read, "neither the file nor the open place holds a script: nothing compared is not "
                                           "current")
    elif differ or only_studio or only_file:
        put("fileCurrent", "fail", read, "the saved file is not the place open in Studio: %d script(s) differ, %d only in "
            "Studio, %d only in the file. The other checks read the open place; the file is what would go live" % (
                len(differ), len(only_studio), len(only_file)),
            "save the place in Studio (File > Save, or rbx_place_save), then run the preflight again")
    elif cut:
        put("fileCurrent", "unread", read, "part of the open place could not be read whole even paged (%s): the %d "
            "script(s) read match the file and the rest were not compared" % (
                "; ".join("%s: %s" % (c["path"], c["why"]) for c in (s.get("cut") or [])[:3]) or "truncated",
                read["compared"]),
            "the person confirms the place was saved after the last change, and names fileCurrent in accept")
    else:
        put("fileCurrent", "pass", read)


STACK_ROOTS = ("Workspace", "ReplicatedStorage", "ServerStorage")


def made_by_scripts(name, sources):
    """The scripts (dotted paths) whose source names `name` as a string literal (a script that makes it at start:
    Instance.new then .Name = "name", or a FindFirstChild before it)."""
    import re as _re
    pat = _re.compile(r"""["']%s["']""" % _re.escape(name))
    out = []
    for s_ in sources or []:
        if isinstance(s_, dict) and pat.search(str(s_.get("source") or "")):
            p_ = s_.get("path")
            out.append(".".join(p_) if isinstance(p_, (list, tuple)) else str(p_))
    return out


def duplicate_top_level(file, under=STACK_ROOTS, at_least=2, sources=None):
    """The top-level children of each root in `under` in a saved place whose name is held at_least times or more, each
    with the scripts that name it (madeBy: a server that makes it at start stacks one copy per Run-mode run). ->
    {stacked: [{under, name, count, classes, madeBy}], children} or {error}."""
    under = (under,) if isinstance(under, str) else tuple(under)
    import os
    from mif_roblox import place_manifest as PM
    try:
        with open(str(file), "rb") as fh:
            blob = fh.read()
        _fmt, rows, _u = PM.read(blob, os.path.splitext(str(file))[1].lower())
    except Exception as exc:  # noqa: BLE001 - unreadable is unread, never a pass
        return {"error": "the file could not be read for stacked copies (%s: %s)" % (type(exc).__name__, exc)}
    counts, classes = {}, {}
    for r in rows:
        p = r.get("path") or []
        if len(p) == 2 and p[0] in under:
            key = (p[0], p[1])
            counts[key] = counts.get(key, 0) + 1
            classes.setdefault(key, set()).add(r.get("className"))
    stacked = [{"under": k[0], "name": k[1], "count": c, "classes": sorted(x for x in classes[k] if x),
                "madeBy": made_by_scripts(k[1], sources)[:5]} for k, c in sorted(counts.items()) if c >= at_least]
    return {"stacked": stacked, "children": sum(counts.values())}


def preflight(file=None, place_id=None, selftest_flag=None, config_module=None, id_keys=None, odds_products=None,
              universe_id=None, play=True, test_switches=None):
    """Every check a first publish needed, each answered with what it read. Never sets anything. -> {ok, checks
    {name: {status pass|fail|warn|unread|ask|unknown, read, why?, step?}}, failed [], toAsk []}. play false (a dry
    run) never starts a play test: a check that needs one answers unknown, which is in toAsk."""
    T = _T()
    checks = {}

    def put(name, status, read=None, why=None, step=None):
        checks[name] = {k: v for k, v in (("status", status), ("read", read), ("why", why), ("step", step)) if v is not None}

    st = T.rbx_status()
    plug = st.get("plugin") or {}
    # With no Studio and a saved file, the checks read the file itself (spec "Extend rbx_place_publish: publish a saved
    # file with no Studio connected"): it is what ships. A check that only an open Studio answers is unread, saying why,
    # never passed (notFromFile).
    file_only = not st.get("ok") or not plug.get("connected", True)
    if file_only and not file:
        return {"ok": False, "ran": False, "refusedRule": "no-studio",
                "error": "the preflight reads the place open in Studio, or a saved file: no Studio is connected (%s) and "
                         "no file was given" % st.get("error")}
    not_from_file = []

    def cannot(name, why, step=None):
        not_from_file.append(name)
        put(name, "unread", {"from": "file"}, why, step)
    place_name = plug.get("placeName")
    if file and file_only:
        import os
        if not os.path.isfile(str(file)):
            return {"ok": False, "ran": False, "error": "file is an existing .rbxl or .rbxlx: %r" % (file,)}
    if file and not file_only:
        import os
        same = os.path.basename(str(file)).lower() == str(place_name or "").lower()
        put("placeOpen", "pass" if same else "fail", {"studio": place_name, "file": os.path.basename(str(file))},
            None if same else "the Studio the bridge reads holds %s, not the file being published" % place_name,
            None if same else "open the file in the connected Studio, or publish the file that is open")
    # 1. the self-test flag
    if selftest_flag and file_only and ("@" in selftest_flag or ":" in selftest_flag):
        cannot("selfTestOff", "%s is %s: read only by the open Studio, not from a file" % (
            selftest_flag, "an attribute" if "@" in selftest_flag else "a ModuleScript's value"),
            "open the file in Studio and run the preflight there, or keep the flag in a BoolValue and name its path")
    elif selftest_flag:
        val, how, err = (_flag_from_file(file, selftest_flag) if file_only else _flag_value(T, selftest_flag))
        if err:
            put("selfTestOff", "unread", None, err)
        elif val is None:
            # GetAttribute of a misspelled attribute is nil, and so is a module key that is not there: no value read
            # is not "off".
            put("selfTestOff", "unread", {"value": None, "from": how},
                "%s read no value (no such attribute, key or Value): a flag that was not read is not off" % selftest_flag,
                "check the name of the flag, then run the preflight again")
        else:
            put("selfTestOff", "pass" if val in (False, None, 0) else "fail", {"value": val, "from": how},
                None if val in (False, None, 0) else "the self-test would run on every joining player",
                None if val in (False, None, 0) else "set %s off in the saved place" % selftest_flag)
    else:
        put("selfTestOff", "ask", why="name the self-test flag (selftest_flag) so it is read back off, or say there is none")
    # 2. config ids
    if config_module and id_keys and file_only:
        cannot("idsSet", "a config module's values are read by running it in Studio, not from a file",
               "open the file in Studio and run the preflight there")
    elif config_module and id_keys:
        unset, bad, read = [], [], {}
        table, err = module_value(T, config_module)
        for k in (id_keys if not err else []):
            # an id key is a dotted path; '*' takes every entry of a list there (a store's passes or products)
            for path, v in id_leaves(table, k):
                read[path] = v
                if v is None:
                    unset.append(path)
                elif not _id_ok(v):
                    bad.append(path)
        if err:
            put("idsSet", "unread", None, err)
        else:
            ok = not unset and not bad
            put("idsSet", "pass" if ok else "fail", read,
                None if ok else "unset: %s; not an id: %s" % (", ".join(unset) or "none", ", ".join(bad) or "none"),
                None if ok else "the person makes each pass, product and badge in the Creator Dashboard and sets its id")
    else:
        put("idsSet", "ask", why="name the config ModuleScript and the id keys (config_module, id_keys) to read them")
    # 3. secrets in scripts. With the saved file, its scripts are read from the file itself, whole: it is what ships,
    # and Studio's script scan stops early (truncated) on very large ModuleScripts (found 2026-11-22: a word-list
    # module left noSecretsInScripts unread, while all 96 scripts, 12.5 M characters, read from the .rbxl held none).
    # Studio is read only when no file is given, or the file cannot be read (fileError says why).
    r, file_error = None, None
    if file:
        import os
        from mif_roblox import save_audit as SA
        try:
            fsrc, origin = SA.sources_from_file(str(file))
            r = {"ok": True, "scanned": len(fsrc), "sources": fsrc, "truncated": False, "from": "file:" + origin}
        except Exception as exc:  # noqa: BLE001 - an unreadable file falls back to Studio, saying why
            file_error = "%s could not be read as a place (%s: %s)" % (os.path.basename(str(file)), type(exc).__name__, exc)
    if r is None and file_only:
        r = {"ok": False, "scanned": 0, "sources": [], "from": "file",
             "error": file_error or "the file's scripts could not be read, and no Studio is connected"}
    if r is None:
        r = dict(T._rbx("script_scan", root=[]))
        r["from"] = "studio"
        if file_error:
            r["fileError"] = file_error
    sources = r.get("sources") or []
    scan_ok = bool(r.get("ok"))
    if not r.get("ok") or not r.get("scanned"):
        put("noSecretsInScripts", "unread", None, r.get("error") or "no scripts were scanned: nothing scanned is not clean")
    else:
        from mif_roblox import opencloud as OC
        own, _e = OC.read_key()
        found = scan_secrets(sources, own)
        cut = r.get("truncated") is True
        read = {"scanned": r.get("scanned"), "found": found, "truncated": cut, "from": r.get("from")}
        if r.get("fileError"):
            read["fileError"] = r["fileError"]
        if found:
            put("noSecretsInScripts", "fail", read, "a key in a script ships to every server that runs it",
                "move it into the experience's Secrets (Creator Hub) and read it with HttpService:GetSecret")
        elif cut:
            put("noSecretsInScripts", "unread", read, "the script scan was cut short (truncated): the scripts it did not "
                                                      "send were not read, so none found is not clean")
        else:
            put("noSecretsInScripts", "pass", read)
    # 3b. the save layer never falls back to memory on a live server
    if not r.get("ok") or not r.get("scanned"):
        put("saveLayer", "unread", None, r.get("error") or "no scripts were scanned: nothing scanned is not clean")
    else:
        from mif_roblox import save_audit as SA
        sa = SA.answer([s for s in sources if isinstance(s, dict) and isinstance(s.get("source"), str)], "studio",
                       truncated=r.get("truncated", False), checks=["save"])
        bad = sa.get("findings") or []
        put("saveLayer", "fail" if bad else ("pass" if sa.get("ran") else "unread"),
            {"scanned": sa.get("scanned"), "dataStoreScripts": sa.get("dataStoreScripts"), "from": r.get("from"),
             "findings": [{k: f.get(k) for k in ("script", "kind", "line", "lines", "why")} for f in bad]},
            "a failed DataStore call would run a live server on memory: empty loads, overwritten scores, receipts "
            "granted into RAM" if bad else (None if sa.get("ran") else sa.get("error")),
            SA.FIX if bad else None)
    # 3c. a game pass is granted only on Roblox's ownership answer
    if not r.get("ok") or not r.get("scanned"):
        put("passGrant", "unread", None, r.get("error") or "no scripts were scanned: nothing scanned is not clean")
    else:
        from mif_roblox import save_audit as SA
        pa = SA.answer([s for s in sources if isinstance(s, dict) and isinstance(s.get("source"), str)], "studio",
                       truncated=r.get("truncated", False), checks=["passes"])
        bad = pa.get("findings") or []
        put("passGrant", "fail" if bad else ("pass" if pa.get("ran") else "unread"),
            {"scanned": pa.get("scanned"), "passScripts": pa.get("passScripts"), "from": r.get("from"),
             "findings": [{k: f.get(k) for k in ("script", "kind", "line", "lines", "why")} for f in bad]},
            "a game pass is granted without UserOwnsGamePassAsync: PromptGamePassPurchaseFinished alone, or a pass "
            "restored from the save on join, is not proof of ownership" if bad else
            (None if pa.get("ran") else pa.get("error")),
            SA.PASS_FIX if bad else None)
    # 4. paid random items
    if odds_products is None:
        put("paidRandomItems", "ask", why="name the Robux items that change odds (odds_products), or [] for none: the "
                                         "maturity questionnaire's paid-random-items answer follows from it")
    elif not odds_products:
        put("paidRandomItems", "pass", {"oddsProducts": []})
    else:
        hidden = any(POLICY_RESTRICTED in strip_luau_comments(s.get("source") or "") for s in sources)
        put("paidRandomItems", "pass" if hidden else "fail",
            {"oddsProducts": odds_products, "questionnaire": "paid random items: yes", "readsPolicy": hidden},
            None if hidden else "Robux items that change odds must be hidden where PolicyService says "
                                "ArePaidRandomItemsRestricted, and no script reads it",
            "answer yes to paid random items on the maturity questionnaire" + ("" if hidden else
                                                                                 "; hide those items per player"))
    # 5. devices, visibility and the game page, from Roblox
    if universe_id is None and place_id:
        universe_id, _uerr = universe_of_place(place_id)
    # 4b. every secret a script reads is one the experience holds: names only, never content
    if not r.get("ok") or not r.get("scanned"):
        put("secretsPresent", "unread", None, r.get("error") or "no scripts were scanned: nothing scanned is not clean")
    else:
        reads, unresolved = scan_secret_reads(sources)
        names = sorted({x["name"] for x in reads})
        seen = {"reads": len(reads) + len(unresolved), "names": names, "unresolved": unresolved}
        computed = ("%d read(s) of a computed name could not be checked: the person confirms each" % len(unresolved)
                    if unresolved else None)
        if not reads:
            put("secretsPresent", "warn" if unresolved else "pass", seen, computed)
        elif not universe_id:
            put("secretsPresent", "ask", seen, "give place_id or universe_id (a published place) to read the names of the "
                                               "experience's secrets: scripts read %s" % ", ".join(names))
        else:
            got = _OC().secret_names(universe_id)
            if not got.get("ok"):
                put("secretsPresent", "unread", seen, got.get("error"), "; ".join(got.get("steps") or []) or None)
            else:
                held = set(got.get("names") or [])
                missing = [x for x in reads if x["name"] not in held]
                gone = sorted({x["name"] for x in missing})
                put("secretsPresent", "fail" if missing else ("warn" if unresolved else "pass"),
                    dict(seen, held=len(held), missing=missing),
                    "the experience holds no secret named %s: HttpService:GetSecret fails on a live server"
                    % ", ".join(gone) if missing else computed,
                    "Creator Hub > the experience > Secrets: the person adds %s, or the script reads a name that is "
                    "there" % ", ".join(gone) if missing else None)
    if universe_id:
        u = rbx_universe_settings(universe_id=universe_id)
        uni = u.get("universe") or {}
        if not u.get("ok"):
            put("devices", "unread", None, u.get("error"))
        else:
            put("visibility", "pass" if uni.get("visibility") != "PUBLIC" else "warn", uni.get("visibility"),
                None if uni.get("visibility") != "PUBLIC" else "already public: a first publish goes private first")
            put("vrOff", "pass" if not uni.get("vrEnabled") else "warn", uni.get("vrEnabled"),
                None if not uni.get("vrEnabled") else "VR is on: only if VR was built and tested")
            if uni.get("consoleEnabled") and file_only:
                cannot("consoleReady", "the console check reads the open place's UI and plays it: not from a file",
                       "open the file in Studio and run the preflight there")
            elif uni.get("consoleEnabled"):
                console_ready(put, T, sources if scan_ok else None, play)
            else:
                put("consoleReady", "pass", {"consoleEnabled": False})
            put("ageGate", "warn", uni.get("ageRating"), "a new experience is limited to players 16 and up until 100 "
                "highly engaged players or the refundable 1,000 Robux Kids & Select fee")
        media = game_media(universe_id)
        if media.get("ok"):
            n = media["count"]
            private = uni.get("visibility") != "PUBLIC"
            put("gamePageThumbnails", "pass" if n else ("warn" if private else "fail"), {"count": n},
                None if n else "the experience's own page lists no media%s: thumbnails filled only on the Home Page "
                               "list leave it on the placeholder" % (
                                   " (it is private, and whether a private page lists its media is unmeasured)"
                                   if private else ""),
                None if n else "Creator Hub > the experience > Thumbnails: add them to the Experience Detail Page list")
        else:
            put("gamePageThumbnails", "unread", None, media.get("error"))
    else:
        put("devices", "ask", why="give place_id or universe_id (a published place) to read devices, visibility and the "
                                  "game page")
    # 6. nothing of MifBridge's own ships (found 2026-11-22: a published version carried the empty Workspace.MifScratch
    # folder the bridge made). The working folder fails; the ServerStorage markers a launch copy carries are inert there
    # (ServerStorage never replicates) and are warned.
    # With the saved file, the file is read, not the open Studio: the plugin makes Workspace.MifScratch again on its next
    # request, so after a Destroy and a save the Studio still shows it while the file - what ships - does not (found
    # 2026-11-22: a publish could never pass). Studio is read only without a file, or when the file cannot be read.
    file_paths, bridge_from = None, "studio"
    if file:
        from mif_roblox import save_audit as SA
        try:
            file_paths = {tuple(p) for p in SA.instance_paths_from_file(str(file))}
            bridge_from = "file"
        except Exception:  # noqa: BLE001 - an unreadable file falls back to Studio, as the script checks do
            file_paths = None

    def present(path):
        if file_paths is not None:
            return tuple(path) in file_paths, None
        if file_only:
            return None, "could not be read: the file did not read and no Studio is connected"
        q = T._rbx("resolve", target=path)
        if q.get("ok"):
            return True, None
        return (False, None) if q.get("refusedRule") == "not-found" else (None, q.get("error") or "could not be read")
    scratch, s_err = present(["Workspace", "MifScratch"])
    if scratch is None:
        put("noBridgeObjects", "unread", None, "Workspace.MifScratch %s" % s_err)
    else:
        put("noBridgeObjects", "fail" if scratch else "pass", {"Workspace.MifScratch": scratch, "from": bridge_from},
            "MifBridge's working folder Workspace.MifScratch would ship in the published place" if scratch else None,
            "delete Workspace.MifScratch (rbx_call Destroy with confirm true), save the place, run the preflight again"
            if scratch else None)
    marks = [n for n in ("MifBridgePort", "MifBridgeLaunched") if present(["ServerStorage", n])[0]]
    if marks:
        put("bridgeMarkers", "warn", {"ServerStorage": marks, "from": bridge_from},
            "MifBridge's markers ship in ServerStorage: inert there (never replicated), and removing them in an open "
            "Studio changes which daemon it reaches and whether the agent may raise its mode")
    put("teamCreateOff", "ask", why="Team Create's state is not readable by a plugin (CollaboratorsService is "
                                    "RobloxScriptSecurity): the person checks File > Collaborate is off")
    # 7. the file is what ships, and the checks that read Studio (selfTestOff, idsSet, consoleReady) read the open place:
    # a fix made in Studio and not saved passed them while the stale file went live (found 2026-11-22).
    if file and checks.get("placeOpen", {}).get("status") == "pass":
        file_current(put, T, r.get("sources") if str(r.get("from") or "").startswith("file:") else None)
    # 8. the self-test's other switches (found 2026-11-26: a place went live with its hazards, enemy attacks and the
    # server's movement check switched off, set as Workspace attributes by its self-test in a Run-mode session, while
    # selfTestOff read its one flag off): read in the file that ships
    if file:
        from mif_roblox import test_switches as TS
        skip = None
        if isinstance(selftest_flag, str) and "@" in selftest_flag:
            fp, fa = selftest_flag.rsplit("@", 1)
            skip = ([x for x in fp.split(".") if x], fa)
        if str(r.get("from") or "").startswith("file:"):
            st_, rd, why, step = TS.check(file, r.get("sources"), test_switches, skip)
        else:
            st_, rd, why, step = ("unread", None, "the file's scripts could not be read: %s" % (
                r.get("fileError") or "no answer"), None)
        put("testSwitchesOff", st_, rd, why, step)
    elif test_switches:
        put("testSwitchesOff", "unread", None, "test_switches are read in the saved file: name file")
    # 9. stacked copies (found 2026-11-26: a place shipped 46 copies of three folders its server makes at start, one per
    # Run-mode run, since Stop keeps what a run made): top-level names under Workspace held more than once, warned
    if file:
        dup = duplicate_top_level(file, sources=r.get("sources") if str(r.get("from") or "").startswith("file:") else None)
        made = [d for d in dup.get("stacked") or [] if d.get("madeBy")]
        if dup.get("error"):
            put("noStackedCopies", "unread", None, dup["error"])
        elif made:
            # a name a script in the place makes, held more than once: the server's start-up copies stacked (fail)
            put("noStackedCopies", "fail", dup, "%d name(s) a script in the place makes are held more than once in the "
                "file - stacked start-up copies that would go live: %s" % (len(made), ", ".join(
                    "%s.%s x%d (named by %s)" % (d["under"], d["name"], d["count"], d["madeBy"][0]) for d in made[:6])),
                "remove the extra copies (rbx_destroy all=true, then let the game make them again), make the script "
                "replace its own copy at start, and save; rbx_run_mode stop's attributes.instances.added lists what a run "
                "made")
        elif dup["stacked"]:
            put("noStackedCopies", "warn", dup, "%d name(s) are held more than once in the file: %s" % (
                len(dup["stacked"]), ", ".join("%s.%s x%d" % (d["under"], d["name"], d["count"])
                                               for d in dup["stacked"][:6])),
                "if a script makes these at start and a Run-mode run left them, remove the extra copies (rbx_destroy "
                "all=true, then let the game make them again) and save; rbx_run_mode stop's attributes.instances.added "
                "lists what a run made")
        else:
            put("noStackedCopies", "pass", dup)
    failed = sorted(n for n, c in checks.items() if c["status"] in ("fail", "unread"))
    # unknown is not a pass: like ask it blocks a publish until the person names it in accept
    ask = sorted(n for n, c in checks.items() if c["status"] in ("ask", "unknown"))
    out = {"ok": not failed, "ran": True, "checks": checks, "failed": failed, "toAsk": ask,
           "note": "read only: nothing was set or published"}
    if file_only:
        out.update(fromFile=True, notFromFile=not_from_file,
                   fileNote="no Studio is connected: every check read the saved file; placeOpen and fileCurrent "
                            "compare a Studio with the file and were not run")
    return out


def _flag_from_file(file, flag):
    """A self-test flag that is a ValueBase path, read from the saved file. -> (value, how, error)."""
    from mif_roblox import save_audit as SA
    from mif_roblox import values as V
    parts = V.split_path(flag)
    if not parts:
        return None, None, "%s is not a path" % flag
    try:
        found, val, why = SA.value_from_file(str(file), parts)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is unread, with its reason
        return None, None, "the file could not be read as a place: %s: %s" % (type(exc).__name__, exc)
    if not found or why:
        return None, None, why
    return val, "Value of %s in the saved file" % flag, None


def game_media(universe_id):
    """The experience's own page media: games.roblox.com GET /v2/games/{id}/media, a public read (no key). ->
    {ok, count, items} or {ok False, error}."""
    OC = _OC()
    try:
        st, _h, raw = OC.REQUEST("GET", "https://games.roblox.com/v2/games/%d/media" % int(universe_id),
                                 {"Accept": "application/json"}, None, 30)
    except OSError as exc:
        return {"ok": False, "error": "the game page media did not read: %s" % exc}
    body = OC._json(raw)
    data = body.get("data") if isinstance(body, dict) else None
    if st != 200 or not isinstance(data, list):
        return {"ok": False, "error": "GET /v2/games/%s/media answered %s: %s" % (universe_id, st, json.dumps(body)[:200])}
    return {"ok": True, "count": len(data), "items": [{k: d.get(k) for k in ("assetType", "imageId", "approved")}
                                                      for d in data if isinstance(d, dict)]}


# ---- game passes, never on sale (Roblox's Game Passes API, BETA: create.roblox.com/docs/cloud/reference/features/
# game-passes, read 2026-11-22). POST /game-passes/v1/universes/{u}/game-passes is a multipart form (name required;
# description, imageFile, isForSale, price optional) under game-pass:write; GET .../game-passes/creator (paged) and
# .../game-passes/{id}/creator answer GamePassConfigV2 {gamePassId, name, description, isForSale, iconAssetId,
# priceInformation} under game-pass:read. This tool makes a pass with isForSale false and no price and no image, so it
# earns nothing and uploads nothing; putting one on sale is the person's step in Creator Hub.
PASSES = "/game-passes/v1/universes/%d/game-passes"
SCOPE_PASS_READ = ("game-pass", "read")
SCOPE_PASS_WRITE = ("game-pass", "write")
PASS_PAGES = 20               # pages of 100 read before a name check gives up (a name not found in a cut list is unknown)


def _pass_marker(universe_id, name):
    """The file a create claims before its POST (O_CREAT|O_EXCL), so no retry, crash or second caller sends it twice."""
    import hashlib
    import os
    from mif_roblox import daemon as D
    d = os.path.join(D.home(), "game_pass_creates")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%d_%s.json" % (universe_id, hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]))


def _all_passes(key, universe_id):
    """(the universe's game passes, None) or (None, error): every page of GET .../game-passes/creator."""
    OC = _OC()
    out, token = [], None
    for _ in range(PASS_PAGES):
        path = (PASSES % universe_id) + "/creator?pageSize=100" + (
            "&pageToken=%s" % urllib.parse.quote(str(token), safe="") if token else "")
        st, body, _w = OC._call("GET", path, key, tries=3)
        body = OC.redact(body, key)
        if st != 200 or not isinstance(body, dict) or not isinstance(body.get("gamePasses", []), list):
            return None, "GET %s answered %s: %s" % (path, st, json.dumps(body)[:300])
        out += [p for p in body.get("gamePasses") or [] if isinstance(p, dict)]
        token = body.get("nextPageToken")
        if not token:
            return out, None
    return None, "more than %d pages of game passes: the name check could not see them all" % PASS_PAGES


def _pass_view(p):
    return {k: p.get(k) for k in ("gamePassId", "name", "description", "isForSale", "iconAssetId", "priceInformation")}


def _form(fields):
    """A multipart/form-data body of text fields only (no file part: nothing is uploaded)."""
    import uuid
    b = "MifBridge" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode("utf-8"))
    return b"".join(parts) + ("--%s--\r\n" % b).encode("utf-8"), "multipart/form-data; boundary=%s" % b


def rbx_game_pass(action: str, universe_id: int, name: str = None, description: str = "", game_pass_id: int = None,
                  confirm: bool = False, dry_run: bool = True, allow_ids: list = None) -> dict:
    "Roblox Open Cloud: an experience's game passes with the person's own key (read from their key file, never shown), Roblox's Game Passes API (BETA). action list: every pass of universe_id (GET /game-passes/v1/universes/{u}/game-passes/creator, all pages) with gamePassId, name, isForSale, price; get: one pass by game_pass_id (.../game-passes/{id}/creator). create: one pass named name that is NEVER for sale - isForSale false, no price, no image, so it earns nothing and uploads nothing (putting a pass on sale is the person's step in Creator Hub). A create is a dry run by default (the exact form it would POST); it is sent only with confirm true and dry_run false, the MifBridge panel in full mode, the key holding game-pass:write over the universe, and the universe on the person's write-target list (roblox_write_targets.json) or in allow_ids (refusedRule write-target). It is refused exists, with that pass, when the universe already has a pass of that name; before the POST it claims a marker file for (universe, name), so a retry, a timeout or a second caller never makes a second pass (refusedRule already-sent: list the passes to see whether the first one landed); a definite refusal (a 4xx other than 408: nothing was made) removes the marker, and only no answer, 408 or a 5xx keeps it. The POST is sent once, never retried. Read back: the new id's pass by get - its name, and isForSale false."
    OC = _OC()
    if action not in ("list", "get", "create"):
        return _no("action is list, get or create")
    if not _pos_int(universe_id):
        return _no("universe_id is a positive whole number")
    if action == "get" and not _pos_int(game_pass_id):
        return _no("get needs game_pass_id, a positive whole number")
    if action == "create":
        if not isinstance(name, str) or not name.strip() or len(name) > 50:
            return _no("create needs name, 1-50 characters")
        if not isinstance(description, str) or len(description) > 1000:
            return _no("description is text, up to 1000 characters")
    if not isinstance(confirm, bool) or not isinstance(dry_run, bool):
        return _no("confirm and dry_run are true or false")
    key, err = OC.read_key()
    if err:
        return err
    try:
        if action == "get":
            path = (PASSES % universe_id) + "/%d/creator" % game_pass_id
            st, body, _w = OC._call("GET", path, key, tries=3)
            body = OC.redact(body, key)
            if st != 200 or not isinstance(body, dict):
                return {"ok": False, "ran": True, "status": st, "answer": body,
                        "error": "GET %s answered %s: %s" % (path, st, json.dumps(body)[:300])}
            return {"ok": True, "ran": True, "verified": True, "universeId": universe_id, "pass": _pass_view(body)}
        passes, perr = _all_passes(key, universe_id)
        if perr:
            return {"ok": False, "ran": True, "error": perr}
        if action == "list":
            return {"ok": True, "ran": True, "verified": True, "universeId": universe_id, "count": len(passes),
                    "passes": [_pass_view(p) for p in passes]}
        same = [p for p in passes if p.get("name") == name]
        if same:
            return {"ok": False, "ran": True, "sent": False, "refusedRule": "exists", "pass": _pass_view(same[0]),
                    "error": "universe %d already has a game pass named %r (id %s): nothing was sent" % (
                        universe_id, name, same[0].get("gamePassId"))}
        import os
        marker = _pass_marker(universe_id, name)
        fields = {"name": name, "description": description, "isForSale": "false"}
        out = {"universeId": universe_id, "request": {"method": "POST", "path": PASSES % universe_id,
                                                      "contentType": "multipart/form-data", "fields": fields}}
        sc = scope_check(SCOPE_PASS_WRITE, universe_id)
        out["scope"] = sc
        tgate = OC.write_target_check(universe_id, False, allow_ids)
        if tgate is not None and "notAllowed" not in tgate:   # a malformed allow_ids: its own refusal, nothing listed
            return dict(out, **tgate)
        out["writeTarget"] = "listed" if tgate is None else {k: tgate[k] for k in ("notAllowed", "targetsFile", "error")}
        if os.path.exists(marker):
            return dict(out, ok=False, ran=False, sent=False, refusedRule="already-sent", marker=marker,
                        error="a create of %r on universe %d was already sent from this machine and no pass of that name "
                              "is listed now: it may still be landing or it failed. Nothing was sent again; list the "
                              "passes, and remove the marker only once you know" % (name, universe_id))
        if dry_run or not confirm:
            return dict(out, ok=bool(sc.get("ok")), ran=True, dryRun=True, sent=False,
                        note="nothing was sent: this is the exact form a create would POST. It is sent with confirm true "
                             "and dry_run false, with the panel in full mode", **({} if sc.get("ok") else {
                                 "refusedRule": "key-scope", "error": sc.get("error"), "steps": sc.get("steps")}))
        if not sc.get("ok"):
            return dict(out, ok=False, ran=False, sent=False, refusedRule="key-scope", error=sc.get("error"),
                        steps=sc.get("steps"))
        mode, _st = _T()._plugin_mode()
        if mode != "full":
            return dict(out, ok=False, ran=False, sent=False, refusedRule="full-mode", mode=mode,
                        error="a game pass is made on the person's account; the MifBridge panel must be in full mode "
                              "(it is %s)" % mode)
        if tgate is not None:
            return dict(out, **tgate)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return dict(out, ok=False, ran=False, sent=False, refusedRule="already-sent", marker=marker,
                        error="another create of this name claimed the marker first: nothing was sent")
        os.write(fd, json.dumps({"universeId": universe_id, "name": name, "at": time.time()}).encode("utf-8"))
        os.close(fd)
        body, ctype = _form(fields)
        st, ans, _w = OC._call("POST", PASSES % universe_id, key, body, ctype, timeout=60, tries=1)
        ans = OC.redact(ans, key)
        out.update(ran=True, sent=True, status=st, answer=ans, marker=marker)
        gid = ans.get("gamePassId") if isinstance(ans, dict) else None
        if st != 200 or not _pos_int(gid):
            unknown = not isinstance(st, int) or st in (0, 408) or st >= 500
            if isinstance(st, int) and 400 <= st < 500 and st != 408:
                # a definite refusal (400, 401, 403, 429...): nothing was made, so the marker would only block a fixed
                # retry with already-sent for good
                try:
                    os.remove(marker)
                    out["markerRemoved"] = True
                except OSError as exc:
                    out["markerRemoved"] = False
                    out["markerError"] = exc.strerror or str(exc)
            return dict(out, ok=False, outcomeUnknown=unknown,
                        error="POST %s answered %s: %s. %s" % (
                            PASSES % universe_id, st, json.dumps(ans)[:300],
                            "Nothing was made; fix what it names and create again" if out.get("markerRemoved") else
                            "It is not sent again: list the passes"))
        back = rbx_game_pass("get", universe_id, game_pass_id=gid)
        p = back.get("pass") or {}
        good = back.get("ok") is True and p.get("name") == name and p.get("isForSale") is False
        out.update(ok=good, verified=good, gamePassId=gid, readBack=p)
        if not good:
            out["error"] = "the new pass %s read back %s (expected name %r, isForSale false)" % (
                gid, p or back.get("error"), name)
        return out
    except Exception as exc:  # noqa: BLE001 - never raise, never let the key out
        return {"ok": False, "ran": None, "error": "game pass call failed: %s" % OC.exc_text(exc)}
