"""Food Stop Finder: the top 3 restaurants worth stopping for along a US driving route."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone, time as dtime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import polyline
import requests
import streamlit as st
from google import genai
from google.genai import types

try:
    from streamlit_geolocation import streamlit_geolocation
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

st.set_page_config(page_title="Food Stop Finder", page_icon="🍽️", layout="centered")

# ---------------- config ----------------
MAPS_KEY = st.secrets["MAPS_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]
PINNED_MODELS = list(st.secrets.get("GEMINI_MODELS", []))  # optional override; otherwise auto-detected
TZ = ZoneInfo(st.secrets.get("TZ", "America/Chicago"))

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_MASK = ("places.id,places.displayName,places.rating,places.userRatingCount,places.priceLevel,"
               "places.priceRange,places.primaryTypeDisplayName,places.googleMapsUri,"
               "places.regularOpeningHours.periods,places.utcOffsetMinutes,routingSummaries")
QUERIES = ("best restaurants", "highly rated local restaurant", "popular restaurant")
NOT_DINE_IN = "Delivery|Takeout|Takeaway|Catering|Grocery|Gas Station|Convenience"
RETRYABLE = ("503", "429", "504", "DEADLINE", "UNAVAILABLE", "timed out", "Timeout")
PRICE = {"PRICE_LEVEL_FREE": "Free", "PRICE_LEVEL_INEXPENSIVE": "$", "PRICE_LEVEL_MODERATE": "$$",
         "PRICE_LEVEL_EXPENSIVE": "$$$", "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$"}
SCORE = dict(C=4.2, m=400, detour_penalty=0.02, over_budget_penalty=0.15, ai_weight=0.1)
DEFAULT_TASTE = ("High standards for food, not a limited palate. Food quality matters most. Prefer local "
                 "spots and signature dishes over chains. Tell me the dish people rave about.")

# ---------------- optional password gate ----------------
if "APP_PASSWORD" in st.secrets and not st.session_state.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw and pw == st.secrets["APP_PASSWORD"]:
        st.session_state.authed = True
        st.rerun()
    if pw:
        st.error("Wrong password")
    st.stop()


# ---------------- helpers ----------------
def secs(s):
    return int(float(str(s).rstrip("s") or 0))


def num(x, default=5.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def waypoint(x):
    """Accepts an address or 'lat,lng'."""
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", x)
    if m:
        return {"location": {"latLng": {"latitude": float(m[1]), "longitude": float(m[2])}}}
    return {"address": x}


def is_open_at(periods, dt):
    """dt is restaurant-local clock time. Google weekdays: 0 = Sunday."""
    if not periods:
        return None
    t = ((dt.weekday() + 1) % 7) * 1440 + dt.hour * 60 + dt.minute
    for p in periods:
        o, c = p["open"], p.get("close")
        if c is None:
            return True  # open 24/7
        s = o["day"] * 1440 + o.get("hour", 0) * 60 + o.get("minute", 0)
        e = c["day"] * 1440 + c.get("hour", 0) * 60 + c.get("minute", 0)
        if (s <= t < e) if s <= e else (t >= s or t < e):
            return True
    return False


# ---------------- Google Maps ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_route(start, dest):
    r = requests.post(ROUTES_URL, timeout=30,
        headers={"X-Goog-Api-Key": MAPS_KEY,
                 "X-Goog-FieldMask": "routes.legs.steps.staticDuration,routes.legs.steps.polyline.encodedPolyline"},
        json={"origin": waypoint(start), "destination": waypoint(dest), "travelMode": "DRIVE"}).json()
    if "routes" not in r:
        raise RuntimeError(f"Couldn't find a route: {r.get('error', {}).get('message', 'check the addresses')}")
    pts, t = [], 0.0
    for s in r["routes"][0]["legs"][0]["steps"]:
        d = secs(s.get("staticDuration", "0s"))
        enc = s.get("polyline", {}).get("encodedPolyline")
        if enc:
            line = polyline.decode(enc)
            n = max(len(line) - 1, 1)
            pts += [(p, t + d * i / n) for i, p in enumerate(line)]
        t += d
    return pts, t


def search_segment(seg, base_min, query, arrive_utc):
    p = requests.post(PLACES_URL, timeout=30,
        headers={"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": PLACES_MASK},
        json={"textQuery": query, "maxResultCount": 20,
              "searchAlongRouteParameters": {"polyline": {"encodedPolyline": polyline.encode(seg)}},
              "routingParameters": {"origin": {"latitude": seg[0][0], "longitude": seg[0][1]}}}).json()
    if "error" in p:
        raise RuntimeError(f"Places API: {p['error'].get('message')}")
    rows = []
    for pl, rs in zip(p.get("places", []), p.get("routingSummaries", [])):
        legs = rs.get("legs") or []
        if not legs:
            continue
        pr = pl.get("priceRange", {})
        local = arrive_utc + timedelta(minutes=pl.get("utcOffsetMinutes", 0))
        rows.append(dict(
            id=pl["id"], name=pl["displayName"]["text"],
            type=pl.get("primaryTypeDisplayName", {}).get("text", ""),
            rating=pl.get("rating"), reviews=pl.get("userRatingCount", 0),
            price_level=pl.get("priceLevel", ""),
            min_price=int(pr.get("startPrice", {}).get("units", 0) or 0),
            max_price=int(pr.get("endPrice", {}).get("units", 0) or 0),
            open_at_arrival=is_open_at(pl.get("regularOpeningHours", {}).get("periods"), local),
            detour_min=round(sum(secs(l.get("duration", "0s")) for l in legs) / 60 - base_min, 1),
            link=pl.get("googleMapsUri", "")))
    return rows


@st.cache_data(ttl=1800, show_spinner=False)
def find_candidates(start, dest, from_min, to_min, depart_iso, chunks=3):
    pts, total = get_route(start, dest)
    total_min = total / 60
    to_min = min(to_min, total_min)
    if to_min - from_min < 5:
        raise ValueError(f"That window is past the end of the trip (the drive is {total_min:.0f} min).")
    dep = datetime.fromisoformat(depart_iso)
    edges = [from_min + (to_min - from_min) * i / chunks for i in range(chunks + 1)]
    jobs = []
    for i in range(chunks):
        lo, hi = edges[i] * 60, edges[i + 1] * 60
        seg = [p for p, ts in pts if lo <= ts <= hi]
        if len(seg) < 2:
            continue
        if len(seg) > 1000:  # keep the request small
            seg = seg[::len(seg) // 1000 + 1] + [seg[-1]]
        arrive = (dep + timedelta(seconds=(lo + hi) / 2)).astimezone(timezone.utc)
        jobs += [(seg, (hi - lo) / 60, q, arrive) for q in QUERIES]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=9) as ex:  # all searches at once instead of one by one
        futures = [ex.submit(search_segment, *j) for j in jobs]
    for f in futures:
        try:
            rows += f.result()
        except Exception as e:
            errors.append(str(e)[:150])
    if not rows:
        raise RuntimeError("No restaurants found. " + (errors[0] if errors else "Try a wider window."))
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True), total_min


def shortlist(df, P):
    n = P["n_analyze"]
    d0 = df[df.open_at_arrival.ne(False)]
    d0 = d0[~d0.type.str.contains(NOT_DINE_IN, case=False, na=False)]
    relaxed = max(25, P["min_reviews"] // 2)
    for min_rev, max_det in [(P["min_reviews"], P["max_detour"]),
                             (relaxed, P["max_detour"]),
                             (relaxed, P["max_detour"] + 5)]:
        d = d0[(d0.reviews >= min_rev) & (d0.detour_min <= max_det)].copy()
        if len(d) >= n:
            break
    v, R = d.reviews.astype(float), d.rating.astype(float).fillna(SCORE["C"])
    m, C = SCORE["m"], SCORE["C"]
    d["score"] = (v / (v + m)) * R + (m / (v + m)) * C          # review-volume-weighted rating
    d["score"] -= SCORE["detour_penalty"] * d.detour_min.clip(lower=0)
    d.loc[d.max_price > P["budget"], "score"] -= SCORE["over_budget_penalty"]
    return d.sort_values("score", ascending=False).head(n), (min_rev, max_det)


# ---------------- Gemini (one request for all places) ----------------
PROMPT = """You are helping someone with high food standards choose a road-trip meal stop.
Their taste: {taste}
Below are {n} candidate restaurants, each with recent customer reviews and numbered customer photos.
Return a JSON list with exactly one object per restaurant, in the same order, each with keys:
place: the restaurant number,
best_dish: the dish (or famous side, e.g. salsa) customers praise most, or "unclear",
evidence: short phrase, e.g. "3 of 5 reviews mention it" or "visible in photos",
food_score: number 1-10 for food quality, judged ONLY from that place's reviews and photos,
fit_score: number 1-10 for how well it fits their taste,
watch_out: the main recurring complaint, or "none",
verdict: one line written for them,
food_photo: the number of that place's photo that best shows the food, or 0 if none shows food.
Never invent dishes that are not in the reviews or visible in the photos."""


@st.cache_resource
def gem_client():
    return genai.Client(api_key=GEMINI_KEY, http_options=types.HttpOptions(timeout=30000))


def get_photo(name, w=400):  # smaller photos = faster download and faster AI
    try:
        r = requests.get(f"https://places.googleapis.com/v1/{name}/media",
                         params={"maxWidthPx": w, "key": MAPS_KEY}, timeout=15)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


def place_material(pid, n_photos=2):
    """Reviews + photos for one place."""
    d = requests.get(f"https://places.googleapis.com/v1/places/{pid}", timeout=20,
                     headers={"X-Goog-Api-Key": MAPS_KEY,
                              "X-Goog-FieldMask": "reviews.text,reviews.rating,photos.name"}).json()
    revs = "\n".join(f"- ({r.get('rating')}★) {r.get('text', {}).get('text', '')}" for r in d.get("reviews", []))
    with ThreadPoolExecutor(max_workers=n_photos) as ex:
        imgs = [img for img in ex.map(get_photo, [p["name"] for p in d.get("photos", [])[:n_photos]]) if img]
    return revs, imgs


@st.cache_data(ttl=21600, show_spinner=False)
def gemini_models():
    """Ask Google which Flash models this key can use right now (names change often)."""
    if PINNED_MODELS:
        return PINNED_MODELS
    skip = ("image", "tts", "audio", "live", "embedding", "native", "exp", "8b")
    found = []
    try:
        for m in gem_client().models.list():
            name = (m.name or "").removeprefix("models/")
            acts = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in acts and not any(s in name for s in skip):
                found.append(name)
    except Exception:
        pass
    def rank(n):
        ver = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", n)[:1]] or [0.0]
        return ("preview" in n, -ver[0], "lite" in n, len(n))  # stable > newest > full flash > shortest name
    found = sorted(set(found), key=rank)
    lite = [n for n in found if "lite" in n][:1]  # a fast lite model as last resort
    picks = [n for n in found if "lite" not in n][:2] + lite
    return picks or ["gemini-flash-latest", "gemini-flash-lite-latest"]


def gemini_json(parts):
    """One Gemini call. Tries each available model; a missing or busy model just moves on to the next."""
    fast = types.GenerateContentConfig(response_mime_type="application/json",
                                       thinking_config=types.ThinkingConfig(thinking_level="low"))
    plain = types.GenerateContentConfig(response_mime_type="application/json")
    errors = []
    for attempt in range(2):
        busy = False
        for model in gemini_models():
            for cfg in (fast, plain):
                try:
                    return json.loads(gem_client().models.generate_content(model=model, contents=parts, config=cfg).text)
                except Exception as e:
                    msg = str(e)
                    errors.append(f"{model}: {msg[:160]}")
                    if any(c in msg for c in RETRYABLE):
                        busy = True
                        break  # busy or slow: next model
                    if "404" in msg or "NOT_FOUND" in msg:
                        break  # model retired: next model
                    # anything else on the fast setting: retry this model with the plain setting
        if not busy:
            break
        time.sleep(3)
    raise RuntimeError(" | ".join(errors[-4:]) or "Gemini failed")


@st.cache_data(ttl=86400, show_spinner=False)
def analyze_all(places, taste):
    """places: tuple of (id, name, type, rating, reviews). Returns {id: (data, photo)}."""
    with ThreadPoolExecutor(max_workers=8) as ex:  # fetch every place's reviews and photos at once
        material = list(ex.map(lambda p: place_material(p[0]), places))
    parts = [PROMPT.format(taste=taste, n=len(places))]
    for i, ((pid, name, ptype, rating, reviews), (revs, imgs)) in enumerate(zip(places, material), 1):
        parts.append(f"\n### Restaurant {i}: {name} ({ptype}). Rating {rating} from {reviews} reviews.\n"
                     f"Reviews:\n{revs or '(no review text)'}")
        for j, img in enumerate(imgs, 1):
            parts += [f"Restaurant {i}, photo {j}:", types.Part.from_bytes(data=img, mime_type="image/jpeg")]
    return parse_batch(gemini_json(parts), places, material)


def parse_batch(data, places, material):
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    out = {}
    for k, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        i = int(num(item.get("place"), k + 1)) - 1
        if not 0 <= i < len(places):
            continue
        imgs = material[i][1]
        pick = int(num(item.get("food_photo"), 0))
        out[places[i][0]] = (item, imgs[pick - 1] if 1 <= pick <= len(imgs) else None)
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def analyze_one(place, taste):
    """Fallback: one small request for a single place."""
    revs_imgs = place_material(place[0])
    pid, name, ptype, rating, reviews = place
    parts = [PROMPT.format(taste=taste, n=1),
             f"\n### Restaurant 1: {name} ({ptype}). Rating {rating} from {reviews} reviews.\n"
             f"Reviews:\n{revs_imgs[0] or '(no review text)'}"]
    for j, img in enumerate(revs_imgs[1], 1):
        parts += [f"Restaurant 1, photo {j}:", types.Part.from_bytes(data=img, mime_type="image/jpeg")]
    return parse_batch(gemini_json(parts), (place,), [revs_imgs]).get(pid)


def ai_rank(sl, taste, k=3):
    recs = sl.to_dict("records")
    places = tuple((r["id"], r["name"], r["type"], r["rating"], int(r["reviews"])) for r in recs)
    error, results = None, {}
    try:
        results = analyze_all(places, taste)
    except Exception as e:
        error = str(e)[:300]
    missing = [p for p in places if p[0] not in results]
    if missing:  # batch failed or skipped places: small requests, all at once (free tier allows 5/min)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {p[0]: ex.submit(analyze_one, p, taste) for p in missing[:5]}
        for pid, f in futs.items():
            try:
                if f.result():
                    results[pid] = f.result()
            except Exception as e:
                error = error or str(e)[:300]
    out, failed = [], 0
    for r in recs:
        if r["id"] in results:
            data, photo = results[r["id"]]
            out.append({**r, **data, "photo": photo})
        else:
            failed += 1
            out.append({**r, "photo": None, "best_dish": "unclear", "verdict": ""})
    for r in out:
        r["final"] = r["score"] + SCORE["ai_weight"] * (num(r.get("food_score")) + num(r.get("fit_score"))) / 2
    return sorted(out, key=lambda r: r["final"], reverse=True)[:k], failed, error


# ---------------- UI pieces ----------------
def fmt_price(r):
    rng = f"${r['min_price']}–{r['max_price']}" if r.get("max_price") else ""
    lvl = PRICE.get(r.get("price_level"), "")
    return (" ".join(x for x in (lvl, rng) if x) or "price n/a").replace("$", "\\$")


def render(picks, dest):
    for i, r in enumerate(picks, 1):
        with st.container(border=True):
            if r.get("photo"):
                st.image(r["photo"])
            st.markdown(f"### {i}. {r['name']}")
            st.markdown(f"**Best dish:** {r.get('best_dish') or 'unclear'}")
            if r.get("evidence"):
                st.caption(r["evidence"])
            rating = r["rating"] if pd.notna(r["rating"]) else "–"
            det = r["detour_min"]
            det_txt = "right on your route" if det <= 1 else f"+{det:.0f} min detour"
            st.markdown(f"⭐ {rating} ({int(r['reviews']):,} reviews) · {det_txt} · {fmt_price(r)}")
            watch = str(r.get("watch_out") or "none")
            if watch.lower() not in ("none", "nan", ""):
                st.markdown(f"⚠️ {watch}")
            if r.get("verdict"):
                st.write(r["verdict"])
            nav = (f"https://www.google.com/maps/dir/?api=1&destination={quote(dest)}"
                   f"&waypoints={quote(r['name'])}&waypoint_place_ids={r['id']}&travelmode=driving")
            c1, c2 = st.columns(2)
            c1.link_button("🧭 Navigate via here", nav)
            if r.get("link"):
                c2.link_button("📍 Google Maps page", r["link"])


def run(start, dest, from_min, to_min, depart_iso, P, key):
    if not start or not dest:
        st.warning("Add both a starting point and a destination.")
        return
    st.session_state[key] = None
    try:
        with st.status("Finding restaurants along your route…", expanded=True) as status:
            df, total = find_candidates(start, dest, from_min, to_min, depart_iso)
            st.write(f"Drive is {total / 60:.1f} hrs · {len(df)} restaurants in your window")
            sl, (mr, md) = shortlist(df, P)
            if sl.empty:
                status.update(label="Nothing matched", state="error")
                st.warning("No open, dine-in places matched. Widen the window or lower min reviews.")
                return
            note = "" if (mr, md) == (P["min_reviews"], P["max_detour"]) else f" (relaxed to {mr}+ reviews, ≤{md} min detour)"
            st.write(f"Shortlisted {len(sl)}{note}. AI is reading reviews and photos…")
            t0 = time.time()
            picks, failed, ai_error = ai_rank(sl, P["taste"])
            status.update(label="Done ✅", state="complete", expanded=False)
        st.session_state[key] = dict(picks=picks, dest=dest, failed=failed, error=ai_error,
                                     ai_secs=round(time.time() - t0))
    except Exception as e:
        st.error(f"Something went wrong: {e}")


def show(key):
    res = st.session_state.get(key)
    if not res:
        return
    if res["failed"]:
        st.info(f"AI couldn't check {res['failed']} place(s); those are ranked on ratings only.")
        if res.get("error"):
            with st.expander("Why? (technical detail)"):
                st.code(res["error"])
    st.caption(f"AI step took {res.get('ai_secs', '?')} s")
    render(res["picks"], res["dest"])


# ---------------- sidebar: saved preferences ----------------
qp = st.query_params


def qp_get(k, default, lo=None, hi=None, cast=int):
    try:
        v = cast(qp[k]) if k in qp else default
    except (TypeError, ValueError):
        return default
    if lo is not None:
        v = min(max(v, lo), hi)
    return v


REVIEW_OPTS = [25, 50, 100, 200, 500]
with st.sidebar:
    st.header("⚙️ Preferences")
    rev_default = qp_get("reviews", 100)
    P = dict(
        budget=st.slider("Soft budget per person ($)", 10, 150, qp_get("budget", 50, 10, 150), 5,
                         help="Places above this get a small penalty, not removed."),
        max_detour=st.slider("Max detour (min)", 5, 30, qp_get("detour", 15, 5, 30)),
        min_reviews=st.select_slider("Min reviews", REVIEW_OPTS,
                                     rev_default if rev_default in REVIEW_OPTS else 100),
        n_analyze=st.slider("Places for AI to check", 3, 8, qp_get("n", 5, 3, 8),
                            help="More = better picks, slightly slower. All places are checked in one AI request."),
        taste=st.text_area("Your taste", qp_get("taste", DEFAULT_TASTE, cast=str), height=130),
    )
    if st.button("💾 Save as my defaults"):
        st.query_params.update(budget=str(P["budget"]), detour=str(P["max_detour"]),
                               reviews=str(P["min_reviews"]), n=str(P["n_analyze"]), taste=P["taste"])
        st.success("Saved into this page's link. Bookmark it or add it to your home screen.")

# ---------------- main ----------------
st.title("🍽️ Food Stop Finder")
st.caption("The 3 places worth stopping for along your route.")
tab_now, tab_plan = st.tabs(["🍔 Hungry now", "🗓️ Plan ahead"])

with tab_now:
    current = ""
    if HAS_GEO:
        st.caption("Tap 📍 to use your current location")
        loc = streamlit_geolocation()
        if loc and loc.get("latitude"):
            current = f"{loc['latitude']:.5f},{loc['longitude']:.5f}"
            st.success(f"Using your location ({current})")
    typed = st.text_input("Where are you?", key="h_from", placeholder="City, address, or lat,lng",
                          help="Leave blank if you used 📍")
    h_dest = st.text_input("Where are you headed?", key="h_dest")
    h_hrs = st.slider("Search the next … hours of road", 0.5, 4.0, 1.5, 0.5, key="h_hrs")
    if st.button("Find food", type="primary", key="h_go"):
        now = datetime.now(TZ).replace(second=0, microsecond=0)
        now -= timedelta(minutes=now.minute % 10)  # 10-min buckets so repeat searches hit the cache
        run(typed.strip() or current, h_dest.strip(), 0, h_hrs * 60, now.isoformat(), P, "h_res")
    show("h_res")

with tab_plan:
    p_from = st.text_input("Starting from", key="p_from")
    p_dest = st.text_input("Going to", key="p_dest")
    c1, c2 = st.columns(2)
    p_date = c1.date_input("Departure date", datetime.now(TZ).date(), key="p_date")
    p_time = c2.time_input("Departure time", dtime(8, 0), key="p_time")
    win = st.slider("When do you want to eat? (hours into the drive)", 0.0, 10.0, (1.5, 2.5), 0.25, key="p_win")
    if st.button("Plan my stop", type="primary", key="p_go"):
        dep = datetime.combine(p_date, p_time, tzinfo=TZ)
        run(p_from.strip(), p_dest.strip(), win[0] * 60, win[1] * 60, dep.isoformat(), P, "p_res")
    show("p_res")
