#!/usr/bin/env python3
"""
Batch reclassify 'click' annotations into subtypes:
  click_dead | click_popout | click_external | click_unknown_nav

Run from anywhere; processes every *_human_interaction_annotation.json found
under BASE_DIR and writes *_click_subtype.json + *_click_subtype_report.html
into the same interaction/ folder.
"""

import json
import os
import glob
import re
import base64
import html as html_lib
import warnings
from pathlib import Path
from io import BytesIO
from typing import Tuple, List, Dict, Any

warnings.filterwarnings("ignore")

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow not installed.  Run: pip install Pillow --break-system-packages")


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_annotation_files(base_dir: str) -> List[str]:
    pattern = os.path.join(base_dir, "**", "*_human_interaction_annotation.json")
    return sorted(glob.glob(pattern, recursive=True))


def find_png(annotation_path: str, source_png: str) -> str:
    interaction_dir = os.path.dirname(annotation_path)
    app_dir         = os.path.dirname(interaction_dir)
    return os.path.join(app_dir, "pages", source_png)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_element(
    img: Image.Image,
    bbox: dict,
    sx: float,
    sy: float,
    padding: int = 60,
) -> Image.Image:
    """Crop element with padding, draw red rect, resize ≤ 400 px wide.

    bbox_png coordinates are always in PNG pixel space (the field name says so).
    sx/sy are kept as parameters for API compatibility but are NOT used for
    coordinate scaling — they were only needed to handle Figma→PNG conversion,
    which the dataset already did when producing bbox_png values.
    """
    img_w, img_h = img.size

    # bbox_png is in PNG pixel coords — use directly, no sx/sy multiplication
    px = int(bbox["x"])
    py = int(bbox["y"])
    pw = max(int(bbox["w"]), 20)
    ph = max(int(bbox["h"]), 20)

    # Clamp to image bounds
    px = max(0, min(px, img_w - 1))
    py = max(0, min(py, img_h - 1))
    pw = min(pw, img_w - px)
    ph = min(ph, img_h - py)

    x1 = max(0, px - padding)
    y1 = max(0, py - padding)
    x2 = min(img_w, px + pw + padding)
    y2 = min(img_h, py + ph + padding)

    crop = img.crop((x1, y1, x2, y2)).copy()

    rect_x1 = px - x1
    rect_y1 = py - y1
    rect_x2 = rect_x1 + pw
    rect_y2 = rect_y1 + ph

    draw = ImageDraw.Draw(crop)
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline="red", width=3)

    cw, ch = crop.size
    if cw > 400:
        crop = crop.resize((400, max(1, int(ch * 400 / cw))), Image.LANCZOS)

    return crop


def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_click(ann: dict, page_name: str) -> Tuple[str, str]:
    """
    Return (subtype, one-sentence reasoning).

    Priority order: click_external → click_dead → click_popout → click_unknown_nav
    (most-specific first; unknown_nav is the safe fallback)
    """
    node_name: str = ann["node"]["name"]
    nl = node_name.lower().strip()
    ntype = ann["node"].get("type", "")
    depth = ann["node"].get("depth", 99)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def R(pattern: str) -> bool:
        return bool(re.search(pattern, nl))

    # ═════════════════════════════════════════════════════════════════════════
    # 1.  click_external
    # ═════════════════════════════════════════════════════════════════════════

    # Raw URL node names → external
    if re.match(r"https?://", nl) or re.match(r"www\.", nl):
        return ("click_external",
                f'"{node_name}" contains a URL and opens an external site in a new tab.')

    # Domain-like names (e.g. "twitter.com/jakegyll", "instagram.com/jakegyll")
    if re.search(r"\.(com|org|net|io|dev|app)\b", nl) and "/" in nl:
        return ("click_external",
                f'"{node_name}" is a domain/profile URL that links to an external site.')

    # App-store badges  (Google Play / App Store)
    if R(r"\b(google\s*play|app\s*store|play\s*store|app\s*badge)\b"):
        return ("click_external",
                f'"{node_name}" is an app-store badge; clicking opens the store in a new tab.')

    # Explicit external resource links (GitHub, docs, blog posts by name)
    if R(r"\b(github|source[-\s]?code|repo|repository)\b"):
        return ("click_external",
                f'"{node_name}" is a link to a code repository / external resource in a new tab.')

    # Social media icons / profile links (not OAuth buttons)
    if R(r"\b(facebook(?!\s*login|\s*sign)|twitter(?!\s*login|\s*sign)|instagram|"
         r"linkedin|youtube|tiktok|pinterest|snapchat|reddit|discord|vimeo|"
         r"behance|dribbble|medium|tumblr)\b"):
        return ("click_external",
                f'"{node_name}" is a social-media icon/link that opens an external site in a new tab.')

    # "Social Icons" component (collection of social share icons)
    if R(r"\bsocial\s*(icons?|sharing|share)\b"):
        return ("click_external",
                f'"{node_name}" is a social-sharing icon group linking to external social platforms.')

    # Partner / sponsor logos (named logos that are NOT the site's own logo)
    if R(r"\b(partner|sponsor|client|brand)\b") and "logo" in nl:
        return ("click_external",
                f'"{node_name}" is a partner/sponsor logo that links to an external site.')

    # Standalone logos that are not the site's own identity
    if "logo" in nl and not R(r"\b(site|header|nav|main|brand|company|app|footer|your|logo\s*icon)\b"):
        return ("click_external",
                f'"{node_name}" appears to be a third-party brand logo that opens an external site.')

    # ═════════════════════════════════════════════════════════════════════════
    # 2.  click_dead
    # ═════════════════════════════════════════════════════════════════════════

    # Explicit "Current Page" / "active" marker
    if nl in ("current page",) or R(r"\bcurrent\s*page\b"):
        return ("click_dead",
                f'"{node_name}" explicitly marks the current page; clicking produces no change.')

    # Active / selected / disabled state
    if R(r"\b(active|selected|disabled)\b") and not R(r"\b(tab|menu|item)\b"):
        return ("click_dead",
                f'"{node_name}" is in an active/selected/disabled state; clicking produces no change.')

    # Active tab on the current page
    if R(r"\btab\b") and R(r"\bactive\b"):
        return ("click_dead",
                f'"{node_name}" is the currently-active tab; clicking it produces no navigation.')

    # Quantity increase/decrease buttons (e-commerce cart)
    if R(r"\b(increase|decrease)\b") or R(r"\b(increase|decrease)\s*quantity\b"):
        return ("click_dead",
                f'"{node_name}" adjusts item quantity in-page; no navigation or overlay occurs.')

    # Remove from cart (in-page update, no navigation)
    if nl in ("remove",) or R(r"\bremove\s*(item|from\s*cart|product)?\b"):
        return ("click_dead",
                f'"{node_name}" removes an item from the cart in-place; no navigation occurs.')

    # Social reactions: likes / dislikes / votes (in-page state change, no nav)
    if R(r"^(likes?|dislikes?|upvote?|downvote?|reaction)s?$"):
        return ("click_dead",
                f'"{node_name}" is a reaction button that updates a count in-place; no navigation or overlay.')

    # Music-player / media controls (play, pause, skip, shuffle, repeat, volume, seek)
    if R(r"\b(play_circle|pause|skip|shuffle|repeat|mute|unmute|seek|scrubber|progress\s*bar|"
         r"audio\s*player|previous\s*track|next\s*track)\b"):
        return ("click_dead",
                f'"{node_name}" is a media-player control; clicking changes playback state without navigating.')

    # In-page "Fill" elements in music/media contexts (SVG fill shapes — play button backgrounds, etc.)
    if nl in ("fill",):
        return ("click_dead",
                f'"{node_name}" is likely an SVG fill shape inside a media control; no standalone navigation.')

    # expand_less / collapse toggle
    if nl in ("expand_less", "expand_more", "collapse"):
        return ("click_dead",
                f'"{node_name}" collapses or expands content in-place; no navigation or overlay.')

    # Nav link whose name overlaps significantly with the current page name → dead link
    page_words = (
        set(re.sub(r"^\d+_", "", page_name).lower().replace("_", " ").replace("-", " ").split())
        - {"page", "the", "a", "an"}
    )
    node_words = (
        set(re.sub(r"[^a-z0-9 ]", " ", nl).split())
        - {"page", "the", "a", "an", "link", "nav", "btn", "button", "item", "menu",
           "container", "text", "icon", "wrapper", "group"}
    )
    overlap = page_words & node_words
    # Also allow FRAME/COMPONENT if the name contains "Link" explicitly (e.g. "Home Link")
    is_link_frame = (ntype in ("FRAME", "COMPONENT", "INSTANCE") and "link" in nl)
    if (
        len(overlap) >= 1
        and len(overlap) / max(len(page_words), 1) >= 0.6
        and (ntype == "TEXT" or is_link_frame)
        and depth <= 7
        and not R(r"\b(card|image|img|photo|thumbnail|banner|cover|list|grid|container|section)\b")
    ):
        return ("click_dead",
                f'Nav link "{node_name}" matches the current page ({page_name}); clicking it produces no navigation.')

    # ═════════════════════════════════════════════════════════════════════════
    # 3.  click_popout
    # ═════════════════════════════════════════════════════════════════════════

    # Emoji picker
    if R(r"\bemoj[iy]\b"):
        return ("click_popout",
                f'"{node_name}" opens an emoji-picker overlay.')

    # File / attachment picker
    if nl in ("file",) or R(r"\b(attach|attachment|file\s*(picker|upload|btn|button))\b"):
        return ("click_popout",
                f'"{node_name}" opens a file-picker or attachment dialog.')

    # Search icon / search popup trigger
    if R(r"\b(search\s*(icon|popup|trigger|toggle|container)|m-search-popup)\b"):
        return ("click_popout",
                f'"{node_name}" opens a search overlay or popout input.')

    # Language / locale switcher
    if R(r"\b(language|lang|locale|translation)\b"):
        return ("click_popout",
                f'"{node_name}" opens a language-selection dropdown.')

    # Settings icon  (not a full settings-page link)
    if R(r"\bsettings?\s*(icon|btn|toggle|container)\b"):
        return ("click_popout",
                f'"{node_name}" opens a settings panel or dropdown.')

    # Notification bell
    if R(r"\b(notification|bell|notificatio[ns]|vuesax.*notification)\b"):
        return ("click_popout",
                f'"{node_name}" opens a notifications panel or dropdown.')

    # Cart icon → cart drawer/modal
    if R(r"\bcart\s*(icon|btn|bag|basket|container)?\b") and not R(r"\b(page|link)\b"):
        return ("click_popout",
                f'"{node_name}" opens the cart drawer or mini-cart modal.')

    # Menu / hamburger / drawer toggle
    if R(r"\b(hamburger|mobile\s*menu|menu\s*(icon|toggle|btn)|nav\s*toggle|"
         r"sidebar\s*(toggle|menu)|drawer\s*toggle)\b"):
        return ("click_popout",
                f'"{node_name}" opens a navigation drawer or mobile menu overlay.')

    # User / account menu icon
    if R(r"\b(avatar|user\s*(icon|menu|dropdown|toggle|profile)|profile\s*(icon|menu|dropdown|toggle)|"
         r"account\s*(icon|menu|dropdown))\b"):
        return ("click_popout",
                f'"{node_name}" opens a user-account dropdown menu.')

    # Profile Menu (named component)
    if nl in ("profile menu", "my profile"):
        return ("click_popout",
                f'"{node_name}" opens a profile/account dropdown menu.')

    # Overflow / more-options menus
    if (
        nl in ("more", "more options", "dots-vertical", "feather:more-vertical",
               "bi:chevron-down", "control_point")
        or R(r"\b(three[\s-]?dot|more[\s-]?option|kebab|ellipsis|overflow|"
             r"feather:more|dots[\s-]vertical)\b")
    ):
        return ("click_popout",
                f'"{node_name}" opens a contextual overflow/more-options dropdown.')

    # FAQ / accordion / expand-collapse
    if R(r"\b(faq|accordion|expand|collapse|toggle\s*(btn|button)|read\s*(less|more)\s*container)\b"):
        return ("click_popout",
                f'"{node_name}" is an accordion/FAQ toggle that expands content in place.')

    # "Add New" → opens a create/add dialog
    if R(r"^add(\s+new)?$") or R(r"\badd\s*(new|item|post|tag|user|member|row)\b"):
        return ("click_popout",
                f'"{node_name}" opens a dialog or form to add a new item.')

    # Edit → opens an edit modal/dialog
    if nl in ("edit",) or R(r"^edit\s*(btn|button|icon)?$"):
        return ("click_popout",
                f'"{node_name}" opens an edit modal or inline form.')

    # Dropdown / select components
    if R(r"\b(dropdown|select\s*component|m-select|enquiry\s*form\s*select)\b"):
        return ("click_popout",
                f'"{node_name}" opens a dropdown selection panel.')

    # Filter & Sort
    if R(r"\bfilter\s*(&|and)\s*sort\b") or R(r"\bfilter\s*(icon|btn|panel|modal)\b"):
        return ("click_popout",
                f'"{node_name}" opens a filter/sort panel or dropdown.')

    # Date pickers (Start Date, End Date)
    if R(r"\b(start\s*date|end\s*date|date\s*picker|date\s*range)\b"):
        return ("click_popout",
                f'"{node_name}" opens a date-picker calendar overlay.')

    # Volume control in media player — in-place state change (not a popout)
    # A standalone "VOL" or volume-slider element directly adjusts volume.
    # Only contextual volume *icon buttons* that open a separate panel → popout.
    if nl in ("vol",) or R(r"\bvol\b(?!\s*ume)"):
        return ("click_dead",
                f'"{node_name}" is an in-player volume control; clicking adjusts volume in-place.')
    if R(r"\bvolume\s*(slider|bar|track|scrubber)\b"):
        return ("click_dead",
                f'"{node_name}" is an in-player volume slider; clicking adjusts volume in-place.')
    if R(r"\bvolume\s*(icon|control|btn|button)\b"):
        return ("click_popout",
                f'"{node_name}" opens a volume control popout.')

    # Reply / comment inputs (inline dialog) — also catches Figma typo "replly"
    if R(r"\brepl+[yi]\b") or R(r"\breply\s*(btn|button|input|form)?\b"):
        return ("click_popout",
                f'"{node_name}" opens a reply input or inline comment form.')

    # "see replies" → expands replies inline
    if R(r"\bsee\s*repl(ies|y)\b"):
        return ("click_popout",
                f'"{node_name}" expands/shows replies inline without full-page navigation.')

    # Map marker → info popup on map
    if R(r"\bmap\s*(marker|pin|popup)\b"):
        return ("click_popout",
                f'"{node_name}" shows a location info popup on the map.')

    # Info tooltip / info-circle
    if nl in ("info-circle", "info") or R(r"\binfo[\s-]?(circle|icon|tooltip|badge)\b"):
        return ("click_popout",
                f'"{node_name}" shows a tooltip or informational popover.')

    # Social sharing component (share dialog, not share icons)
    if R(r"\bsocial\s*sharing\b"):
        return ("click_popout",
                f'"{node_name}" opens a social-sharing dialog.')

    # Share button (opens share panel)
    if R(r"\bshare\s*(btn|button|icon|link)?$"):
        return ("click_popout",
                f'"{node_name}" opens a share panel or social-sharing dialog.')

    # Contact / newsletter form submits → toast / modal confirmation
    if R(r"\b(contact\s*(us|form|submit|btn|button)|newsletter\s*(submit|btn|button|subscribe)|"
         r"sell\s*(btn|button|cta)|send\s*(message|btn|button|email)|"
         r"message\s*(btn|button|send))\b"):
        return ("click_popout",
                f'"{node_name}" submits a contact/newsletter form and likely triggers a confirmation modal or toast.')

    # "Maximize" / "Button Resize" → in-page view toggle
    if R(r"\b(maximize|fullscreen|button\s*resize)\b"):
        return ("click_popout",
                f'"{node_name}" toggles a maximized or fullscreen overlay view.')

    # Payment Type / Country / Travel Type — selector dropdowns on forms
    if R(r"\b(payment\s*type|travel\s*type|country)\b"):
        return ("click_popout",
                f'"{node_name}" opens a form-field selector dropdown.')

    # ═════════════════════════════════════════════════════════════════════════
    # 3b. click_social_oauth  — social / OAuth authentication buttons
    # ═════════════════════════════════════════════════════════════════════════

    # "Social button" component, or explicit "Continue/Sign in with <Provider>"
    if R(r"\b(social\s*button)\b"):
        return ("click_social_oauth",
                f'"{node_name}" is a social OAuth button that redirects to an authentication provider.')

    if R(r"\b(continue\s*with|sign[\s-]?in\s*with|log[\s-]?in\s*with|sign[\s-]?up\s*with)\s+"
         r"(google|apple|facebook|twitter|github|microsoft|amazon|linkedin)\b"):
        return ("click_social_oauth",
                f'"{node_name}" is a social OAuth button that redirects to an authentication provider.')

    if R(r"\b(google(?!\s*play)|apple(?!\s*store)|facebook)\s*(login|sign[\s-]?in|sign[\s-]?up|auth|oauth|btn|button)?\b"):
        return ("click_social_oauth",
                f'"{node_name}" is a social OAuth button that redirects to an authentication provider.')

    # ═════════════════════════════════════════════════════════════════════════
    # 3c. click_upload_file  — file upload / picker triggers
    # ═════════════════════════════════════════════════════════════════════════

    if R(r"\b(upload|attach(ment)?)\s*(file|photo|image|avatar|document|media|btn|button|icon)?\b"):
        return ("click_upload_file",
                f'"{node_name}" opens a file upload dialog.')

    if R(r"\badd\s*(file|attachment|photo|image|document|media)\b"):
        return ("click_upload_file",
                f'"{node_name}" opens a file picker or attachment dialog.')

    # ═════════════════════════════════════════════════════════════════════════
    # 4.  click_unknown_nav  (safe fallback — URL changes, target not in dataset)
    # ═════════════════════════════════════════════════════════════════════════

    # OAuth / Social login buttons (Google, Apple, Facebook sign-in) — fallback catch
    if R(r"\b(google(?!\s*play)|apple(?!\s*store)|facebook\s*login|sign\s*in\s*with)\b"):
        return ("click_social_oauth",
                f'"{node_name}" is a social/OAuth login button that triggers an authentication redirect.')

    # Sign up / register
    if R(r"\b(sign\s*up|signup|register|create\s*(an?\s*)?account|join\s*(now|us|free)?)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a sign-up/register CTA that navigates to a registration or onboarding page.')

    # Sign in / log in
    if R(r"\b(sign\s*in|signin|log[\s-]?in|login\s*(btn|button|buttom)?)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a login CTA that navigates to the authenticated area.')

    # Get started / free trial / onboarding CTAs
    if R(r"\b(get\s*started|start\s*(free\s*)?trial|free\s*trial|try\s*(free|now)?)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is an onboarding CTA that navigates to the next step or a new page.')

    # Purchase / checkout / subscription CTAs
    if R(r"\b(buy\s*now|purchase|checkout|proceed|order\s*now|pay\s*now|add\s*to\s*cart|"
         r"select\s*plan|choose\s*plan|upgrade|subscribe\s*now)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a purchase/checkout CTA that navigates to an order or payment page.')

    # View details / see more / load more
    if R(r"\b(view\s*(details?|more|all|property|listing|job|profile|tour|jobs?)|"
         r"see\s*(all|more|details?)|show\s*(all\s*(users?|transactions?|more))|"
         r"load\s*more|read\s*more\b(?!\s*container)|show\s*more)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" navigates to a detail, listing, or expanded page.')

    # Explore / discover / learn more
    if R(r"\b(explore|discover|learn\s*more|explore\s*more|getting\s*started)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a discovery/onboarding CTA that navigates to another page.')

    # Search submit
    if R(r"\b(search\s*(btn|button|submit|go|now|bar|box)?|button\s*search|find\s*(btn|button|jobs?))\b"):
        return ("click_unknown_nav",
                f'"{node_name}" submits a search query and navigates to search results.')

    # Forgot / reset password
    if R(r"\b(forgot|forget|reset)\s*password\b"):
        return ("click_unknown_nav",
                f'"{node_name}" navigates to the password-reset flow.')

    # Apply / book / reserve / request
    if R(r"\b(apply\s*(now|job)?|book\s*(now|a\s*tour|tour)?|reserve|"
         r"request\s*(info|quote|demo|call|appointment|viewing)|"
         r"get\s*a?\s*quote|schedule\b(?!\s*task))\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a booking/application CTA that navigates to a form or confirmation page.')

    # Job-board specific CTAs
    if R(r"\b(apply|post\s*a?\s*job|find\s*jobs?|browse\s*(jobs?|companies?)|job\s*alerts)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a job-board CTA that navigates to a listing or application page.')

    # Upload actions → file picker (not unknown nav)
    if R(r"\bupload\b"):
        return ("click_upload_file",
                f'"{node_name}" triggers a file upload action that opens a file picker.')

    # Download actions
    if R(r"\bdownload\s*(icon|file|btn|button)?\b") and "icon" not in nl:
        return ("click_unknown_nav",
                f'"{node_name}" triggers a file download action.')

    # Named content links: Blog/Agency/Agent/Listing/Pages/Document links
    if R(r"\b(blog|agency|agent|listing|pages?|document|category|tag|author|"
         r"post\s*sidebar|job\s*(list|category)|upcomming\s*packages)\s*(link|container|text)?\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a content navigation link to a related page.')

    # "Item ⏵ Link" style filter / facet navigation
    if "link" in nl and ("⏵" in node_name or R(r"\bitem\b")):
        return ("click_unknown_nav",
                f'"{node_name}" is a facet/filter navigation link that updates the listing page.')

    # Pagination
    if R(r"\b(pagination|page\s*number|prev(ious)?|next)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" navigates to the previous or next page of results.')

    # Tab navigation (non-active tabs)
    if R(r"\btab[\s-]?(btn|button|menu|bar)?\b") or R(r"\bspanPage\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a tab that switches to a different view or section.')

    # Generic nav link  /  menu item
    if R(r"\b(link|nav\s*(item|link)?|menu\s*(item|hover)?|navigation|pages?\s*link)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a navigation link that leads to another page in the app.')

    # Generic button not already classified
    if R(r"\b(btn|button|cta|_button\s*base)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" is a button that likely triggers navigation to another page or route.')

    # Any name containing a known page-transition verb
    if R(r"\b(view|open|go\s*to|navigate|proceed)\b"):
        return ("click_unknown_nav",
                f'"{node_name}" triggers a page transition based on its name.')

    # Fallback
    return ("click_unknown_nav",
            f'"{node_name}" is an unclassified clickable element; defaulting to unknown navigation '
            f'as the target page is not in the dataset.')


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generation
# ─────────────────────────────────────────────────────────────────────────────

SUBTYPE_COLORS = {
    "click_dead":         "#94a3b8",
    "click_popout":       "#a78bfa",
    "click_external":     "#34d399",
    "click_unknown_nav":  "#fb923c",
    "click_social_oauth": "#60a5fa",
    "click_upload_file":  "#fbbf24",
    "click_play_music":   "#4ade80",
    "click_vol":          "#facc15",
    "click_next_misuc":   "#f97316",
    "click_pre_misuc":    "#c084fc",
}

SUBTYPE_LABELS = {
    "click_dead":         "Dead Click",
    "click_popout":       "Popout / Overlay",
    "click_external":     "External Link",
    "click_unknown_nav":  "Unknown Nav",
    "click_social_oauth": "Social OAuth",
    "click_upload_file":  "Upload File",
    "click_play_music":   "Play / Pause",
    "click_vol":          "Volume Control",
    "click_next_misuc":   "Next Track",
    "click_pre_misuc":    "Prev Track",
}


def _badge_html(subtype: str) -> str:
    color = SUBTYPE_COLORS.get(subtype, "#888")
    label = SUBTYPE_LABELS.get(subtype, subtype)
    return (
        f'<span style="background:{color};color:#0f172a;padding:2px 8px;'
        f'border-radius:9999px;font-size:.75rem;font-weight:700;">{label}</span>'
    )


def generate_html_report(
    page_name: str,
    results: List[Dict[str, Any]],
    subtype_counts: Dict[str, int],
) -> str:
    legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">'
        f'<span style="width:13px;height:13px;border-radius:3px;background:{SUBTYPE_COLORS[st]};display:inline-block;"></span>'
        f'<span style="color:#cbd5e1;font-size:.83rem;">{SUBTYPE_LABELS[st]}</span>'
        f'</span>'
        for st in SUBTYPE_COLORS
    )

    badge_counts = "".join(
        f'<span style="background:{SUBTYPE_COLORS[st]};color:#0f172a;padding:3px 12px;'
        f'border-radius:9999px;font-weight:700;font-size:.83rem;white-space:nowrap;">'
        f'{SUBTYPE_LABELS[st]}: {subtype_counts.get(st, 0)}</span> '
        for st in ["click_dead", "click_popout", "click_external", "click_unknown_nav",
                   "click_social_oauth", "click_upload_file",
                   "click_play_music", "click_vol", "click_next_misuc", "click_pre_misuc"]
    )

    cards_html = ""
    for r in results:
        b64 = r.get("b64", "")
        img_tag = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:5px;margin-bottom:7px;" />'
            if b64 else
            '<div style="background:#0f172a;height:70px;border-radius:5px;margin-bottom:7px;'
            'display:flex;align-items:center;justify-content:center;color:#475569;font-size:.72rem;">'
            'No image</div>'
        )
        cards_html += f"""
        <div style="background:#1e293b;border-radius:10px;padding:13px;break-inside:avoid;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
            <span style="background:#0f172a;color:#64748b;padding:2px 8px;border-radius:5px;font-size:.72rem;font-weight:700;">#{html_lib.escape(str(r['id']))}</span>
            {_badge_html(r['subtype'])}
          </div>
          <div style="color:#e2e8f0;font-size:.83rem;font-weight:600;margin-bottom:7px;word-break:break-all;">{html_lib.escape(r['node_name'])}</div>
          {img_tag}
          <div style="color:#94a3b8;font-size:.76rem;line-height:1.55;">{html_lib.escape(r['reasoning'])}</div>
        </div>
        """

    total = sum(subtype_counts.values())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{html_lib.escape(page_name)} — Click Subtype Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;padding:24px}}
  h1{{font-size:1.35rem;font-weight:700;color:#f1f5f9;margin-bottom:5px}}
  .sub{{color:#64748b;font-size:.88rem;margin-bottom:14px}}
  .counts{{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px}}
  .legend{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:22px;padding:11px 15px;background:#1e293b;border-radius:8px}}
  .grid{{columns:4 270px;gap:15px}}
  @media(max-width:700px){{.grid{{columns:1}}}}
</style>
</head>
<body>
  <h1>📋 {html_lib.escape(page_name)}</h1>
  <p class="sub">Total <em>click</em> annotations: <strong>{total}</strong></p>
  <div class="counts">{badge_counts}</div>
  <div class="legend">{legend_html}</div>
  <div class="grid">{cards_html}</div>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Visual-inspection overrides  (app, page_name, annotation_id) → (subtype, reasoning)
# ─────────────────────────────────────────────────────────────────────────────

def _dead(reason):  return ("click_dead", reason)
def _pop(reason):   return ("click_popout", reason)
def _oauth(reason): return ("click_social_oauth", reason)
def _upl(reason):   return ("click_upload_file", reason)
def _play(reason):  return ("click_play_music", reason)
def _vol(reason):   return ("click_vol", reason)
def _next(reason):  return ("click_next_misuc", reason)
def _prev(reason):  return ("click_pre_misuc", reason)

VISUAL_OVERRIDES: Dict[Tuple[str, str, int], Tuple[str, str]] = {
    # ── 10_streaming_music-streaming ─────────────────────────────────────────
    ("10_streaming_music-streaming","02_Log_in_Page",5): _oauth("Facebook OAuth login button — triggers social authentication."),
    ("10_streaming_music-streaming","02_Log_in_Page",6): _oauth("Google OAuth login button — triggers social authentication."),

    # Heart/like fill shapes → dead (in-place toggle)
    ("10_streaming_music-streaming","05_Home",12): _dead("Heart/like SVG fill shape — in-place like toggle, no navigation."),
    ("10_streaming_music-streaming","06_Playlist",6): _dead("Heart/like SVG fill — in-place like toggle, no navigation."),
    ("10_streaming_music-streaming","07_Browser",4):  _dead("Heart/like SVG fill — in-place like toggle, no navigation."),
    ("10_streaming_music-streaming","09_Permium_Subscriptions",7): _dead("Heart/like SVG fill — in-place like toggle, no navigation."),

    # Notification bell icons (top-right, y≈36) → popout
    ("10_streaming_music-streaming","05_Home",3):             _pop("Notification bell (Icons, top-right) — opens notifications panel."),
    ("10_streaming_music-streaming","08_Music_Player_Page",5):_pop("Notification bell (Icons, top-right) — opens notifications panel."),
    ("10_streaming_music-streaming","10_Settings_-_Profile",3):_pop("Notification bell (Icons, top-right) — opens notifications panel."),
    ("10_streaming_music-streaming","11_Settings_-_Detailes",7): _pop("Notification bell (Vector) — opens notifications panel."),

    # ── Bottom mini-player controls (x=704→prev, x=760→play, x=816→next) ───
    # 05_Home mini-player
    ("10_streaming_music-streaming","05_Home",97): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","05_Home",98): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","05_Home",96): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","05_Home",99): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 06_Playlist mini-player
    ("10_streaming_music-streaming","06_Playlist",34): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","06_Playlist",35): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","06_Playlist",33): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","06_Playlist",36): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 07_Browser mini-player
    ("10_streaming_music-streaming","07_Browser",96): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","07_Browser",97): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","07_Browser",95): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","07_Browser",98): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 08_Music_Player_Page full player
    ("10_streaming_music-streaming","08_Music_Player_Page",1): _play("Play/Pause button (full player) — toggles music playback."),
    ("10_streaming_music-streaming","08_Music_Player_Page",2): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 10_Settings_-_Profile mini-player
    ("10_streaming_music-streaming","10_Settings_-_Profile",15): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","10_Settings_-_Profile",16): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","10_Settings_-_Profile",14): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","10_Settings_-_Profile",13): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 11_Settings_-_Detailes mini-player
    ("10_streaming_music-streaming","11_Settings_-_Detailes",25): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","11_Settings_-_Detailes",26): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","11_Settings_-_Detailes",24): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","11_Settings_-_Detailes",27): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 12_Settings_-_Contact_Us mini-player
    ("10_streaming_music-streaming","12_Settings_-_Contact_Us",21): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","12_Settings_-_Contact_Us",22): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","12_Settings_-_Contact_Us",20): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","12_Settings_-_Contact_Us",23): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # 13_Settings_-_FAQ mini-player
    ("10_streaming_music-streaming","13_Settings_-_FAQ",19): _prev("Previous track button (x=704) — plays previous song."),
    ("10_streaming_music-streaming","13_Settings_-_FAQ",20): _play("Play/Pause button (x=760) — toggles music playback."),
    ("10_streaming_music-streaming","13_Settings_-_FAQ",18): _next("Next track button (x=816) — plays next song."),
    ("10_streaming_music-streaming","13_Settings_-_FAQ",21): _vol("VOL — in-player volume slider; adjusts volume in-place."),

    # ── 2_real-estate ────────────────────────────────────────────────────────
    ("2_real-estate","01_sign_in",4): _oauth("Social OAuth login button — triggers social authentication redirect."),
    ("2_real-estate","01_sign_in",5): _oauth("Social OAuth login button — triggers social authentication redirect."),
    ("2_real-estate","02_sign_up",7): _oauth("Social OAuth signup button — triggers social authentication redirect."),
    ("2_real-estate","02_sign_up",8): _oauth("Social OAuth signup button — triggers social authentication redirect."),
    **{("2_real-estate","06_Buy_Grid_With_Map",i): _pop("Map filter container — opens filter overlay panel.") for i in range(33,44)},
    ("2_real-estate","07_Buy_Details_-_Request_Info",14): _dead("View toggle icon — switches display mode in-place."),
    ("2_real-estate","07_Buy_Details_-_Request_Info",15): _dead("View toggle icon — switches display mode in-place."),
    ("2_real-estate","07_Buy_Details_-_Request_Info",16): _dead("View toggle icon — switches display mode in-place."),
    ("2_real-estate","07_Buy_Details_-_Request_Info",18): _dead("Carousel thumbnail button — cycles images in-place."),
    ("2_real-estate","07_Buy_Details_-_Request_Info",19): _dead("Carousel thumbnail button — cycles images in-place."),
    ("2_real-estate","07_Buy_Details_-_Request_Info",43): _pop("Play button (Vector) — opens video player overlay."),
    ("2_real-estate","14_Agent_Details",15): _pop('"Read More Container" — expands/collapses text in place.'),

    # ── 1_newsletter ─────────────────────────────────────────────────────────
    ("1_newsletter","01_Home",1): _pop("Subscribe form frame — submitting triggers confirmation modal/toast."),
    ("1_newsletter","01_Home",2): _pop("Subscribe form frame — submitting triggers confirmation modal/toast."),
    ("1_newsletter","01_Home",3): _pop("Subscribe form frame — submitting triggers confirmation modal/toast."),
    ("1_newsletter","01_Home",4): _pop("Subscribe form frame — submitting triggers confirmation modal/toast."),

    # ── 3_job-board ──────────────────────────────────────────────────────────
    ("3_job-board","01_Landing_page",4):  _pop("Location dropdown frame — opens a location selector panel."),
    ("3_job-board","01_Landing_page",34): _pop("Subscribe button — triggers newsletter confirmation modal."),
    ("3_job-board","02_Find_jobs",44):    _pop("Sort dropdown frame — opens sort/order selector panel."),
    ("3_job-board","17_Dashboard_-_Settings",11): _dead("Active settings tab (Caption) — already on this tab, no navigation."),

    # ── 5_travel-booking ─────────────────────────────────────────────────────
    ("5_travel-booking","01_Homepage",20):        _pop("Subscribe/newsletter button — triggers confirmation modal/toast."),
    ("5_travel-booking","03_ABout_us",7):         _pop("Play button (Vector) — opens video player overlay."),
    ("5_travel-booking","04_Package_archive",13): _pop("Filter rectangle — opens filter/sort panel overlay."),

    # ── 6_chat ───────────────────────────────────────────────────────────────
    ("6_chat","04_Chat",6): _pop("Microphone icon — opens audio recording panel or permission dialog."),

    # ── 8_ecommerce ──────────────────────────────────────────────────────────
    ("8_ecommerce","01_Sign_in",1): _oauth('"Sign up with Google" rectangle — OAuth authentication button.'),
    ("8_ecommerce","03_Home_page",9):  _dead("Carousel arrow — advances carousel in-place, no page navigation."),
    ("8_ecommerce","03_Home_page",10): _dead("Carousel arrow — advances carousel in-place, no page navigation."),
    ("8_ecommerce","03_Home_page",26): _dead("Carousel arrow — advances carousel in-place, no page navigation."),
    ("8_ecommerce","03_Home_page",27): _dead("Carousel arrow — advances carousel in-place, no page navigation."),
    ("8_ecommerce","03_Home_page",28): _pop("Subscribe Now button — triggers newsletter confirmation modal."),
    ("8_ecommerce","04_Shop_page",3):  _pop("Pagination dropdown — opens page-number selector panel."),
    **{("8_ecommerce","04_Shop_page",i): _dead("Color selector chip (Item) — in-place color selection.") for i in range(43,83)},
    ("8_ecommerce","04_Shop_page",88): _dead("View toggle (Button) — switches grid/list view in-place."),
    ("8_ecommerce","04_Shop_page",89): _dead("View toggle (Button) — switches grid/list view in-place."),
    ("8_ecommerce","04_Shop_page",90): _dead("View toggle (Button) — switches grid/list view in-place."),
    ("8_ecommerce","04_Shop_page",91): _dead("View toggle (Button) — switches grid/list view in-place."),
    ("8_ecommerce","04_Shop_page",92): _dead("View toggle (Button) — switches grid/list view in-place."),
    **{("8_ecommerce","05_Product_Page",i): _dead("Size selector chip (Label) — in-place size selection.") for i in range(18,26)},

    # ── 9_project-management ─────────────────────────────────────────────────
    ("9_project-management","02_User_Dashboard",1): _pop('"This Week" dropdown — opens time-period selector panel.'),
    ("9_project-management","02_User_Dashboard",2): _pop('"This Week" dropdown — opens time-period selector panel.'),
    ("9_project-management","02_User_Dashboard",3): _pop('"This Week" dropdown — opens time-period selector panel.'),
    ("9_project-management","06_User_Profile",23):  _pop("Date picker control — opens a calendar overlay."),
    ("9_project-management","10_User_Task_PopUp",4): _upl('"Add attachment" — triggers a file upload dialog.'),
    ("9_project-management","10_User_Task_PopUp",5): _pop("Priority selector (Frame 12) — opens priority dropdown."),

    # ── 7_cloud-storage ──────────────────────────────────────────────────────
    # Social OAuth (welcome / login page buttons)
    ("7_cloud-storage","01_welcome",3): _oauth("Continue with Google/Facebook OAuth — triggers social authentication."),
    ("7_cloud-storage","01_welcome",4): _oauth("Continue with Apple/Google OAuth — triggers social authentication."),
    ("7_cloud-storage","03_Login",3):   _oauth("Continue with Google/Facebook OAuth — triggers social authentication."),
    ("7_cloud-storage","03_Login",4):   _oauth("Continue with Apple/Google OAuth — triggers social authentication."),

    # Social login settings — connect buttons (Group 73)
    ("7_cloud-storage","23_Social_login",1): _oauth("Connect social OAuth provider (Group 73) — opens OAuth authorization."),
    ("7_cloud-storage","23_Social_login",5): _oauth("Connect social OAuth provider (Group 73) — opens OAuth authorization."),
    # Social login settings — enable/disable toggles (Rectangle 88)
    ("7_cloud-storage","23_Social_login",2):  _dead("Social login enable/disable toggle (Rect 88) — in-place state change."),
    ("7_cloud-storage","23_Social_login",6):  _dead("Social login enable/disable toggle (Rect 88) — in-place state change."),
    ("7_cloud-storage","23_Social_login",10): _dead("Social login enable/disable toggle (Rect 88) — in-place state change."),

    # Upload file — Group 70 (primary upload/new button)
    ("7_cloud-storage","04_Home",17): _upl("Upload/New button (Group 70) — opens file upload dialog."),
    ("7_cloud-storage","05_Home",6):  _upl("Upload/New button (Group 70) — opens file upload dialog."),
    # Upload file — Group 71
    ("7_cloud-storage","04_Home",47): _upl("Upload button (Group 71) — opens file upload dialog."),
    ("7_cloud-storage","05_Home",5):  _upl("Upload button (Group 71) — opens file upload dialog."),
    ("7_cloud-storage","06_Photos",40): _upl("Upload button (Group 71) — opens file upload dialog."),
    ("7_cloud-storage","08_Transfer",16): _upl("Upload button (Group 71) — opens file upload dialog."),
    # Upload file — Group 23
    ("7_cloud-storage","04_Home",42): _upl("Upload button (Group 23) — opens file upload dialog."),
    ("7_cloud-storage","05_Home",44): _upl("Upload button (Group 23) — opens file upload dialog."),
    ("7_cloud-storage","06_Photos",29): _upl("Upload button (Group 23) — opens file upload dialog."),
    ("7_cloud-storage","07_Folder_list",30): _upl("Upload button (Group 23) — opens file upload dialog."),
    # Upload file — Branding assets (Group 252, 253, Rectangle 88)
    ("7_cloud-storage","13_Branding",1): _upl("Upload brand asset (Group 252) — opens file picker."),
    ("7_cloud-storage","13_Branding",2): _upl("Upload brand asset (Group 253) — opens file picker."),
    ("7_cloud-storage","13_Branding",3): _upl("Upload button (Rectangle 88) — opens file picker."),
    # Upload file — profile photo and payment documents
    ("7_cloud-storage","20_Settings",4):  _upl("Upload profile photo (Rectangle 88) — opens file picker."),
    ("7_cloud-storage","32_Payments",31): _upl("Upload payment document (Rectangle 88) — opens file picker."),
    ("7_cloud-storage","32_Payments",32): _upl("Upload payment document (Rectangle 88) — opens file picker."),
    ("7_cloud-storage","15_Refer_a_friend",4): _upl("Upload/attach file (Rectangle 88) — opens file picker."),
    # Upload file — Appearance theme uploads (Group 281-286)
    ("7_cloud-storage","21_Appearance",5):  _upl("Upload theme image (Group 286) — opens file picker."),
    ("7_cloud-storage","21_Appearance",6):  _upl("Upload theme image (Group 285) — opens file picker."),
    ("7_cloud-storage","21_Appearance",7):  _upl("Upload theme image (Group 284) — opens file picker."),
    ("7_cloud-storage","21_Appearance",8):  _upl("Upload theme image (Group 283) — opens file picker."),
    ("7_cloud-storage","21_Appearance",9):  _upl("Upload theme image (Group 282) — opens file picker."),
    ("7_cloud-storage","21_Appearance",10): _upl("Upload theme image (Group 281) — opens file picker."),

    # Popout — share menus (Group 40, 24, 27, 28)
    ("7_cloud-storage","04_Home",40): _pop("Share button (Group 40) — opens share/context menu."),
    ("7_cloud-storage","04_Home",41): _pop("Share button (Group 24) — opens share/context menu."),
    ("7_cloud-storage","04_Home",45): _pop("Share button (Group 27) — opens share/context menu."),
    ("7_cloud-storage","04_Home",46): _pop("Share button (Group 28) — opens share/context menu."),
    ("7_cloud-storage","05_Home",42): _pop("Share button (Group 40) — opens share/context menu."),
    ("7_cloud-storage","05_Home",43): _pop("Share button (Group 24) — opens share/context menu."),
    ("7_cloud-storage","05_Home",47): _pop("Share button (Group 27) — opens share/context menu."),
    ("7_cloud-storage","05_Home",48): _pop("Share button (Group 28) — opens share/context menu."),
    ("7_cloud-storage","06_Photos",35): _pop("Share button (Group 40) — opens share/context menu."),
    ("7_cloud-storage","06_Photos",30): _pop("Share button (Group 24) — opens share/context menu."),
    ("7_cloud-storage","06_Photos",38): _pop("Share button (Group 27) — opens share/context menu."),
    ("7_cloud-storage","06_Photos",39): _pop("Share button (Group 28) — opens share/context menu."),
    ("7_cloud-storage","07_Folder_list",24): _pop("Share button (Group 40) — opens share/context menu."),
    ("7_cloud-storage","07_Folder_list",29): _pop("Share button (Group 24) — opens share/context menu."),
    ("7_cloud-storage","07_Folder_list",27): _pop("Share button (Group 27) — opens share/context menu."),
    ("7_cloud-storage","07_Folder_list",28): _pop("Share button (Group 28) — opens share/context menu."),
    ("7_cloud-storage","08_Transfer",14): _pop("Share button (Group 40) — opens share/context menu."),

    # Popout — help/support button (Group 45)
    ("7_cloud-storage","04_Home",31):           _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","05_Home",16):           _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","06_Photos",55):         _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","07_Folder_list",44):    _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","08_Transfer",26):       _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","09_Account_Profile",37):_pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","10_Security",34):       _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","11_Storage",16):        _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","13_Branding",25):       _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","14_Notification",33):   _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","15_Refer_a_friend",23): _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","16_Application",27):    _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","17_Developer",28):      _pop("Help button (Group 45) — opens help/support panel."),
    ("7_cloud-storage","18_Privacy",19):        _pop("Help button (Group 45) — opens help/support panel."),

    # Popout — edit button (Rectangle 57)
    ("7_cloud-storage","04_Home",34):        _pop("Edit button (Rectangle 57) — opens inline edit form."),
    ("7_cloud-storage","05_Home",35):        _pop("Edit button (Rectangle 57) — opens inline edit form."),
    ("7_cloud-storage","07_Folder_list",17): _pop("Edit button (Rectangle 57) — opens inline edit form."),

    # Popout — avatar/account menu (Group 230, 231, 42)
    ("7_cloud-storage","09_Account_Profile",25): _pop("Avatar group (Group 230) — opens profile/account dropdown."),
    ("7_cloud-storage","09_Account_Profile",26): _pop("Avatar group (Group 231) — opens profile/account dropdown."),
    ("7_cloud-storage","09_Account_Profile",31): _pop("Account group (Group 42) — opens profile dropdown."),
    ("7_cloud-storage","10_Security",32):        _pop("Account group (Group 42) — opens profile dropdown."),
    ("7_cloud-storage","16_Application",24):     _pop("Account group (Group 42) — opens profile dropdown."),

    # Popout — dropdown selectors (Rectangle 84, 85)
    ("7_cloud-storage","09_Account_Profile",15): _pop("Dropdown selector (Rectangle 84) — opens option selector."),
    ("7_cloud-storage","09_Account_Profile",16): _pop("Dropdown selector (Rectangle 85) — opens option selector."),

    # Popout — security/settings buttons (Rectangle 117, 119)
    ("7_cloud-storage","10_Security",2):   _pop("Security action button (Rectangle 117) — opens confirmation dialog."),
    ("7_cloud-storage","10_Security",6):   _pop("Security action button (Rectangle 119) — opens confirmation dialog."),
    ("7_cloud-storage","12_Billing",25):   _pop("Billing action button (Rectangle 119) — opens plan/payment dialog."),
    ("7_cloud-storage","20_Settings",1):   _pop("Settings action button (Rectangle 117) — opens settings dialog."),

    # Popout — Frame 1/2 context-specific
    ("7_cloud-storage","12_Billing",2):          _pop("Billing action (Frame 1) — opens plan/payment dialog."),
    ("7_cloud-storage","13_Branding",5):         _pop("Branding action (Frame 1) — triggers save confirmation toast."),
    ("7_cloud-storage","15_Refer_a_friend",1):   _pop("Refer invite (Frame 1) — opens send invite confirmation."),
    ("7_cloud-storage","15_Refer_a_friend",2):   _pop("Refer copy/send (Frame 2) — opens share confirmation dialog."),
    ("7_cloud-storage","17_Developer",6):        _pop("Developer API action (Frame 1) — opens API key dialog."),
    ("7_cloud-storage","32_Payments",19):        _pop("Payments action (Frame 1) — opens payment form/confirmation."),

    # Popout — tag create/filter buttons (Group 337, 338)
    ("7_cloud-storage","28_Tage",16):  _pop("Create tag button (Group 337) — opens tag creation dialog."),
    ("7_cloud-storage","28_Tage",17):  _pop("Filter tag button (Group 338) — opens tag filter dialog."),
    ("7_cloud-storage","29_Users",16): _pop("Filter button (Group 338) — opens user filter dialog."),
    ("7_cloud-storage","29_Users",17): _pop("Create button (Group 337) — opens user creation dialog."),
    ("7_cloud-storage","30_Plans",16): _pop("Filter/create button (Group 337) — opens plan filter dialog."),

    # Popout — edit action in Pages list (Group 312)
    ("7_cloud-storage","26_Pages",21): _pop("Edit page action (Group 312) — opens inline edit form."),
    ("7_cloud-storage","26_Pages",22): _pop("Edit page action (Group 312) — opens inline edit form."),
    ("7_cloud-storage","26_Pages",23): _pop("Edit page action (Group 312) — opens inline edit form."),
    ("7_cloud-storage","26_Pages",24): _pop("Edit page action (Group 312) — opens inline edit form."),
    ("7_cloud-storage","26_Pages",25): _pop("Edit page action (Group 312) — opens inline edit form."),

    # Popout — env var type dropdowns (Rectangle 512 on 25_Environment)
    ("7_cloud-storage","25_Environment",16): _pop("Env var type selector (Rectangle 512) — opens type dropdown."),
    ("7_cloud-storage","25_Environment",18): _pop("Env var type selector (Rectangle 512) — opens type dropdown."),
    ("7_cloud-storage","25_Environment",20): _pop("Env var type selector (Rectangle 512) — opens type dropdown."),
    # Env var value input fields (Frame 1 on 25_Environment) → dead
    ("7_cloud-storage","25_Environment",17): _dead("Env var input field (Frame 1) — text input, no navigation."),
    ("7_cloud-storage","25_Environment",19): _dead("Env var input field (Frame 1) — text input, no navigation."),
    ("7_cloud-storage","25_Environment",21): _dead("Env var input field (Frame 1) — text input, no navigation."),

    # Dead — Transfer controls (pause/cancel/clear)
    ("7_cloud-storage","08_Transfer",1): _dead("Transfer control (Rectangle 222) — pauses/cancels in-place, no navigation."),
    ("7_cloud-storage","08_Transfer",2): _dead("Transfer control (Rectangle 221) — pauses/cancels in-place, no navigation."),
    ("7_cloud-storage","08_Transfer",3): _dead("Transfer control (Rectangle 220) — pauses/cancels in-place, no navigation."),

    # Dead — Active nav items (app sidebar)
    ("7_cloud-storage","10_Security",16):       _dead("Security nav item (Group 375) — already on Security page."),
    ("7_cloud-storage","14_Notification",23):   _dead("Notification nav item (Group 371) — already on Notification page."),
    ("7_cloud-storage","15_Refer_a_friend",15): _dead("Refer nav item (Group 367) — already on Refer page."),
    ("7_cloud-storage","16_Application",19):    _dead("Application nav item (Group 368) — already on Application page."),
    ("7_cloud-storage","17_Developer",18):      _dead("Developer nav item (Group 369) — already on Developer page."),
    ("7_cloud-storage","18_Privacy",13):        _dead("Privacy nav item (Group 370) — already on Privacy page."),
    ("7_cloud-storage","23_Social_login",17):   _dead("Social Login nav item (Group 380) — already on Social Login page."),
    ("7_cloud-storage","24_Adsense",6):         _dead("Adsense nav item (Group 381) — already on Adsense page."),
    ("7_cloud-storage","12_Billing",10):        _dead("Billing nav item (Group 373) — already on Billing page."),
    ("7_cloud-storage","13_Branding",14):       _dead("Branding nav item (Group 372) — already on Branding page."),
    ("7_cloud-storage","20_Settings",15):       _dead("Settings nav item (Group 377) — already on Settings page."),
    ("7_cloud-storage","21_Appearance",17):     _dead("Appearance nav item (Group 378) — already on Appearance page."),
    # Admin sidebar active nav items
    ("7_cloud-storage","26_Pages",8):           _dead("Pages admin nav item (Group 225) — already on Pages page."),
    ("7_cloud-storage","27_Languages",9):       _dead("Languages admin nav item (Group 224) — already on Languages page."),
    ("7_cloud-storage","29_Users",11):          _dead("Users admin nav item (Group 226) — already on Users page."),
    ("7_cloud-storage","30_Plans",12):          _dead("Plans admin nav item (Group 220) — already on Plans page."),
    ("7_cloud-storage","31_Transactions",13):   _dead("Transactions admin nav item (Group 221) — already on Transactions page."),
    ("7_cloud-storage","32_Payments",14):       _dead("Payments admin nav item (Group 222) — already on Payments page."),
    ("7_cloud-storage","33_Billing",15):        _dead("Billing admin nav item (Group 227) — already on Billing page."),

    # Dead — Language toggles / list items
    ("7_cloud-storage","27_Languages",16): _dead("Language toggle (Group 322) — in-place selection, no navigation."),
    ("7_cloud-storage","27_Languages",17): _dead("Language toggle (Group 318) — in-place selection, no navigation."),
    ("7_cloud-storage","27_Languages",18): _dead("Language toggle (Group 319) — in-place selection, no navigation."),
    ("7_cloud-storage","27_Languages",19): _dead("Language toggle (Group 321) — in-place selection, no navigation."),
    ("7_cloud-storage","27_Languages",20): _dead("Language toggle (Group 320) — in-place selection, no navigation."),
    ("7_cloud-storage","27_Languages",21): _dead("Language list item (Rectangle 512) — in-place language toggle."),
    ("7_cloud-storage","27_Languages",24): _dead("Language list item (Rectangle 512) — in-place language toggle."),
    ("7_cloud-storage","27_Languages",25): _dead("Language list item (Rectangle 512) — in-place language toggle."),
    ("7_cloud-storage","27_Languages",26): _dead("Language list item (Rectangle 512) — in-place language toggle."),
    ("7_cloud-storage","27_Languages",27): _dead("Language list item (Rectangle 512) — in-place language toggle."),
    ("7_cloud-storage","27_Languages",28): _dead("Language list item (Rectangle 512) — in-place language toggle."),

    # Dead — Security: remove trusted device / session X buttons
    ("7_cloud-storage","10_Security",7):  _dead("Remove trusted device (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",8):  _dead("Remove trusted device (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",9):  _dead("Remove trusted device (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",10): _dead("Remove trusted device (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",11): _dead("Remove trusted device (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",12): _dead("Remove active session (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",13): _dead("Remove active session (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",14): _dead("Remove active session (cross-small) — removes row in-place."),
    ("7_cloud-storage","10_Security",29): _dead("Security vector element — in-place action, no navigation."),

    # Dead — Notification preference checkboxes (14_Notification ids 1-14)
    **{("7_cloud-storage","14_Notification",i): _dead("Notification preference checkbox — in-place toggle, no navigation.") for i in range(1,15)},
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-page processor
# ─────────────────────────────────────────────────────────────────────────────

def process_page(annotation_path: str) -> Dict[str, Any]:
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    page_name: str  = data["page_name"]
    source_png: str = data["source_png"]
    figma_meta: dict = data["figma_meta"]
    annotations: list = data.get("annotations", [])

    # Derive app name from directory structure: {base}/{app}/interaction/{file}
    app_name = os.path.basename(os.path.dirname(os.path.dirname(annotation_path)))

    png_path = find_png(annotation_path, source_png)
    if not os.path.exists(png_path):
        print(f"  [WARN] PNG not found: {png_path} — skipping")
        return {"page_name": page_name, "skipped": True}

    try:
        img = Image.open(png_path).convert("RGB")
    except Exception as e:
        print(f"  [WARN] Cannot open PNG {png_path}: {e} — skipping")
        return {"page_name": page_name, "skipped": True}

    png_w, png_h = img.size
    sx = png_w / figma_meta["figma_w"]
    sy = png_h / figma_meta["figma_h"]

    click_anns = [a for a in annotations if a.get("type") == "click"]
    subtype_counts = {k: 0 for k in SUBTYPE_COLORS}
    output_annotations: list = []
    report_items: list = []

    for ann in click_anns:
        override_key = (app_name, page_name, ann["id"])
        if override_key in VISUAL_OVERRIDES:
            subtype, reasoning = VISUAL_OVERRIDES[override_key]
        else:
            subtype, reasoning = classify_click(ann, page_name)
        subtype_counts[subtype] += 1

        bbox = ann["bbox_png"]
        b64  = ""
        try:
            crop_img = crop_element(img, bbox, sx, sy)
            b64 = img_to_b64(crop_img)
        except Exception as e:
            print(f"    [WARN] Crop failed id={ann['id']}: {e}")

        output_annotations.append({
            "id":            ann["id"],
            "original_type": "click",
            "subtype":       subtype,
            "node_name":     ann["node"]["name"],
            "reasoning":     reasoning,
            "bbox_png":      bbox,
        })
        report_items.append({
            "id":        ann["id"],
            "subtype":   subtype,
            "node_name": ann["node"]["name"],
            "reasoning": reasoning,
            "b64":       b64,
        })

    out_dir = os.path.dirname(annotation_path)

    # Write JSON
    json_out = {
        "page_name":   page_name,
        "total_click": len(click_anns),
        "subtypes":    subtype_counts,
        "annotations": output_annotations,
    }
    with open(os.path.join(out_dir, f"{page_name}_click_subtype.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # Write HTML report
    html_content = generate_html_report(page_name, report_items, subtype_counts)
    with open(os.path.join(out_dir, f"{page_name}_click_subtype_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    print(
        f"  ✓ {page_name}: {len(click_anns)} clicks → "
        f"dead={subtype_counts['click_dead']} "
        f"popout={subtype_counts['click_popout']} "
        f"ext={subtype_counts['click_external']} "
        f"nav={subtype_counts['click_unknown_nav']} "
        f"oauth={subtype_counts['click_social_oauth']} "
        f"upload={subtype_counts['click_upload_file']}"
    )
    return {"page_name": page_name, "total_click": len(click_anns), "subtypes": subtype_counts}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    base_dir = "/sessions/serene-affectionate-newton/mnt/tasks"
    annotation_files = find_annotation_files(base_dir)
    print(f"Found {len(annotation_files)} annotation files.\n")

    all_results = []
    for i, ann_path in enumerate(annotation_files, 1):
        rel = os.path.relpath(ann_path, base_dir)
        print(f"[{i}/{len(annotation_files)}] {rel}")
        result = process_page(ann_path)
        all_results.append(result)

    # Aggregate summary
    total_pages  = sum(1 for r in all_results if not r.get("skipped"))
    total_clicks = sum(r.get("total_click", 0) for r in all_results)
    agg = {k: sum(r.get("subtypes", {}).get(k, 0) for r in all_results) for k in SUBTYPE_COLORS}

    print(f"\n{'='*60}")
    print(f"Done.  {total_pages} pages processed, {total_clicks} click annotations classified.")
    print(f"  click_dead:         {agg['click_dead']}")
    print(f"  click_popout:       {agg['click_popout']}")
    print(f"  click_external:     {agg['click_external']}")
    print(f"  click_unknown_nav:  {agg['click_unknown_nav']}")
    print(f"  click_social_oauth: {agg['click_social_oauth']}")
    print(f"  click_upload_file:  {agg['click_upload_file']}")
    print(f"  click_play_music:   {agg['click_play_music']}")
    print(f"  click_vol:          {agg['click_vol']}")
    print(f"  click_next_misuc:   {agg['click_next_misuc']}")
    print(f"  click_pre_misuc:    {agg['click_pre_misuc']}")


if __name__ == "__main__":
    main()
