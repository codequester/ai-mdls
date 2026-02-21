/**
 * custom.js — Platform Assistant UI Injections
 * Loaded via .chainlit/config.toml → custom_js
 *
 * Responsibilities:
 *   1. Inject a thin yellow accent band directly below the red header
 *   2. Inject "Dummy User Name  ⏻" into the right side of the header
 *
 * Both are injected once the header DOM node appears (MutationObserver).
 * When logout is wired up, replace the click handler on #cl-user-banner.
 */

(function () {
    "use strict";

    /* ── Config ──────────────────────────────────────────────── */
    const DISPLAY_NAME = "Dummy User Name";  // ← replace with real session user later
    const POWER_EMOJI = "⏻";

    /* ── Avoid double-injection on hot-reloads ───────────────── */
    if (document.getElementById("cl-yellow-band")) return;

    /* ──────────────────────────────────────────────────────────
       findHeader — try several Chainlit DOM shapes
    ────────────────────────────────────────────────────────── */
    function findHeader() {
        return (
            document.querySelector("header") ||
            document.querySelector("[class*='header'][class*='sticky']") ||
            document.querySelector(".sticky.top-0")
        );
    }

    /* ──────────────────────────────────────────────────────────
       injectUserBanner — adds username + power icon to header right
    ────────────────────────────────────────────────────────── */
    function injectUserBanner(header) {
        if (document.getElementById("cl-user-banner")) return;

        const banner = document.createElement("div");
        banner.id = "cl-user-banner";
        banner.title = "Logout (coming soon)";
        banner.innerHTML = `
      <span style="opacity:0.85;font-size:12px;">👤</span>
      <span>${DISPLAY_NAME}</span>
      <span style="font-size:16px;margin-left:4px;" title="Logout">${POWER_EMOJI}</span>
    `;

        // Click handler — wire up real logout here later
        banner.addEventListener("click", function () {
            console.log("[Platform Assistant] Logout clicked — not yet wired up.");
        });

        header.appendChild(banner);
    }

    /* ──────────────────────────────────────────────────────────
       injectYellowBand — inserts thin yellow stripe after header
    ────────────────────────────────────────────────────────── */
    function injectYellowBand(header) {
        if (document.getElementById("cl-yellow-band")) return;

        const band = document.createElement("div");
        band.id = "cl-yellow-band";
        // Inline style as fallback — custom.css also targets #cl-yellow-band
        band.style.cssText = [
            "width:100%",
            "height:6px",
            "background-color:#FFD000",
            "position:sticky",
            "top:0",
            "z-index:49",
            "box-shadow:0 1px 3px rgba(0,0,0,0.12)",
            "flex-shrink:0",
        ].join(";");

        // Insert immediately after the header in the DOM
        if (header.nextSibling) {
            header.parentNode.insertBefore(band, header.nextSibling);
        } else {
            header.parentNode.appendChild(band);
        }
    }

    /* ──────────────────────────────────────────────────────────
       inject — run both injections once the header is present
    ────────────────────────────────────────────────────────── */
    function inject() {
        const header = findHeader();
        if (!header) return false;          // not ready yet

        injectUserBanner(header);
        injectYellowBand(header);
        return true;
    }

    /* ──────────────────────────────────────────────────────────
       Wait for the Chainlit React app to render the header.
       Chainlit is a SPA — DOM may not exist on script load.
    ────────────────────────────────────────────────────────── */
    if (inject()) return;   // already rendered (unlikely but possible)

    const observer = new MutationObserver(function (_mutations, obs) {
        if (inject()) {
            obs.disconnect();   // stop watching once injected
        }
    });

    observer.observe(document.body || document.documentElement, {
        childList: true,
        subtree: true,
    });

    // Safety net: stop observing after 15 s regardless
    setTimeout(function () { observer.disconnect(); }, 15000);
})();
