/**
 * custom.js — Platform Assistant Header Injection
 * Loaded via .chainlit/config.toml → custom_js
 *
 * Injects two fixed-position elements above the Chainlit UI:
 *   1. Red header bar (48px)  — "Platform Assistant" left, "👤 Dummy User Name ⏻" right
 *   2. Yellow accent band (6px) — immediately below the red bar
 *
 * Uses position:fixed so it floats above the React SPA tree,
 * independent of any Chainlit component class names.
 * Adds padding-top to #root so the chat content doesn't hide behind it.
 */
(function () {
    "use strict";

    const RED_HEIGHT = 48;   // px — red header bar
    const YELLOW_HEIGHT = 3;    // px — yellow accent band
    const TOTAL_HEIGHT = RED_HEIGHT + YELLOW_HEIGHT;  // 51px

    // Avoid double-injection on hot-reloads
    if (document.getElementById("cl-header-bar")) return;

    /* ── 1. Red header bar ─────────────────────────────────────── */
    const header = document.createElement("div");
    header.id = "cl-header-bar";
    Object.assign(header.style, {
        position: "fixed",
        top: "0",
        left: "0",
        right: "0",
        height: RED_HEIGHT + "px",
        backgroundColor: "#CC0000",
        color: "#FFFFFF",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 16px",
        zIndex: "99999",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Inter', sans-serif",
        fontSize: "14px",
        fontWeight: "600",
        boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
        boxSizing: "border-box",
    });

    header.innerHTML = `
    <span style="display:flex;align-items:center;gap:8px;font-size:15px;font-weight:700;letter-spacing:0.2px;">
      🖥️ Platform Assistant
    </span>
    <span id="cl-user-info"
          style="display:flex;align-items:center;gap:6px;font-size:13px;font-weight:500;opacity:0.95;cursor:pointer;"
          title="Logout (coming soon)">
      👤 Dummy User Name
      <span style="font-size:17px;margin-left:2px;" title="Logout">⏻</span>
    </span>
  `;

    header.querySelector("#cl-user-info").addEventListener("click", function () {
        // TODO: wire up real logout / session clear here
        console.log("[Platform Assistant] Logout clicked — not yet implemented.");
    });

    /* ── 2. Yellow accent band ─────────────────────────────────── */
    const band = document.createElement("div");
    band.id = "cl-yellow-band";
    Object.assign(band.style, {
        position: "fixed",
        top: RED_HEIGHT + "px",
        left: "0",
        right: "0",
        height: YELLOW_HEIGHT + "px",
        backgroundColor: "#FFD000",
        zIndex: "99998",
        boxShadow: "0 1px 3px rgba(0,0,0,0.10)",
    });

    /* ── 3. Inject into the page ───────────────────────────────────
     Layout offset (margin-top + height shrink) is handled entirely
     by CSS: #root > div:first-child in custom.css.
     JS only needs to append the two fixed bars to the body.        */
    function inject() {
        if (!document.body) return false;
        document.body.appendChild(header);
        document.body.appendChild(band);
        return true;
    }

    // Try immediately (body usually exists when custom_js runs)
    if (!inject()) {
        document.addEventListener("DOMContentLoaded", inject);
    }
})();
