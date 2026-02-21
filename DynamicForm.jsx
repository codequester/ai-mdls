import { useState } from "react";

// Chainlit injects submitElement(data) and cancelElement() as globals
// into the browser context for custom elements — do NOT import from @chainlit/sdk
// (that is an npm package and cannot be resolved at runtime in public/elements/).

/**
 * DynamicForm.jsx
 *
 * A fully dynamic form element rendered by Chainlit when an MCP tool
 * call requires user input. Field definitions are passed via props.
 *
 * Props:
 *   title       {string}   - Form title
 *   description {string}   - Brief description of what the action does
 *   tool        {string}   - Internal tool name (echoed back in response)
 *   server_name {string}   - MCP server name (echoed back in response)
 *   fields      {Array}    - Field definitions:
 *     id          {string}  - parameter key
 *     label       {string}  - Display label
 *     type        {string}  - "text" | "number" | "select" | "textarea"
 *     required    {boolean}
 *     value       {any}     - Pre-filled value (null if empty)
 *     placeholder {string}
 *     options     {string[]} - Only for type="select"
 */
export default function DynamicForm() {
    // Chainlit injects a global `props` object — do NOT use function argument destructuring
    // (the renderer does not pass props as React component args, same pattern as JiraTicket.jsx)
    const title = props.title || "Provide Details";
    const description = props.description || "";
    const tool = props.tool || "";
    const server_name = props.server_name || "";
    const fields = props.fields || [];

    // Initialise values from props (pre-filled by LLM where possible)
    const initValues = {};
    fields.forEach((f) => {
        initValues[f.id] =
            f.value !== null && f.value !== undefined ? String(f.value) : "";
    });

    const [values, setValues] = useState(initValues);
    const [errors, setErrors] = useState({});
    const [submitting, setSubmitting] = useState(false);

    // ── Styles ──────────────────────────────────────────────────────────────
    const colors = {
        bg: "#FFFFFF",
        bgCard: "#F9FAFB",
        border: "#E5E7EB",
        borderFocus: "#6366F1",
        text: "#111827",
        textMuted: "#6B7280",
        textLabel: "#374151",
        error: "#EF4444",
        accent: "#CC0000",       /* red — matches global button override */
        accentHover: "#A30000",
        accentText: "#FFFFFF",
        badgeBg: "#EFF6FF",
        badgeText: "#1D4ED8",
    };

    const inputBase = (hasError) => ({
        width: "100%",
        padding: "9px 12px",
        borderRadius: "6px",
        border: `1px solid ${hasError ? colors.error : colors.border}`,
        background: colors.bg,
        color: colors.text,
        fontSize: "14px",
        outline: "none",
        boxSizing: "border-box",
        fontFamily: "inherit",
        transition: "border-color 0.15s",
    });

    // ── Event handlers ───────────────────────────────────────────────────────
    const handleChange = (id, value) => {
        setValues((prev) => ({ ...prev, [id]: value }));
        if (errors[id]) setErrors((prev) => ({ ...prev, [id]: null }));
    };

    const validate = () => {
        const errs = {};
        fields.forEach((f) => {
            if (f.required && (!values[f.id] || String(values[f.id]).trim() === "")) {
                errs[f.id] = `${f.label} is required`;
            }
        });
        return errs;
    };

    const handleSubmit = () => {
        const errs = validate();
        if (Object.keys(errs).length > 0) {
            setErrors(errs);
            return;
        }
        setSubmitting(true);
        // submitElement is injected by Chainlit's frontend runtime
        submitElement({ ...values, tool, server_name, submitted: true });
    };

    const handleCancel = () => {
        // cancelElement is injected by Chainlit's frontend runtime
        cancelElement();
    };

    // ── Field renderers ──────────────────────────────────────────────────────
    const renderField = (field) => {
        const { id, label, type, required, placeholder, options } = field;
        const hasError = !!errors[id];
        const style = inputBase(hasError);

        let input;
        switch (type) {
            case "select":
                input = (
                    <select
                        value={values[id]}
                        onChange={(e) => handleChange(id, e.target.value)}
                        style={{ ...style, cursor: "pointer" }}
                    >
                        <option value="">Select {label}…</option>
                        {(options || []).map((opt) => (
                            <option key={opt} value={opt}>
                                {opt}
                            </option>
                        ))}
                    </select>
                );
                break;

            case "textarea":
                input = (
                    <textarea
                        value={values[id]}
                        onChange={(e) => handleChange(id, e.target.value)}
                        placeholder={placeholder || `Enter ${label}…`}
                        rows={4}
                        style={{ ...style, resize: "vertical" }}
                    />
                );
                break;

            case "number":
                input = (
                    <input
                        type="number"
                        value={values[id]}
                        onChange={(e) => handleChange(id, e.target.value)}
                        placeholder={placeholder || `Enter ${label}…`}
                        style={style}
                    />
                );
                break;

            default: // "text"
                input = (
                    <input
                        type="text"
                        value={values[id]}
                        onChange={(e) => handleChange(id, e.target.value)}
                        placeholder={placeholder || `Enter ${label}…`}
                        style={style}
                    />
                );
        }

        return (
            <div key={id} style={{ marginBottom: "18px" }}>
                <label
                    style={{
                        display: "block",
                        marginBottom: "6px",
                        fontSize: "13px",
                        fontWeight: "500",
                        color: colors.textLabel,
                    }}
                >
                    {label}
                    {required && (
                        <span style={{ color: colors.error, marginLeft: "4px" }}>*</span>
                    )}
                </label>
                {input}
                {hasError && (
                    <p
                        style={{
                            color: colors.error,
                            fontSize: "12px",
                            marginTop: "4px",
                            margin: "4px 0 0 0",
                        }}
                    >
                        {errors[id]}
                    </p>
                )}
            </div>
        );
    };

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div
            style={{
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
                background: colors.bg,
                borderRadius: "10px",
                border: `1.5px solid ${colors.border}`,
                borderTop: `4px solid ${colors.accent}`,
                padding: "24px",
                maxWidth: "480px",
                boxShadow: "0 4px 20px rgba(204,0,0,0.10)",
                color: colors.text,
            }}
        >
            {/* Header */}
            <div style={{ marginBottom: "16px" }}>
                <h3
                    style={{
                        margin: "0 0 6px 0",
                        fontSize: "17px",
                        fontWeight: "600",
                        color: colors.text,
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                    }}
                >
                    ⚙️ {title}
                </h3>

                {description && (
                    <p
                        style={{
                            margin: "0 0 10px 0",
                            fontSize: "13px",
                            color: colors.textMuted,
                            lineHeight: "1.5",
                        }}
                    >
                        {description}
                    </p>
                )}

                {server_name && (
                    <span
                        style={{
                            display: "inline-block",
                            padding: "2px 10px",
                            borderRadius: "4px",
                            background: colors.badgeBg,
                            color: colors.badgeText,
                            fontSize: "11px",
                            fontWeight: "500",
                            letterSpacing: "0.3px",
                        }}
                    >
                        🔧 {server_name}
                    </span>
                )}
            </div>

            {/* Divider */}
            <div
                style={{
                    height: "1px",
                    background: colors.border,
                    marginBottom: "20px",
                }}
            />

            {/* Fields */}
            <div>{fields.map((f) => renderField(f))}</div>

            {/* Buttons */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "flex-end",
                    gap: "10px",
                    marginTop: "20px",
                    paddingTop: "16px",
                    borderTop: `1px solid ${colors.border}`,
                }}
            >
                <button
                    onClick={handleCancel}
                    disabled={submitting}
                    style={{
                        padding: "8px 20px",
                        borderRadius: "6px",
                        border: `1.5px solid ${colors.border}`,
                        background: "transparent",
                        color: colors.textMuted,
                        fontSize: "14px",
                        cursor: submitting ? "not-allowed" : "pointer",
                        transition: "all 0.15s",
                    }}
                    onMouseOver={(e) => { e.currentTarget.style.borderColor = colors.accent; e.currentTarget.style.color = colors.accent; }}
                    onMouseOut={(e) => { e.currentTarget.style.borderColor = colors.border; e.currentTarget.style.color = colors.textMuted; }}
                >
                    Cancel
                </button>

                <button
                    onClick={handleSubmit}
                    disabled={submitting}
                    style={{
                        padding: "8px 22px",
                        borderRadius: "6px",
                        border: "none",
                        background: submitting ? "#E0E0E0" : colors.accent,
                        color: submitting ? colors.textMuted : colors.accentText,
                        fontSize: "14px",
                        fontWeight: "600",
                        cursor: submitting ? "not-allowed" : "pointer",
                        boxShadow: submitting ? "none" : "0 2px 8px rgba(204,0,0,0.3)",
                        transition: "all 0.15s",
                    }}
                    onMouseOver={(e) => { if (!submitting) e.currentTarget.style.background = colors.accentHover; }}
                    onMouseOut={(e) => { if (!submitting) e.currentTarget.style.background = colors.accent; }}
                >
                    {submitting ? "Submitting…" : "Submit"}
                </button>
            </div>
        </div>
    );
}
