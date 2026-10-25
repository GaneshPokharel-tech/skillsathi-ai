(() => {
  "use strict";

  const script = document.currentScript;
  if (!script || !script.src) return;

  const chatbotUrl = new URL("/", script.src);
  chatbotUrl.searchParams.set("widget_version", "logo-20261025-v2");
  const widgetId = "skillshikshya-chat-widget";

  function mount() {
    if (document.getElementById(widgetId)) return;

    const host = document.createElement("div");
    host.id = widgetId;

    const shadow = host.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = `
      :host {
        all: initial;
        position: fixed;
        inset: 0;
        z-index: 2147483000;
        pointer-events: none;
        font-family: system-ui, -apple-system, sans-serif;
      }

      button {
        font: inherit;
        cursor: pointer;
      }

      .launcher {
        position: fixed;
        right: 24px;
        bottom: 24px;
        min-height: 58px;
        border: 0;
        border-radius: 999px;
        padding: 0 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        background: #087d5d;
        color: white;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 0 8px 28px #003b2d40;
        pointer-events: auto;
      }

      .launcher:hover {
        background: #05694e;
      }

      .launcher:focus-visible,
      .close:focus-visible {
        outline: 3px solid #f4b942;
        outline-offset: 3px;
      }

      .launcher-icon {
        font-size: 23px;
        line-height: 1;
      }

      .panel {
        position: fixed;
        right: 18px;
        bottom: 94px;
        width: min(440px, calc(100vw - 36px));
        height: min(730px, calc(100dvh - 112px));
        display: none;
        flex-direction: column;
        overflow: hidden;
        background: white;
        border: 1px solid #d7e7df;
        border-radius: 18px;
        box-shadow: 0 16px 60px #092d2340;
        pointer-events: auto;
      }

      .panel[data-open="true"] {
        display: flex;
      }

      .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex: 0 0 auto;
        min-height: 45px;
        padding: 8px 14px;
        background: #065c47;
        color: white;
      }

      .panel-title {
        font-size: 14px;
        font-weight: 700;
      }

      .close {
        min-width: 36px;
        min-height: 36px;
        border: 0;
        border-radius: 10px;
        background: #ffffff26;
        color: white;
        font-size: 22px;
      }

      .frame {
        width: 100%;
        flex: 1 1 auto;
        min-height: 0;
        border: 0;
        background: #fff;
      }

      @media (max-width: 640px) {
        .launcher {
          right: 16px;
          bottom: 16px;
        }

        .launcher[data-open="true"] {
          display: none;
        }

        .panel {
          inset: 0;
          width: 100vw;
          height: 100dvh;
          border: 0;
          border-radius: 0;
        }
      }
    `;

    const launcher = document.createElement("button");
    launcher.type = "button";
    launcher.className = "launcher";
    launcher.setAttribute("aria-expanded", "false");
    launcher.setAttribute("aria-label", "Open SkillShikshya chat");

    const icon = document.createElement("span");
    icon.className = "launcher-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "✦";

    const label = document.createElement("span");
    label.textContent = "Ask SkillShikshya";

    launcher.append(icon, label);

    const panel = document.createElement("section");
    panel.className = "panel";
    panel.setAttribute("data-open", "false");
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "SkillShikshya chatbot");

    const header = document.createElement("div");
    header.className = "panel-header";

    const title = document.createElement("span");
    title.className = "panel-title";
    title.textContent = "SkillShikshya Guide";

    const close = document.createElement("button");
    close.type = "button";
    close.className = "close";
    close.textContent = "×";
    close.setAttribute("aria-label", "Close chatbot");

    header.append(title, close);

    const frame = document.createElement("iframe");
    frame.className = "frame";
    frame.title = "SkillShikshya learning assistant";
    frame.setAttribute(
      "referrerpolicy",
      "strict-origin-when-cross-origin"
    );

    panel.append(header, frame);
    shadow.append(style, launcher, panel);
    document.body.append(host);

    let frameLoaded = false;

    function setOpen(open) {
      if (open && !frameLoaded) {
        frame.src = chatbotUrl.href;
        frameLoaded = true;
      }

      panel.dataset.open = String(open);
      launcher.dataset.open = String(open);
      launcher.setAttribute("aria-expanded", String(open));
      launcher.setAttribute(
        "aria-label",
        open ? "Close SkillShikshya chat" : "Open SkillShikshya chat"
      );

      if (open) {
        close.focus();
      } else {
        launcher.focus();
      }
    }

    launcher.addEventListener("click", () => {
      setOpen(panel.dataset.open !== "true");
    });

    close.addEventListener("click", () => setOpen(false));

    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && panel.dataset.open === "true") {
        setOpen(false);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount, {
      once: true
    });
  } else {
    mount();
  }
})();
